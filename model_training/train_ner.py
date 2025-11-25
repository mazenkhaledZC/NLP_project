"""
NER Model Training Script
Train Named Entity Recognition models for CV field extraction.

Usage:
    python train_ner.py --model_type bert --epochs 10 --batch_size 16
"""

import os
import sys
import argparse
import pickle
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    BertTokenizerFast,
    RobertaTokenizerFast,
    DistilBertTokenizerFast,
    get_linear_schedule_with_warmup
)
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, classification_report
from collections import Counter

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from models.ner_model import get_ner_model


def build_vocabulary(tokens_list, vocab_size=30000):
    """
    Build vocabulary from token lists.

    Args:
        tokens_list: List of token lists
        vocab_size: Maximum vocabulary size

    Returns:
        word2id: Dictionary mapping tokens to IDs
        id2word: Dictionary mapping IDs to tokens
    """
    # Count token frequencies
    token_counter = Counter()
    for tokens in tokens_list:
        token_counter.update(tokens)

    # Get most common tokens
    # Reserve IDs: 0=PAD, 1=UNK
    most_common = token_counter.most_common(vocab_size - 2)

    # Build mappings
    word2id = {'<PAD>': 0, '<UNK>': 1}
    id2word = {0: '<PAD>', 1: '<UNK>'}

    for idx, (token, _) in enumerate(most_common, start=2):
        word2id[token] = idx
        id2word[idx] = token

    print(f"Built vocabulary with {len(word2id)} tokens")
    print(f"  Coverage: {len(most_common)}/{len(token_counter)} unique tokens")

    return word2id, id2word


class NERDataset(Dataset):
    """Dataset for NER with BIO tags."""

    def __init__(self, tokens_list, tags_list, tokenizer=None, max_length=512, label_to_id=None, word2id=None):
        """
        Args:
            tokens_list: List of token lists
            tags_list: List of tag ID lists
            tokenizer: HuggingFace tokenizer (for BERT-based models)
            max_length: Maximum sequence length
            label_to_id: Label to ID mapping
            word2id: Word to ID mapping (for LSTM-based models)
        """
        self.tokens_list = tokens_list
        self.tags_list = tags_list
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_to_id = label_to_id
        self.word2id = word2id

    def __len__(self):
        return len(self.tokens_list)

    def __getitem__(self, idx):
        tokens = self.tokens_list[idx]
        tags = self.tags_list[idx]

        if self.tokenizer is not None:
            # Clean tokens to remove invalid Unicode characters (e.g., lone surrogates)
            # Lone surrogates (like \ud83d) cause issues with the tokenizer's Rust backend
            cleaned_tokens = []
            for token in tokens:
                try:
                    # Encode to UTF-8 with surrogatepass, then decode with replace
                    # This handles lone surrogates by replacing them
                    cleaned_token = token.encode('utf-8', errors='surrogatepass').decode('utf-8', errors='replace')
                    # Remove replacement characters
                    cleaned_token = cleaned_token.replace('\ufffd', '')
                    if cleaned_token.strip():
                        cleaned_tokens.append(cleaned_token)
                    else:
                        cleaned_tokens.append('[UNK]')
                except Exception:
                    cleaned_tokens.append('[UNK]')

            # Use transformer tokenizer
            encoding = self.tokenizer(
                cleaned_tokens,
                is_split_into_words=True,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )

            # Align labels with subword tokens
            word_ids = encoding.word_ids(batch_index=0)
            aligned_labels = []
            previous_word_idx = None

            for word_idx in word_ids:
                if word_idx is None:
                    aligned_labels.append(-100)  # Ignore padding
                elif word_idx != previous_word_idx:
                    aligned_labels.append(tags[word_idx] if word_idx < len(tags) else -100)
                else:
                    # For subword tokens, use the same label or -100
                    aligned_labels.append(-100)
                previous_word_idx = word_idx

            return {
                'input_ids': encoding['input_ids'].squeeze(0),
                'attention_mask': encoding['attention_mask'].squeeze(0),
                'labels': torch.tensor(aligned_labels, dtype=torch.long)
            }
        else:
            # For LSTM-based models, convert tokens to IDs using vocabulary
            if self.word2id is None:
                raise ValueError(
                    "word2id vocabulary is required for LSTM models. "
                    "Please provide a word2id mapping when creating the dataset."
                )

            # Convert tokens to IDs
            token_ids = []
            for token in tokens[:self.max_length]:
                # Use <UNK> (ID=1) for unknown tokens
                token_id = self.word2id.get(token, 1)
                token_ids.append(token_id)

            # Track actual sequence length for attention mask
            actual_length = len(token_ids)

            # Pad sequences to max_length
            if len(token_ids) < self.max_length:
                # Pad with <PAD> (ID=0)
                token_ids.extend([0] * (self.max_length - len(token_ids)))
                # Pad labels with 2 (CRF doesn't support -100)
                # Use 2 which corresponds to 'O' (Outside) tag - safe for padding
                tags = list(tags[:self.max_length]) + [2] * (self.max_length - len(tags))
            else:
                tags = tags[:self.max_length]

            # Create attention mask: 1 for real tokens, 0 for padding
            attention_mask = [1] * actual_length + [0] * (self.max_length - actual_length)

            return {
                'input_ids': torch.tensor(token_ids, dtype=torch.long),
                'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
                'labels': torch.tensor(tags, dtype=torch.long)
            }


def load_data(data_dir: str):
    """Load preprocessed NER data."""
    data_dir = Path(data_dir)

    print(f"Loading data from {data_dir}...")

    with open(data_dir / 'ner_train.pkl', 'rb') as f:
        train_data = pickle.load(f)

    with open(data_dir / 'ner_val.pkl', 'rb') as f:
        val_data = pickle.load(f)

    with open(data_dir / 'ner_test.pkl', 'rb') as f:
        test_data = pickle.load(f)

    with open(data_dir / 'ner_label_mappings.pkl', 'rb') as f:
        label_mappings = pickle.load(f)

    print(f"  Train samples: {len(train_data['tokens'])}")
    print(f"  Val samples: {len(val_data['tokens'])}")
    print(f"  Test samples: {len(test_data['tokens'])}")
    print(f"  Number of labels: {len(label_mappings['label_to_id'])}")

    return train_data, val_data, test_data, label_mappings


def compute_metrics(predictions, labels, id_to_label):
    """
    Compute NER metrics (precision, recall, F1).

    Args:
        predictions: Predicted label IDs
        labels: Ground truth label IDs
        id_to_label: ID to label mapping

    Returns:
        Dictionary of metrics
    """
    # Flatten predictions and labels, ignoring -100
    pred_flat = []
    label_flat = []

    for pred_seq, label_seq in zip(predictions, labels):
        for p, l in zip(pred_seq, label_seq):
            if l != -100:
                pred_flat.append(p)
                label_flat.append(l)

    # Convert to numpy
    pred_flat = np.array(pred_flat)
    label_flat = np.array(label_flat)

    # Compute metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        label_flat, pred_flat, average='weighted', zero_division=0
    )

    # Per-class metrics
    label_names = [id_to_label[i] for i in sorted(id_to_label.keys())]
    report = classification_report(
        label_flat, pred_flat,
        target_names=label_names,
        output_dict=True,
        zero_division=0
    )

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'classification_report': report
    }


def train_epoch(model, dataloader, optimizer, scheduler, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_predictions = []
    all_labels = []

    progress_bar = tqdm(dataloader, desc="Training")

    for batch in progress_bar:
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        attention_mask = batch.get('attention_mask')
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs['loss']

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        # Get predictions
        if 'logits' in outputs:
            # BERT-based models return logits
            predictions = torch.argmax(outputs['logits'], dim=-1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        elif 'predictions' in outputs:
            # CRF-based models return decoded predictions as list of lists
            predictions = outputs['predictions']
            # Convert to numpy arrays with same shape as labels
            for pred_seq, label_seq in zip(predictions, labels.cpu().numpy()):
                # Pad predictions to match label sequence length
                pred_padded = pred_seq + [2] * (len(label_seq) - len(pred_seq))
                all_predictions.append(pred_padded[:len(label_seq)])
                all_labels.append(label_seq)

        progress_bar.set_postfix({'loss': loss.item()})

    avg_loss = total_loss / len(dataloader)
    return avg_loss, all_predictions, all_labels


def evaluate(model, dataloader, device, id_to_label):
    """Evaluate model on validation/test set."""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch.get('attention_mask')
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            if outputs['loss'] is not None:
                total_loss += outputs['loss'].item()

            # Get predictions
            if 'logits' in outputs:
                # BERT-based models return logits
                predictions = torch.argmax(outputs['logits'], dim=-1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            elif 'predictions' in outputs:
                # CRF-based models return decoded predictions as list of lists
                predictions = outputs['predictions']
                # Convert to numpy arrays with same shape as labels
                for pred_seq, label_seq in zip(predictions, labels.cpu().numpy()):
                    # Pad predictions to match label sequence length
                    pred_padded = pred_seq + [2] * (len(label_seq) - len(pred_seq))
                    all_predictions.append(pred_padded[:len(label_seq)])
                    all_labels.append(label_seq)

    avg_loss = total_loss / len(dataloader)
    metrics = compute_metrics(all_predictions, all_labels, id_to_label)

    return avg_loss, metrics


def train(args):
    """Main training function."""
    print("=" * 80)
    print("NER Model Training")
    print("=" * 80)
    print(f"\nConfiguration:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Load data
    train_data, val_data, test_data, label_mappings = load_data(args.data_dir)
    label_to_id = label_mappings['label_to_id']
    id_to_label = label_mappings['id_to_label']
    num_labels = len(label_to_id)

    # Initialize tokenizer for transformer models (use Fast tokenizers)
    tokenizer = None
    word2id = None
    id2word = None

    if args.model_type in ['bert', 'roberta', 'distilbert']:
        if args.model_type == 'bert':
            tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        elif args.model_type == 'roberta':
            # RoBERTa requires add_prefix_space=True for pre-tokenized inputs
            tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base', add_prefix_space=True)
        elif args.model_type == 'distilbert':
            tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    else:
        # Build vocabulary for LSTM-based models
        print(f"\nBuilding vocabulary from training data (vocab_size={args.vocab_size})...")
        word2id, id2word = build_vocabulary(train_data['tokens'], vocab_size=args.vocab_size)

    # Create datasets
    train_dataset = NERDataset(
        train_data['tokens'],
        train_data['tags'],
        tokenizer=tokenizer,
        max_length=args.max_length,
        label_to_id=label_to_id,
        word2id=word2id
    )

    val_dataset = NERDataset(
        val_data['tokens'],
        val_data['tags'],
        tokenizer=tokenizer,
        max_length=args.max_length,
        label_to_id=label_to_id,
        word2id=word2id
    )

    # Create dataloaders
    # Note: num_workers=0 to avoid multiprocessing issues with tokenizers and tensor conversion
    # Multiprocessing can cause issues with tokenizer serialization and data type conversions
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    # Initialize model
    print(f"\nInitializing {args.model_type} model...")
    model_kwargs = {'num_labels': num_labels}

    if args.model_type == 'bilstm-crf':
        # Use actual vocabulary size from built vocabulary
        actual_vocab_size = len(word2id) if word2id is not None else args.vocab_size
        model_kwargs.update({
            'vocab_size': actual_vocab_size,
            'embedding_dim': args.embedding_dim,
            'hidden_dim': args.hidden_dim,
            'num_layers': args.num_lstm_layers,
            'dropout': args.dropout
        })
        print(f"  Using vocabulary size: {actual_vocab_size}")

    model = get_ner_model(args.model_type, **model_kwargs)
    model = model.to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    best_f1 = 0
    training_history = []

    # Organize models by task type (ner) and then model type
    output_dir = Path(args.output_dir) / 'ner' / args.model_type
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nModel will be saved to: {output_dir}")

    for epoch in range(args.epochs):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'='*80}")

        start_time = time.time()

        # Train
        train_loss, train_preds, train_labels = train_epoch(
            model, train_loader, optimizer, scheduler, device
        )
        train_metrics = compute_metrics(train_preds, train_labels, id_to_label)

        # Validate
        val_loss, val_metrics = evaluate(model, val_loader, device, id_to_label)

        elapsed = time.time() - start_time

        # Print metrics
        print(f"\nEpoch {epoch + 1} Results:")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Train Loss: {train_loss:.4f}, F1: {train_metrics['f1']:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, F1: {val_metrics['f1']:.4f}")
        print(f"  Val Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}")

        # Save history
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_f1': train_metrics['f1'],
            'val_loss': val_loss,
            'val_f1': val_metrics['f1'],
            'val_precision': val_metrics['precision'],
            'val_recall': val_metrics['recall']
        })

        # Save best model
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            print(f"\n  New best F1: {best_f1:.4f}! Saving model...")

            # Save model checkpoint
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1,
                'val_metrics': val_metrics,
                'model_type': args.model_type,
                'num_labels': num_labels,
                'label_to_id': label_to_id,
                'id_to_label': id_to_label
            }

            # Add vocabulary for LSTM models
            if word2id is not None:
                checkpoint['word2id'] = word2id
                checkpoint['id2word'] = id2word

            torch.save(checkpoint, output_dir / 'best_model.pt')

        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'model_type': args.model_type,
                'num_labels': num_labels
            }
            # Add vocabulary for LSTM models
            if word2id is not None:
                checkpoint['word2id'] = word2id
                checkpoint['id2word'] = id2word

            torch.save(checkpoint, output_dir / f'checkpoint_epoch_{epoch + 1}.pt')

    # Save training history
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(training_history, f, indent=2)

    print(f"\n{'='*80}")
    print("Training completed!")
    print(f"Best validation F1: {best_f1:.4f}")
    print(f"Model saved to: {output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NER model")

    # Data arguments
    parser.add_argument('--data_dir', type=str, default='pre-processed-data/ner',
                        help='Directory containing preprocessed data')
    parser.add_argument('--output_dir', type=str, default='trained_models',
                        help='Directory to save trained models')

    # Model arguments
    parser.add_argument('--model_type', type=str, default='bert',
                        choices=['bilstm-crf', 'bert', 'roberta', 'distilbert'],
                        help='Type of model to train')
    parser.add_argument('--max_length', type=int, default=512,
                        help='Maximum sequence length')

    # LSTM-specific arguments
    parser.add_argument('--vocab_size', type=int, default=30000,
                        help='Vocabulary size for LSTM models')
    parser.add_argument('--embedding_dim', type=int, default=300,
                        help='Embedding dimension for LSTM')
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='Hidden dimension for LSTM')
    parser.add_argument('--num_lstm_layers', type=int, default=2,
                        help='Number of LSTM layers')

    # Training arguments
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout probability')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--save_every', type=int, default=5,
                        help='Save checkpoint every N epochs')

    args = parser.parse_args()

    train(args)
