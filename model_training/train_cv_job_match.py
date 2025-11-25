"""
CV-Job Match Model Training Script
Train semantic matching models for CV and Job Description pairs.

Usage:
    python train_cv_job_match.py --model_type bert --epochs 5 --batch_size 8
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
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from models.cv_job_match_model import get_cv_job_match_model


class CVJobMatchDataset(Dataset):
    """Dataset for CV-Job matching."""

    def __init__(self, data_df, tokenizer=None, max_length=512, is_siamese=False):
        """
        Args:
            data_df: DataFrame with CV and job description texts
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
            is_siamese: Whether using siamese architecture (separate encoding)
        """
        self.data_df = data_df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_siamese = is_siamese

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        row = self.data_df.iloc[idx]
        cv_text = row['resume_text_clean']
        job_text = row['job_description_text_clean']
        label = row['label_id']

        if self.is_siamese:
            # Separate encoding for siamese models
            cv_encoding = self.tokenizer(
                cv_text,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )

            job_encoding = self.tokenizer(
                job_text,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )

            return {
                'cv_input_ids': cv_encoding['input_ids'].squeeze(0),
                'cv_attention_mask': cv_encoding['attention_mask'].squeeze(0),
                'job_input_ids': job_encoding['input_ids'].squeeze(0),
                'job_attention_mask': job_encoding['attention_mask'].squeeze(0),
                'labels': torch.tensor(label, dtype=torch.long)
            }
        else:
            # Combined encoding for BERT sentence pair classification
            encoding = self.tokenizer(
                cv_text,
                job_text,
                padding='max_length',
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt'
            )

            return {
                'input_ids': encoding['input_ids'].squeeze(0),
                'attention_mask': encoding['attention_mask'].squeeze(0),
                'token_type_ids': encoding.get('token_type_ids', torch.zeros_like(encoding['input_ids'])).squeeze(0),
                'labels': torch.tensor(label, dtype=torch.long)
            }


def load_data(data_dir: str):
    """Load preprocessed CV-Job match data."""
    data_dir = Path(data_dir)

    print(f"Loading data from {data_dir}...")

    train_df = pd.read_csv(data_dir / 'cv_job_match_train.csv')
    val_df = pd.read_csv(data_dir / 'cv_job_match_val.csv')
    test_df = pd.read_csv(data_dir / 'cv_job_match_test.csv')

    with open(data_dir / 'cv_job_match_metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)

    print(f"  Train samples: {len(train_df)}")
    print(f"  Val samples: {len(val_df)}")
    print(f"  Test samples: {len(test_df)}")
    print(f"  Number of classes: {metadata['num_classes']}")
    print(f"  Class weights: {metadata['class_weights']}")

    return train_df, val_df, test_df, metadata


def compute_metrics(predictions, labels, id_to_label):
    """
    Compute classification metrics.

    Args:
        predictions: Predicted label IDs
        labels: Ground truth label IDs
        id_to_label: ID to label mapping

    Returns:
        Dictionary of metrics
    """
    # Compute metrics
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted', zero_division=0
    )

    # Per-class metrics
    label_names = [id_to_label[i] for i in sorted(id_to_label.keys())]
    report = classification_report(
        labels, predictions,
        target_names=label_names,
        output_dict=True,
        zero_division=0
    )

    # Confusion matrix
    cm = confusion_matrix(labels, predictions)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'classification_report': report,
        'confusion_matrix': cm.tolist()
    }


def train_epoch(model, dataloader, optimizer, scheduler, device, class_weights, is_siamese):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_predictions = []
    all_labels = []

    # Loss function with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    progress_bar = tqdm(dataloader, desc="Training")

    for batch in progress_bar:
        labels = batch['labels'].to(device)

        if is_siamese:
            # Siamese model
            outputs = model(
                cv_input_ids=batch['cv_input_ids'].to(device),
                cv_attention_mask=batch['cv_attention_mask'].to(device),
                job_input_ids=batch['job_input_ids'].to(device),
                job_attention_mask=batch['job_attention_mask'].to(device),
                labels=labels
            )
        else:
            # Standard classification model
            token_type_ids = batch.get('token_type_ids', None)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            outputs = model(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device),
                token_type_ids=token_type_ids,
                labels=labels
            )

        loss = outputs['loss'] if outputs['loss'] is not None else criterion(outputs['logits'], labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        # Get predictions
        predictions = torch.argmax(outputs['logits'], dim=-1)
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        progress_bar.set_postfix({'loss': loss.item()})

    avg_loss = total_loss / len(dataloader)
    return avg_loss, np.array(all_predictions), np.array(all_labels)


def evaluate(model, dataloader, device, id_to_label, is_siamese):
    """Evaluate model on validation/test set."""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            labels = batch['labels'].to(device)

            if is_siamese:
                outputs = model(
                    cv_input_ids=batch['cv_input_ids'].to(device),
                    cv_attention_mask=batch['cv_attention_mask'].to(device),
                    job_input_ids=batch['job_input_ids'].to(device),
                    job_attention_mask=batch['job_attention_mask'].to(device),
                    labels=labels
                )
            else:
                token_type_ids = batch.get('token_type_ids', None)
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(device)

                outputs = model(
                    input_ids=batch['input_ids'].to(device),
                    attention_mask=batch['attention_mask'].to(device),
                    token_type_ids=token_type_ids,
                    labels=labels
                )

            loss = outputs['loss'] if outputs['loss'] is not None else criterion(outputs['logits'], labels)
            total_loss += loss.item()

            # Get predictions
            predictions = torch.argmax(outputs['logits'], dim=-1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    metrics = compute_metrics(all_predictions, all_labels, id_to_label)

    return avg_loss, metrics


def train(args):
    """Main training function."""
    print("=" * 80)
    print("CV-Job Match Model Training")
    print("=" * 80)
    print(f"\nConfiguration:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Load data
    train_df, val_df, test_df, metadata = load_data(args.data_dir)
    label_to_id = metadata['label_to_id']
    id_to_label = metadata['id_to_label']
    num_classes = metadata['num_classes']
    class_weights = metadata['class_weights']

    # Convert class weights to tensor
    class_weight_tensor = torch.tensor(
        [class_weights[i] for i in range(num_classes)],
        dtype=torch.float32
    ).to(device)

    # Determine if siamese architecture
    is_siamese = args.model_type in ['lstm-siamese', 'sbert']

    # Initialize tokenizer (use Fast tokenizers)
    tokenizer = None
    if args.model_type == 'bert':
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    elif args.model_type == 'roberta':
        tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
    elif args.model_type == 'sbert':
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    # Create datasets
    train_dataset = CVJobMatchDataset(
        train_df,
        tokenizer=tokenizer,
        max_length=args.max_length,
        is_siamese=is_siamese
    )

    val_dataset = CVJobMatchDataset(
        val_df,
        tokenizer=tokenizer,
        max_length=args.max_length,
        is_siamese=is_siamese
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    # Initialize model
    print(f"\nInitializing {args.model_type} model...")
    model_kwargs = {'num_classes': num_classes}

    if args.model_type == 'lstm-siamese':
        model_kwargs.update({
            'vocab_size': args.vocab_size,
            'embedding_dim': args.embedding_dim,
            'hidden_dim': args.hidden_dim,
            'num_layers': args.num_lstm_layers,
            'dropout': args.dropout
        })

    model = get_cv_job_match_model(args.model_type, **model_kwargs)
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

    # Organize models by task type (cv_jd_matching) and then model type
    output_dir = Path(args.output_dir) / 'cv_jd_matching' / args.model_type
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nModel will be saved to: {output_dir}")

    for epoch in range(args.epochs):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'='*80}")

        start_time = time.time()

        # Train
        train_loss, train_preds, train_labels = train_epoch(
            model, train_loader, optimizer, scheduler, device, class_weight_tensor, is_siamese
        )
        train_metrics = compute_metrics(train_preds, train_labels, id_to_label)

        # Validate
        val_loss, val_metrics = evaluate(model, val_loader, device, id_to_label, is_siamese)

        elapsed = time.time() - start_time

        # Print metrics
        print(f"\nEpoch {epoch + 1} Results:")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Train Loss: {train_loss:.4f}, Acc: {train_metrics['accuracy']:.4f}, F1: {train_metrics['f1']:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}")
        print(f"  Val Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}")

        # Save history
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_accuracy': float(train_metrics['accuracy']),
            'train_f1': float(train_metrics['f1']),
            'val_loss': val_loss,
            'val_accuracy': float(val_metrics['accuracy']),
            'val_f1': float(val_metrics['f1']),
            'val_precision': float(val_metrics['precision']),
            'val_recall': float(val_metrics['recall'])
        })

        # Save best model
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            print(f"\n  New best F1: {best_f1:.4f}! Saving model...")

            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1,
                'val_metrics': {k: v for k, v in val_metrics.items() if k != 'classification_report'}
            }, output_dir / 'best_model.pt')

            # Save classification report
            with open(output_dir / 'best_classification_report.json', 'w') as f:
                json.dump(val_metrics['classification_report'], f, indent=2)

        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, output_dir / f'checkpoint_epoch_{epoch + 1}.pt')

    # Save training history
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(training_history, f, indent=2)

    print(f"\n{'='*80}")
    print("Training completed!")
    print(f"Best validation F1: {best_f1:.4f}")
    print(f"Model saved to: {output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CV-Job Match model")

    # Data arguments
    parser.add_argument('--data_dir', type=str, default='pre-processed-data/cv_job_match',
                        help='Directory containing preprocessed data')
    parser.add_argument('--output_dir', type=str, default='trained_models',
                        help='Directory to save trained models')

    # Model arguments
    parser.add_argument('--model_type', type=str, default='bert',
                        choices=['lstm-siamese', 'bert', 'roberta', 'sbert'],
                        help='Type of model to train')
    parser.add_argument('--max_length', type=int, default=256,
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
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout probability')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--save_every', type=int, default=2,
                        help='Save checkpoint every N epochs')

    args = parser.parse_args()

    train(args)
