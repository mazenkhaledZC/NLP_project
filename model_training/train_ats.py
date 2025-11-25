"""
ATS Model Training Script
Train ATS score prediction models (regression task).

Usage:
    python train_ats.py --model_type bert --epochs 5 --batch_size 8
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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from models.ats_model import get_ats_model


class ATSDataset(Dataset):
    """Dataset for ATS score prediction."""

    def __init__(self, data_df, tokenizer, max_length=512, use_normalized_scores=True):
        """
        Args:
            data_df: DataFrame with text and ATS scores
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
            use_normalized_scores: Use normalized scores (0-1) instead of original
        """
        self.data_df = data_df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_normalized_scores = use_normalized_scores

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        row = self.data_df.iloc[idx]
        text = row['text_clean']

        # Get score
        if self.use_normalized_scores and 'ats_score_normalized' in row:
            score = row['ats_score_normalized']
        else:
            score = row['ats_score']

        # Tokenize
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(score, dtype=torch.float32)
        }


class ATSSiameseDataset(Dataset):
    """Dataset for ATS with siamese models (separate resume and job encoding)."""

    def __init__(self, data_df, tokenizer, max_length=512, use_normalized_scores=True):
        """
        Args:
            data_df: DataFrame with separate resume and job texts
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
            use_normalized_scores: Use normalized scores
        """
        self.data_df = data_df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_normalized_scores = use_normalized_scores

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        row = self.data_df.iloc[idx]

        # Split text if it contains separator (simplified - you may need to parse differently)
        text = row['text_clean']
        parts = text.split('[SEP]') if '[SEP]' in text else [text, text]

        resume_text = parts[0].strip()
        job_text = parts[1].strip() if len(parts) > 1 else parts[0].strip()

        # Get score
        if self.use_normalized_scores and 'ats_score_normalized' in row:
            score = row['ats_score_normalized']
        else:
            score = row['ats_score']

        # Tokenize
        resume_encoding = self.tokenizer(
            resume_text,
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
            'resume_input_ids': resume_encoding['input_ids'].squeeze(0),
            'resume_attention_mask': resume_encoding['attention_mask'].squeeze(0),
            'job_input_ids': job_encoding['input_ids'].squeeze(0),
            'job_attention_mask': job_encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(score, dtype=torch.float32)
        }


def load_data(data_dir: str):
    """Load preprocessed ATS data."""
    data_dir = Path(data_dir)

    print(f"Loading data from {data_dir}...")

    train_df = pd.read_csv(data_dir / 'ats_train.csv')
    val_df = pd.read_csv(data_dir / 'ats_val.csv')

    with open(data_dir / 'ats_metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)

    # Load scaler if exists
    scaler_path = data_dir / 'ats_score_scaler.pkl'
    scaler = None
    if scaler_path.exists():
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)

    print(f"  Train samples: {len(train_df)}")
    print(f"  Val samples: {len(val_df)}")
    print(f"  Score range: {metadata['score_range']}")

    return train_df, val_df, metadata, scaler


def compute_metrics(predictions, labels):
    """
    Compute regression metrics.

    Args:
        predictions: Predicted scores
        labels: Ground truth scores

    Returns:
        Dictionary of metrics
    """
    mse = mean_squared_error(labels, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(labels, predictions)
    r2 = r2_score(labels, predictions)

    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }


def train_epoch(model, dataloader, optimizer, scheduler, device, is_siamese):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_predictions = []
    all_labels = []

    criterion = nn.MSELoss()

    progress_bar = tqdm(dataloader, desc="Training")

    for batch in progress_bar:
        labels = batch['labels'].to(device)

        if is_siamese:
            # Siamese model
            outputs = model(
                resume_input_ids=batch['resume_input_ids'].to(device),
                resume_attention_mask=batch['resume_attention_mask'].to(device),
                job_input_ids=batch['job_input_ids'].to(device),
                job_attention_mask=batch['job_attention_mask'].to(device),
                labels=labels
            )
        else:
            # Standard regression model
            outputs = model(
                input_ids=batch['input_ids'].to(device),
                attention_mask=batch['attention_mask'].to(device),
                labels=labels
            )

        loss = outputs['loss'] if outputs['loss'] is not None else criterion(outputs['predictions'], labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

        # Get predictions
        predictions = outputs['predictions']
        all_predictions.extend(predictions.detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        progress_bar.set_postfix({'loss': loss.item()})

    avg_loss = total_loss / len(dataloader)
    return avg_loss, np.array(all_predictions), np.array(all_labels)


def evaluate(model, dataloader, device, is_siamese):
    """Evaluate model on validation/test set."""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []

    criterion = nn.MSELoss()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            labels = batch['labels'].to(device)

            if is_siamese:
                outputs = model(
                    resume_input_ids=batch['resume_input_ids'].to(device),
                    resume_attention_mask=batch['resume_attention_mask'].to(device),
                    job_input_ids=batch['job_input_ids'].to(device),
                    job_attention_mask=batch['job_attention_mask'].to(device),
                    labels=labels
                )
            else:
                outputs = model(
                    input_ids=batch['input_ids'].to(device),
                    attention_mask=batch['attention_mask'].to(device),
                    labels=labels
                )

            loss = outputs['loss'] if outputs['loss'] is not None else criterion(outputs['predictions'], labels)
            total_loss += loss.item()

            # Get predictions
            predictions = outputs['predictions']
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    metrics = compute_metrics(all_predictions, all_labels)

    return avg_loss, metrics


def train(args):
    """Main training function."""
    print("=" * 80)
    print("ATS Score Prediction Model Training")
    print("=" * 80)
    print(f"\nConfiguration:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Load data
    train_df, val_df, metadata, scaler = load_data(args.data_dir)

    # Determine if siamese architecture
    is_siamese = args.model_type == 'sbert'

    # Initialize tokenizer (use Fast tokenizers)
    tokenizer = None
    if args.model_type == 'bert':
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    elif args.model_type == 'roberta':
        tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
    elif args.model_type == 'sbert':
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    # Create datasets
    if is_siamese:
        train_dataset = ATSSiameseDataset(
            train_df,
            tokenizer=tokenizer,
            max_length=args.max_length,
            use_normalized_scores=args.use_normalized_scores
        )

        val_dataset = ATSSiameseDataset(
            val_df,
            tokenizer=tokenizer,
            max_length=args.max_length,
            use_normalized_scores=args.use_normalized_scores
        )
    else:
        train_dataset = ATSDataset(
            train_df,
            tokenizer=tokenizer,
            max_length=args.max_length,
            use_normalized_scores=args.use_normalized_scores
        )

        val_dataset = ATSDataset(
            val_df,
            tokenizer=tokenizer,
            max_length=args.max_length,
            use_normalized_scores=args.use_normalized_scores
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
    model_kwargs = {}

    if args.model_type == 'lstm':
        model_kwargs.update({
            'vocab_size': args.vocab_size,
            'embedding_dim': args.embedding_dim,
            'hidden_dim': args.hidden_dim,
            'num_layers': args.num_lstm_layers,
            'dropout': args.dropout,
            'use_attention': True
        })

    model = get_ats_model(args.model_type, **model_kwargs)
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
    best_rmse = float('inf')
    training_history = []

    # Organize models by task type (ats) and then model type
    output_dir = Path(args.output_dir) / 'ats' / args.model_type
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nModel will be saved to: {output_dir}")

    for epoch in range(args.epochs):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'='*80}")

        start_time = time.time()

        # Train
        train_loss, train_preds, train_labels = train_epoch(
            model, train_loader, optimizer, scheduler, device, is_siamese
        )
        train_metrics = compute_metrics(train_preds, train_labels)

        # Validate
        val_loss, val_metrics = evaluate(model, val_loader, device, is_siamese)

        elapsed = time.time() - start_time

        # Print metrics
        print(f"\nEpoch {epoch + 1} Results:")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Train Loss: {train_loss:.4f}, RMSE: {train_metrics['rmse']:.4f}, MAE: {train_metrics['mae']:.4f}, R²: {train_metrics['r2']:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, RMSE: {val_metrics['rmse']:.4f}, MAE: {val_metrics['mae']:.4f}, R²: {val_metrics['r2']:.4f}")

        # Save history
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_rmse': float(train_metrics['rmse']),
            'train_mae': float(train_metrics['mae']),
            'train_r2': float(train_metrics['r2']),
            'val_loss': val_loss,
            'val_rmse': float(val_metrics['rmse']),
            'val_mae': float(val_metrics['mae']),
            'val_r2': float(val_metrics['r2'])
        })

        # Save best model (based on RMSE)
        if val_metrics['rmse'] < best_rmse:
            best_rmse = val_metrics['rmse']
            print(f"\n  New best RMSE: {best_rmse:.4f}! Saving model...")

            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_rmse': best_rmse,
                'val_metrics': val_metrics,
                'scaler': scaler
            }, output_dir / 'best_model.pt')

        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler': scaler
            }, output_dir / f'checkpoint_epoch_{epoch + 1}.pt')

    # Save training history
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(training_history, f, indent=2)

    print(f"\n{'='*80}")
    print("Training completed!")
    print(f"Best validation RMSE: {best_rmse:.4f}")
    print(f"Model saved to: {output_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ATS model")

    # Data arguments
    parser.add_argument('--data_dir', type=str, default='pre-processed-data/ats',
                        help='Directory containing preprocessed data')
    parser.add_argument('--output_dir', type=str, default='trained_models',
                        help='Directory to save trained models')
    parser.add_argument('--use_normalized_scores', action='store_true', default=True,
                        help='Use normalized scores (0-1) for training')

    # Model arguments
    parser.add_argument('--model_type', type=str, default='bert',
                        choices=['lstm', 'bert', 'roberta', 'sbert'],
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
