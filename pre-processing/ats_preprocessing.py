"""
ATS Scoring Dataset Preprocessing Script
Preprocesses resume-job pairs with ATS compatibility scores for regression/classification.

Input: CSV files with combined text and ATS scores
Output: Cleaned and normalized data ready for training
"""

import pandas as pd
import numpy as np
import re
import json
import pickle
from pathlib import Path
from typing import Dict, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from collections import Counter


class ATSPreprocessor:
    """Preprocessor for ATS scoring dataset."""

    def __init__(self, data_dir: str, output_dir: str):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.score_scaler = None
        self.label_to_id = {}
        self.id_to_label = {}

    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if pd.isna(text) or text == '':
            return ''

        # Convert to string
        text = str(text)

        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)

        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\-\(\)\:\;\!]', ' ', text)

        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)

        return text.strip()

    def split_combined_text(self, text: str) -> Tuple[str, str]:
        """
        Split combined resume+job text (if format is 'resume [SEP] job_description').

        Returns:
            Tuple of (resume_text, job_description_text)
        """
        # Check for common separators
        separators = [' [SEP] ', '[SEP]', '\n\n\n', '|||']

        for sep in separators:
            if sep in text:
                parts = text.split(sep, 1)
                if len(parts) == 2:
                    return parts[0].strip(), parts[1].strip()

        # If no separator found, treat entire text as combined
        # You might want to keep it combined for sentence-transformers
        return text, text

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load train and validation CSV files."""
        print("\nLoading CSV files...")
        train_df = pd.read_csv(self.data_dir / 'train.csv')
        val_df = pd.read_csv(self.data_dir / 'validation.csv')

        print(f"  Train samples: {len(train_df)}")
        print(f"  Validation samples: {len(val_df)}")

        return train_df, val_df

    def analyze_score_distribution(self, df: pd.DataFrame, split_name: str = ""):
        """Analyze and print score distribution."""
        print(f"\n{split_name} Score Distribution:")
        print(f"  Mean: {df['ats_score'].mean():.2f}")
        print(f"  Median: {df['ats_score'].median():.2f}")
        print(f"  Std: {df['ats_score'].std():.2f}")
        print(f"  Min: {df['ats_score'].min():.2f}")
        print(f"  Max: {df['ats_score'].max():.2f}")
        print(f"  25th percentile: {df['ats_score'].quantile(0.25):.2f}")
        print(f"  75th percentile: {df['ats_score'].quantile(0.75):.2f}")

        if 'original_label' in df.columns:
            print(f"\n{split_name} Label Distribution:")
            label_counts = df['original_label'].value_counts()
            for label, count in label_counts.items():
                percentage = (count / len(df)) * 100
                print(f"  {label}: {count} ({percentage:.2f}%)")

    def build_label_mapping(self, labels: pd.Series):
        """Build label to ID mapping for categorical labels."""
        unique_labels = sorted(labels.unique())
        self.label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
        self.id_to_label = {idx: label for label, idx in self.label_to_id.items()}

        print(f"\nLabel mapping:")
        for label, idx in self.label_to_id.items():
            print(f"  {label}: {idx}")

    def normalize_scores(self, train_scores: np.ndarray, val_scores: np.ndarray,
                        method: str = 'minmax') -> Tuple[np.ndarray, np.ndarray]:
        """
        Normalize ATS scores.

        Args:
            train_scores: Training scores
            val_scores: Validation scores
            method: 'minmax' (0-1) or 'standard' (z-score)

        Returns:
            Normalized train and validation scores
        """
        print(f"\nNormalizing scores using {method} method...")

        if method == 'minmax':
            self.score_scaler = MinMaxScaler()
        elif method == 'standard':
            self.score_scaler = StandardScaler()
        else:
            raise ValueError(f"Unknown normalization method: {method}")

        # Fit on training data only
        train_scores_norm = self.score_scaler.fit_transform(train_scores.reshape(-1, 1)).flatten()
        val_scores_norm = self.score_scaler.transform(val_scores.reshape(-1, 1)).flatten()

        print(f"  Normalized train scores - Mean: {train_scores_norm.mean():.4f}, Std: {train_scores_norm.std():.4f}")
        print(f"  Normalized val scores - Mean: {val_scores_norm.mean():.4f}, Std: {val_scores_norm.std():.4f}")

        return train_scores_norm, val_scores_norm

    def create_score_bins(self, scores: pd.Series, bins: int = 3) -> pd.Series:
        """
        Create categorical bins from continuous scores for classification.

        Args:
            scores: Continuous ATS scores
            bins: Number of bins (default: 3 for Low/Medium/High)

        Returns:
            Categorical labels
        """
        if bins == 3:
            labels = ['Low', 'Medium', 'High']
        else:
            labels = [f'Bin_{i}' for i in range(bins)]

        return pd.cut(scores, bins=bins, labels=labels)

    def preprocess(self, normalize_scores: bool = True,
                  normalization_method: str = 'minmax',
                  create_bins: bool = False,
                  random_state: int = 42):
        """
        Main preprocessing pipeline.

        Args:
            normalize_scores: Whether to normalize continuous scores
            normalization_method: 'minmax' or 'standard'
            create_bins: Whether to create categorical bins for classification
            random_state: Random seed for reproducibility
        """
        print("=" * 80)
        print("ATS Scoring Dataset Preprocessing")
        print("=" * 80)

        # Load data
        print("\n[1/6] Loading data...")
        train_df, val_df = self.load_data()

        # Analyze initial distribution
        print("\n[2/6] Analyzing score distribution...")
        self.analyze_score_distribution(train_df, "Training")
        self.analyze_score_distribution(val_df, "Validation")

        # Clean text
        print("\n[3/6] Cleaning text...")
        for df in [train_df, val_df]:
            print(f"  Cleaning {len(df)} samples...")
            df['text_clean'] = df['text'].apply(self.clean_text)

            # Calculate text statistics
            df['text_length'] = df['text_clean'].apply(len)
            df['word_count'] = df['text_clean'].apply(lambda x: len(x.split()))

        # Normalize scores if requested
        if normalize_scores:
            print("\n[4/6] Normalizing scores...")
            train_norm, val_norm = self.normalize_scores(
                train_df['ats_score'].values,
                val_df['ats_score'].values,
                method=normalization_method
            )
            train_df['ats_score_normalized'] = train_norm
            val_df['ats_score_normalized'] = val_norm
        else:
            print("\n[4/6] Skipping score normalization...")
            train_df['ats_score_normalized'] = train_df['ats_score']
            val_df['ats_score_normalized'] = val_df['ats_score']

        # Create categorical bins if requested
        if create_bins:
            print("\n[5/6] Creating score bins for classification...")
            train_df['score_bin'] = self.create_score_bins(train_df['ats_score'])
            val_df['score_bin'] = self.create_score_bins(val_df['ats_score'])

            print("\nBin distribution (Training):")
            print(train_df['score_bin'].value_counts())
            print("\nBin distribution (Validation):")
            print(val_df['score_bin'].value_counts())
        else:
            print("\n[5/6] Skipping bin creation...")

        # Build label mapping for original categorical labels
        print("\n[6/6] Building label mappings...")
        if 'original_label' in train_df.columns:
            all_labels = pd.concat([train_df['original_label'], val_df['original_label']])
            self.build_label_mapping(all_labels)

            # Convert labels to IDs
            train_df['label_id'] = train_df['original_label'].map(self.label_to_id)
            val_df['label_id'] = val_df['original_label'].map(self.label_to_id)

        # Save preprocessed data
        print("\n[7/7] Saving preprocessed data...")

        # Save as CSV
        train_df.to_csv(self.output_dir / 'ats_train.csv', index=False)
        val_df.to_csv(self.output_dir / 'ats_val.csv', index=False)

        # Save as pickle for faster loading
        with open(self.output_dir / 'ats_train.pkl', 'wb') as f:
            pickle.dump(train_df, f)

        with open(self.output_dir / 'ats_val.pkl', 'wb') as f:
            pickle.dump(val_df, f)

        # Save metadata
        metadata = {
            'label_to_id': self.label_to_id,
            'id_to_label': self.id_to_label,
            'num_classes': len(self.label_to_id) if self.label_to_id else 0,
            'score_normalization': normalization_method if normalize_scores else None,
            'score_range': {
                'original_min': float(train_df['ats_score'].min()),
                'original_max': float(train_df['ats_score'].max()),
                'original_mean': float(train_df['ats_score'].mean()),
                'original_std': float(train_df['ats_score'].std())
            }
        }

        # Save scaler
        if self.score_scaler is not None:
            with open(self.output_dir / 'ats_score_scaler.pkl', 'wb') as f:
                pickle.dump(self.score_scaler, f)
            metadata['scaler_saved'] = True

        with open(self.output_dir / 'ats_metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)

        with open(self.output_dir / 'ats_metadata.json', 'w') as f:
            json_metadata = {k: v for k, v in metadata.items() if k != 'scaler'}
            json.dump(json_metadata, f, indent=2)

        # Calculate and save statistics
        stats = {
            'train_samples': int(len(train_df)),
            'val_samples': int(len(val_df)),
            'score_statistics': {
                'train': {
                    'mean': float(train_df['ats_score'].mean()),
                    'std': float(train_df['ats_score'].std()),
                    'min': float(train_df['ats_score'].min()),
                    'max': float(train_df['ats_score'].max()),
                    'median': float(train_df['ats_score'].median())
                },
                'val': {
                    'mean': float(val_df['ats_score'].mean()),
                    'std': float(val_df['ats_score'].std()),
                    'min': float(val_df['ats_score'].min()),
                    'max': float(val_df['ats_score'].max()),
                    'median': float(val_df['ats_score'].median())
                }
            },
            'label_distribution_train': {k: int(v) for k, v in train_df['original_label'].value_counts().items()} if 'original_label' in train_df.columns else {},
            'label_distribution_val': {k: int(v) for k, v in val_df['original_label'].value_counts().items()} if 'original_label' in val_df.columns else {},
            'avg_text_length_chars': {
                'train': float(train_df['text_length'].mean()),
                'val': float(val_df['text_length'].mean())
            },
            'avg_word_count': {
                'train': float(train_df['word_count'].mean()),
                'val': float(val_df['word_count'].mean())
            },
            'max_word_count': int(train_df['word_count'].max()),
            'score_normalized': normalize_scores,
            'normalization_method': normalization_method if normalize_scores else None,
            'bins_created': create_bins
        }

        with open(self.output_dir / 'ats_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"\n✓ Preprocessing complete!")
        print(f"  Output directory: {self.output_dir}")
        print(f"  Files saved:")
        print(f"    - ats_train.csv / .pkl")
        print(f"    - ats_val.csv / .pkl")
        print(f"    - ats_metadata.json / .pkl")
        if self.score_scaler:
            print(f"    - ats_score_scaler.pkl")
        print(f"    - ats_stats.json")
        print("=" * 80)

        return stats


if __name__ == "__main__":
    # Configuration
    DATA_DIR = "data/ats"
    OUTPUT_DIR = "pre-processed-data/ats"

    # Initialize preprocessor
    preprocessor = ATSPreprocessor(DATA_DIR, OUTPUT_DIR)

    # Run preprocessing
    stats = preprocessor.preprocess(
        normalize_scores=True,
        normalization_method='minmax',  # Options: 'minmax' or 'standard'
        create_bins=False,  # Set to True if you want categorical classification
        random_state=42
    )

    print("\nPreprocessing Statistics:")
    print(json.dumps(stats, indent=2))
