"""
CV-Job Match Dataset Preprocessing Script
Preprocesses CV and job description pairs for semantic matching task.

Input: CSV files with resume_text, job_description_text, and labels
Output: Cleaned and tokenized data ready for transformer-based models
"""

import pandas as pd
import numpy as np
import re
import json
import pickle
from pathlib import Path
from typing import Dict, Tuple
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter


class CVJobMatchPreprocessor:
    """Preprocessor for CV-Job matching dataset."""

    def __init__(self, data_dir: str, output_dir: str):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

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

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load train and test CSV files."""
        print("\nLoading CSV files...")
        train_df = pd.read_csv(self.data_dir / 'train.csv')
        test_df = pd.read_csv(self.data_dir / 'test.csv')

        print(f"  Train samples: {len(train_df)}")
        print(f"  Test samples: {len(test_df)}")

        return train_df, test_df

    def analyze_class_distribution(self, df: pd.DataFrame, split_name: str = ""):
        """Analyze and print class distribution."""
        print(f"\n{split_name} Class Distribution:")
        class_counts = df['label'].value_counts()
        class_percentages = df['label'].value_counts(normalize=True) * 100

        for label in class_counts.index:
            count = class_counts[label]
            percentage = class_percentages[label]
            print(f"  {label}: {count} ({percentage:.2f}%)")

        return class_counts

    def build_label_mapping(self, labels: pd.Series):
        """Build label to ID mapping."""
        unique_labels = sorted(labels.unique())
        self.label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
        self.id_to_label = {idx: label for label, idx in self.label_to_id.items()}

        print(f"\nLabel mapping:")
        for label, idx in self.label_to_id.items():
            print(f"  {label}: {idx}")

    def compute_class_weights(self, labels: np.ndarray) -> Dict[int, float]:
        """Compute class weights for handling imbalance."""
        unique_classes = np.unique(labels)
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=unique_classes,
            y=labels
        )

        class_weight_dict = {cls: weight for cls, weight in zip(unique_classes, class_weights)}

        print("\nClass weights (for handling imbalance):")
        for cls, weight in class_weight_dict.items():
            print(f"  Class {cls} ({self.id_to_label[cls]}): {weight:.4f}")

        return class_weight_dict

    def preprocess(self, val_size: float = 0.15, random_state: int = 42):
        """
        Main preprocessing pipeline.

        Args:
            val_size: Proportion of train data for validation
            random_state: Random seed for reproducibility
        """
        print("=" * 80)
        print("CV-Job Match Dataset Preprocessing")
        print("=" * 80)

        # Load data
        print("\n[1/7] Loading data...")
        train_df, test_df = self.load_data()

        # Analyze initial distribution
        print("\n[2/7] Analyzing class distribution...")
        self.analyze_class_distribution(train_df, "Training")
        self.analyze_class_distribution(test_df, "Test")

        # Clean text
        print("\n[3/7] Cleaning text...")
        for df in [train_df, test_df]:
            print(f"  Cleaning {len(df)} samples...")
            df['resume_text_clean'] = df['resume_text'].apply(self.clean_text)
            df['job_description_text_clean'] = df['job_description_text'].apply(self.clean_text)

            # Calculate text lengths
            df['resume_length'] = df['resume_text_clean'].apply(len)
            df['job_desc_length'] = df['job_description_text_clean'].apply(len)
            df['resume_word_count'] = df['resume_text_clean'].apply(lambda x: len(x.split()))
            df['job_desc_word_count'] = df['job_description_text_clean'].apply(lambda x: len(x.split()))

        # Build label mapping
        print("\n[4/7] Building label mappings...")
        all_labels = pd.concat([train_df['label'], test_df['label']])
        self.build_label_mapping(all_labels)

        # Convert labels to IDs
        print("\n[5/7] Converting labels to IDs...")
        train_df['label_id'] = train_df['label'].map(self.label_to_id)
        test_df['label_id'] = test_df['label'].map(self.label_to_id)

        # Compute class weights
        print("\n[6/7] Computing class weights...")
        class_weights = self.compute_class_weights(train_df['label_id'].values)

        # Split train into train and validation
        print("\n[7/7] Splitting into train/val sets...")
        train_data, val_data = train_test_split(
            train_df,
            test_size=val_size,
            random_state=random_state,
            stratify=train_df['label_id']  # Stratified split to maintain class distribution
        )

        print(f"  Train samples: {len(train_data)}")
        print(f"  Validation samples: {len(val_data)}")
        print(f"  Test samples: {len(test_df)}")

        # Analyze split distributions
        self.analyze_class_distribution(train_data, "Final Training")
        self.analyze_class_distribution(val_data, "Validation")

        # Save preprocessed data
        print("\n[8/8] Saving preprocessed data...")

        # Save as CSV
        train_data.to_csv(self.output_dir / 'cv_job_match_train.csv', index=False)
        val_data.to_csv(self.output_dir / 'cv_job_match_val.csv', index=False)
        test_df.to_csv(self.output_dir / 'cv_job_match_test.csv', index=False)

        # Save as pickle for faster loading
        with open(self.output_dir / 'cv_job_match_train.pkl', 'wb') as f:
            pickle.dump(train_data, f)

        with open(self.output_dir / 'cv_job_match_val.pkl', 'wb') as f:
            pickle.dump(val_data, f)

        with open(self.output_dir / 'cv_job_match_test.pkl', 'wb') as f:
            pickle.dump(test_df, f)

        # Save label mappings and class weights
        metadata = {
            'label_to_id': self.label_to_id,
            'id_to_label': self.id_to_label,
            'class_weights': class_weights,
            'num_classes': len(self.label_to_id)
        }

        with open(self.output_dir / 'cv_job_match_metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)

        with open(self.output_dir / 'cv_job_match_metadata.json', 'w') as f:
            # Convert class_weights keys to strings for JSON
            json_metadata = {
                'label_to_id': self.label_to_id,
                'id_to_label': self.id_to_label,
                'class_weights': {str(k): v for k, v in class_weights.items()},
                'num_classes': len(self.label_to_id)
            }
            json.dump(json_metadata, f, indent=2)

        # Calculate and save statistics
        stats = {
            'train_samples': int(len(train_data)),
            'val_samples': int(len(val_data)),
            'test_samples': int(len(test_df)),
            'num_classes': int(len(self.label_to_id)),
            'class_distribution_train': {k: int(v) for k, v in train_data['label'].value_counts().items()},
            'class_distribution_val': {k: int(v) for k, v in val_data['label'].value_counts().items()},
            'class_distribution_test': {k: int(v) for k, v in test_df['label'].value_counts().items()},
            'avg_resume_length_chars': {
                'train': float(train_data['resume_length'].mean()),
                'val': float(val_data['resume_length'].mean()),
                'test': float(test_df['resume_length'].mean())
            },
            'avg_job_desc_length_chars': {
                'train': float(train_data['job_desc_length'].mean()),
                'val': float(val_data['job_desc_length'].mean()),
                'test': float(test_df['job_desc_length'].mean())
            },
            'avg_resume_word_count': {
                'train': float(train_data['resume_word_count'].mean()),
                'val': float(val_data['resume_word_count'].mean()),
                'test': float(test_df['resume_word_count'].mean())
            },
            'avg_job_desc_word_count': {
                'train': float(train_data['job_desc_word_count'].mean()),
                'val': float(val_data['job_desc_word_count'].mean()),
                'test': float(test_df['job_desc_word_count'].mean())
            },
            'max_resume_word_count': int(train_data['resume_word_count'].max()),
            'max_job_desc_word_count': int(train_data['job_desc_word_count'].max()),
        }

        with open(self.output_dir / 'cv_job_match_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"\n✓ Preprocessing complete!")
        print(f"  Output directory: {self.output_dir}")
        print(f"  Files saved:")
        print(f"    - cv_job_match_train.csv / .pkl")
        print(f"    - cv_job_match_val.csv / .pkl")
        print(f"    - cv_job_match_test.csv / .pkl")
        print(f"    - cv_job_match_metadata.json / .pkl")
        print(f"    - cv_job_match_stats.json")
        print("=" * 80)

        return stats


if __name__ == "__main__":
    # Configuration
    DATA_DIR = "data/cv_job_match"
    OUTPUT_DIR = "pre-processed-data/cv_job_match"

    # Initialize preprocessor
    preprocessor = CVJobMatchPreprocessor(DATA_DIR, OUTPUT_DIR)

    # Run preprocessing
    stats = preprocessor.preprocess(
        val_size=0.15,
        random_state=42
    )

    print("\nPreprocessing Statistics:")
    print(json.dumps(stats, indent=2))
