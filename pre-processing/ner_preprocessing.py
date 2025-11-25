"""
NER Dataset Preprocessing Script
Preprocesses resume JSON annotations for Named Entity Recognition task.

Input: JSON files with annotations in data/ner/ResumesJsonAnnotated/
Output: Tokenized and labeled data in BIO format for NER training
"""

import json
import os
import re
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
import pickle


class NERPreprocessor:
    """Preprocessor for NER dataset with BIO tagging scheme."""

    def __init__(self, data_dir: str, output_dir: str):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # BIO tagging scheme
        self.label_to_id = {}
        self.id_to_label = {}

    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\-\@\(\)]', ' ', text)
        return text.strip()

    def load_annotations(self) -> List[Dict]:
        """Load all JSON annotation files."""
        json_files = list(self.data_dir.glob('*.json'))
        print(f"Found {len(json_files)} JSON files")

        all_data = []
        skipped = 0

        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Check if annotations exist
                if 'annotations' in data and len(data['annotations']) > 0:
                    all_data.append({
                        'text': data['text'],
                        'annotations': data['annotations'],
                        'file_name': json_file.name
                    })
                else:
                    skipped += 1
            except Exception as e:
                print(f"Error loading {json_file.name}: {e}")
                skipped += 1

        print(f"Loaded {len(all_data)} files successfully, skipped {skipped} files")
        return all_data

    def create_bio_tags(self, text: str, annotations: List) -> Tuple[List[str], List[str]]:
        """
        Convert character-level annotations to BIO-tagged tokens.

        Returns:
            tokens: List of word tokens
            bio_tags: List of BIO tags for each token
        """
        # Simple whitespace tokenization (can be improved with spaCy or NLTK)
        words = text.split()

        # Initialize all tags as 'O' (Outside)
        bio_tags = ['O'] * len(words)

        # Track character positions for each word
        char_to_word = {}
        current_pos = 0
        for word_idx, word in enumerate(words):
            # Find word in text starting from current position
            word_start = text.find(word, current_pos)
            if word_start != -1:
                word_end = word_start + len(word)
                for char_pos in range(word_start, word_end):
                    char_to_word[char_pos] = word_idx
                current_pos = word_end

        # Process annotations
        for annotation in annotations:
            start_pos, end_pos, label = annotation

            # Extract entity type (e.g., "SKILL: WINDOWS XP" -> "SKILL")
            if ':' in label:
                entity_type = label.split(':')[0].strip()
            else:
                entity_type = label.strip()

            # Find which words are covered by this annotation
            covered_words = set()
            for char_pos in range(start_pos, min(end_pos, len(char_to_word))):
                if char_pos in char_to_word:
                    covered_words.add(char_to_word[char_pos])

            # Apply BIO tagging
            covered_words = sorted(covered_words)
            for idx, word_idx in enumerate(covered_words):
                if idx == 0:
                    bio_tags[word_idx] = f'B-{entity_type}'
                else:
                    bio_tags[word_idx] = f'I-{entity_type}'

        return words, bio_tags

    def build_label_mapping(self, all_bio_tags: List[List[str]]):
        """Build label to ID mapping for all unique BIO tags."""
        all_labels = set(['O'])  # Start with 'O' tag

        for tags in all_bio_tags:
            all_labels.update(tags)

        # Sort for consistency
        sorted_labels = sorted(all_labels)
        self.label_to_id = {label: idx for idx, label in enumerate(sorted_labels)}
        self.id_to_label = {idx: label for label, idx in self.label_to_id.items()}

        print(f"\nFound {len(self.label_to_id)} unique BIO tags:")
        print(f"Labels: {sorted_labels[:20]}...")  # Show first 20

    def preprocess(self, test_size: float = 0.15, val_size: float = 0.15, random_state: int = 42):
        """
        Main preprocessing pipeline.

        Args:
            test_size: Proportion of data for test set
            val_size: Proportion of remaining data for validation set
            random_state: Random seed for reproducibility
        """
        print("=" * 80)
        print("NER Dataset Preprocessing")
        print("=" * 80)

        # Load data
        print("\n[1/6] Loading annotations...")
        all_data = self.load_annotations()

        if len(all_data) == 0:
            raise ValueError("No valid annotation files found!")

        # Convert to BIO format
        print("\n[2/6] Converting to BIO format...")
        processed_samples = []
        entity_stats = Counter()

        for idx, sample in enumerate(all_data):
            if idx % 500 == 0:
                print(f"  Processing {idx}/{len(all_data)}...")

            text = sample['text']
            annotations = sample['annotations']

            # Create BIO tags
            tokens, bio_tags = self.create_bio_tags(text, annotations)

            # Collect entity statistics
            for tag in bio_tags:
                if tag.startswith('B-'):
                    entity_type = tag[2:]
                    entity_stats[entity_type] += 1

            processed_samples.append({
                'tokens': tokens,
                'bio_tags': bio_tags,
                'file_name': sample['file_name']
            })

        print(f"\n  Processed {len(processed_samples)} samples")
        print(f"\n  Entity distribution:")
        for entity, count in entity_stats.most_common(10):
            print(f"    {entity}: {count}")

        # Extract tokens and tags
        all_tokens = [sample['tokens'] for sample in processed_samples]
        all_bio_tags = [sample['bio_tags'] for sample in processed_samples]

        # Build label mapping
        print("\n[3/6] Building label mappings...")
        self.build_label_mapping(all_bio_tags)

        # Convert tags to IDs
        print("\n[4/6] Converting tags to IDs...")
        all_tag_ids = []
        for tags in all_bio_tags:
            tag_ids = [self.label_to_id[tag] for tag in tags]
            all_tag_ids.append(tag_ids)

        # Split data
        print("\n[5/6] Splitting into train/val/test sets...")
        # First split: train+val and test
        train_val_tokens, test_tokens, train_val_tags, test_tags = train_test_split(
            all_tokens, all_tag_ids, test_size=test_size, random_state=random_state
        )

        # Second split: train and val
        val_size_adjusted = val_size / (1 - test_size)
        train_tokens, val_tokens, train_tags, val_tags = train_test_split(
            train_val_tokens, train_val_tags, test_size=val_size_adjusted, random_state=random_state
        )

        print(f"  Train samples: {len(train_tokens)}")
        print(f"  Validation samples: {len(val_tokens)}")
        print(f"  Test samples: {len(test_tokens)}")

        # Save preprocessed data
        print("\n[6/6] Saving preprocessed data...")

        # Save as pickle files
        with open(self.output_dir / 'ner_train.pkl', 'wb') as f:
            pickle.dump({'tokens': train_tokens, 'tags': train_tags}, f)

        with open(self.output_dir / 'ner_val.pkl', 'wb') as f:
            pickle.dump({'tokens': val_tokens, 'tags': val_tags}, f)

        with open(self.output_dir / 'ner_test.pkl', 'wb') as f:
            pickle.dump({'tokens': test_tokens, 'tags': test_tags}, f)

        # Save label mappings
        with open(self.output_dir / 'ner_label_mappings.pkl', 'wb') as f:
            pickle.dump({
                'label_to_id': self.label_to_id,
                'id_to_label': self.id_to_label
            }, f)

        # Save as JSON for inspection
        with open(self.output_dir / 'ner_label_mappings.json', 'w') as f:
            json.dump({
                'label_to_id': self.label_to_id,
                'id_to_label': self.id_to_label,
                'num_labels': len(self.label_to_id)
            }, f, indent=2)

        # Save statistics
        stats = {
            'total_samples': len(all_tokens),
            'train_samples': len(train_tokens),
            'val_samples': len(val_tokens),
            'test_samples': len(test_tokens),
            'num_labels': len(self.label_to_id),
            'entity_distribution': dict(entity_stats),
            'avg_tokens_per_sample': np.mean([len(tokens) for tokens in all_tokens]),
            'max_tokens_per_sample': max([len(tokens) for tokens in all_tokens]),
        }

        with open(self.output_dir / 'ner_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"\n✓ Preprocessing complete!")
        print(f"  Output directory: {self.output_dir}")
        print(f"  Files saved:")
        print(f"    - ner_train.pkl")
        print(f"    - ner_val.pkl")
        print(f"    - ner_test.pkl")
        print(f"    - ner_label_mappings.pkl")
        print(f"    - ner_label_mappings.json")
        print(f"    - ner_stats.json")
        print("=" * 80)

        return stats


if __name__ == "__main__":
    # Configuration
    DATA_DIR = "data/ner/ResumesJsonAnnotated"
    OUTPUT_DIR = "pre-processed-data/ner"

    # Initialize preprocessor
    preprocessor = NERPreprocessor(DATA_DIR, OUTPUT_DIR)

    # Run preprocessing
    stats = preprocessor.preprocess(
        test_size=0.15,
        val_size=0.15,
        random_state=42
    )

    print("\nPreprocessing Statistics:")
    print(json.dumps(stats, indent=2))
