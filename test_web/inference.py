"""
Model Inference Utilities
Load and run inference on all three models:
1. NER - Extract skills from CV
2. CV-Job Matching - Match CV to job description
3. ATS Scoring - Predict ATS compatibility score
"""

import sys
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from transformers import BertTokenizerFast, RobertaTokenizerFast

# Add parent directory to path for model imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.ner_model import get_ner_model
from models.ats_model import get_ats_model
from models.cv_job_match_model import get_cv_job_match_model


class NERInference:
    """Named Entity Recognition for skill extraction from CVs"""

    def __init__(self, model_path: str = None, model_type: str = 'bert'):
        """
        Initialize NER model.

        Args:
            model_path: Path to the trained model checkpoint
            model_type: Type of model ('bert', 'roberta', 'bilstm-crf')
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = model_type

        # Default model path
        if model_path is None:
            base_path = Path(__file__).parent.parent / 'trained_models' / 'ner'
            model_path = str(base_path / model_type / 'best_model.pt')

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        # Get model configuration from checkpoint
        num_labels = checkpoint.get('num_labels', 3)  # B-SKILL, I-SKILL, O
        self.id_to_label = checkpoint.get('id_to_label', {0: 'O', 1: 'B-SKILL', 2: 'I-SKILL'})

        # Ensure id_to_label keys are integers
        self.id_to_label = {int(k): v for k, v in self.id_to_label.items()}

        # Initialize model
        if model_type == 'bilstm-crf':
            vocab_size = checkpoint.get('vocab_size', 30000)
            self.model = get_ner_model(
                model_type,
                num_labels=num_labels,
                vocab_size=vocab_size,
                embedding_dim=300,
                hidden_dim=256
            )
            self.word2id = checkpoint.get('word2id', {})
            self.tokenizer = None
        else:
            self.model = get_ner_model(model_type, num_labels=num_labels)
            if model_type == 'roberta':
                self.tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
            else:
                self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

    def extract_skills(self, text: str) -> List[str]:
        """
        Extract skills from text.

        Args:
            text: Input text (CV content)

        Returns:
            List of extracted skills
        """
        if not text.strip():
            return []

        # Tokenize
        words = text.split()

        if self.tokenizer is None:
            # BiLSTM-CRF
            input_ids = [self.word2id.get(w.lower(), 1) for w in words]  # 1 = UNK
            input_ids = torch.tensor([input_ids], device=self.device)
            attention_mask = torch.ones_like(input_ids)

            with torch.no_grad():
                output = self.model(input_ids, attention_mask=attention_mask)
                predictions = output['predictions'][0]
        else:
            # Transformer models
            encoding = self.tokenizer(
                words,
                is_split_into_words=True,
                padding='max_length',
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )

            with torch.no_grad():
                output = self.model(
                    input_ids=encoding['input_ids'].to(self.device),
                    attention_mask=encoding['attention_mask'].to(self.device)
                )

                if 'logits' in output:
                    predictions = torch.argmax(output['logits'], dim=-1)[0].cpu().numpy()
                else:
                    predictions = output['predictions'][0]

            # Map subword predictions back to words
            word_ids = encoding.word_ids(batch_index=0)
            word_predictions = []
            prev_word_id = None

            for idx, word_id in enumerate(word_ids):
                if word_id is None:
                    continue
                if word_id != prev_word_id:
                    word_predictions.append(predictions[idx])
                prev_word_id = word_id

            predictions = word_predictions

        # Extract skill entities using BIO tagging
        skills = []
        current_skill = []

        for i, (word, pred_id) in enumerate(zip(words, predictions)):
            if i >= len(predictions):
                break

            label = self.id_to_label.get(int(pred_id), 'O')

            if label == 'B-SKILL':
                if current_skill:
                    skills.append(' '.join(current_skill))
                current_skill = [word]
            elif label == 'I-SKILL' and current_skill:
                current_skill.append(word)
            else:
                if current_skill:
                    skills.append(' '.join(current_skill))
                    current_skill = []

        if current_skill:
            skills.append(' '.join(current_skill))

        # Remove duplicates while preserving order
        seen = set()
        unique_skills = []
        for skill in skills:
            skill_lower = skill.lower()
            if skill_lower not in seen:
                seen.add(skill_lower)
                unique_skills.append(skill)

        return unique_skills


class CVJobMatchInference:
    """CV-Job Description Matching"""

    def __init__(self, model_path: str = None, model_type: str = 'bert'):
        """
        Initialize CV-Job matching model.

        Args:
            model_path: Path to the trained model checkpoint
            model_type: Type of model ('bert', 'sbert')
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = model_type

        # Default model path
        if model_path is None:
            base_path = Path(__file__).parent.parent / 'trained_models' / 'cv_jd_matching'
            model_path = str(base_path / model_type / 'best_model.pt')

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        # Initialize model
        num_classes = 3  # No Fit, Potential Fit, Good Fit
        self.model = get_cv_job_match_model(model_type, num_classes=num_classes)

        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        # Initialize tokenizer
        self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

        # Label mapping
        self.label_names = {0: 'No Fit', 1: 'Potential Fit', 2: 'Good Fit'}

    def match(self, cv_text: str, job_text: str) -> Dict:
        """
        Match CV to job description.

        Args:
            cv_text: CV/Resume text
            job_text: Job description text

        Returns:
            Dictionary with prediction and probabilities
        """
        if self.model_type == 'sbert':
            # SBERT uses separate encodings
            cv_encoding = self.tokenizer(
                cv_text,
                padding='max_length',
                truncation=True,
                max_length=256,
                return_tensors='pt'
            )
            job_encoding = self.tokenizer(
                job_text,
                padding='max_length',
                truncation=True,
                max_length=256,
                return_tensors='pt'
            )

            with torch.no_grad():
                output = self.model(
                    cv_input_ids=cv_encoding['input_ids'].to(self.device),
                    cv_attention_mask=cv_encoding['attention_mask'].to(self.device),
                    job_input_ids=job_encoding['input_ids'].to(self.device),
                    job_attention_mask=job_encoding['attention_mask'].to(self.device)
                )
        else:
            # BERT uses text pair encoding
            encoding = self.tokenizer(
                cv_text,
                job_text,
                padding='max_length',
                truncation=True,
                max_length=256,
                return_tensors='pt'
            )

            with torch.no_grad():
                output = self.model(
                    input_ids=encoding['input_ids'].to(self.device),
                    attention_mask=encoding['attention_mask'].to(self.device),
                    token_type_ids=encoding.get('token_type_ids', torch.zeros_like(encoding['input_ids'])).to(self.device)
                )

        logits = output['logits']
        probabilities = torch.softmax(logits, dim=-1)[0].cpu().numpy()
        prediction = torch.argmax(logits, dim=-1).item()

        return {
            'prediction': self.label_names[prediction],
            'prediction_id': prediction,
            'probabilities': {
                'no_fit': float(probabilities[0]),
                'potential_fit': float(probabilities[1]),
                'good_fit': float(probabilities[2])
            },
            'confidence': float(probabilities[prediction])
        }


class ATSScoreInference:
    """ATS (Applicant Tracking System) Score Prediction"""

    def __init__(self, model_path: str = None, model_type: str = 'bert'):
        """
        Initialize ATS scoring model.

        Args:
            model_path: Path to the trained model checkpoint
            model_type: Type of model ('bert', 'roberta')
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = model_type

        # Default model path
        if model_path is None:
            base_path = Path(__file__).parent.parent / 'trained_models' / 'ats'
            model_path = str(base_path / model_type / 'best_model.pt')

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        # Initialize model
        self.model = get_ats_model(model_type)

        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        # Load scaler for denormalization (if available)
        self.scaler = checkpoint.get('scaler', None)

        # Initialize tokenizer
        if model_type == 'roberta':
            self.tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')
        else:
            self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    def score(self, text: str) -> Dict:
        """
        Predict ATS compatibility score for a resume.

        Args:
            text: Resume/CV text

        Returns:
            Dictionary with score and interpretation
        """
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=256,
            return_tensors='pt'
        )

        with torch.no_grad():
            output = self.model(
                input_ids=encoding['input_ids'].to(self.device),
                attention_mask=encoding['attention_mask'].to(self.device)
            )

            normalized_score = output['predictions'].item()

        # Denormalize if scaler available
        if self.scaler is not None:
            try:
                actual_score = self.scaler.inverse_transform([[normalized_score]])[0][0]
            except Exception:
                actual_score = normalized_score * 100
        else:
            actual_score = normalized_score * 100

        # Clamp score to 0-100 range
        actual_score = max(0, min(100, actual_score))

        # Interpret score
        if actual_score >= 80:
            interpretation = "Excellent - Your CV is highly ATS-compatible"
            color = "success"
        elif actual_score >= 60:
            interpretation = "Good - Your CV should pass most ATS systems"
            color = "info"
        elif actual_score >= 40:
            interpretation = "Fair - Consider optimizing your CV for ATS"
            color = "warning"
        else:
            interpretation = "Needs Improvement - Your CV may be filtered by ATS"
            color = "danger"

        return {
            'score': round(actual_score, 1),
            'normalized_score': normalized_score,
            'interpretation': interpretation,
            'color': color
        }


class CVAnalyzer:
    """
    Combined CV Analysis using all three models.
    """

    def __init__(
        self,
        ner_model_type: str = 'bert',
        match_model_type: str = 'bert',
        ats_model_type: str = 'bert'
    ):
        """
        Initialize all models.

        Args:
            ner_model_type: NER model type
            match_model_type: CV-Job matching model type
            ats_model_type: ATS scoring model type
        """
        print("Loading NER model...")
        self.ner = NERInference(model_type=ner_model_type)
        print("Loading CV-Job matching model...")
        self.matcher = CVJobMatchInference(model_type=match_model_type)
        print("Loading ATS scoring model...")
        self.ats = ATSScoreInference(model_type=ats_model_type)
        print("All models loaded!")

    def analyze(self, cv_text: str, job_description: str) -> Dict:
        """
        Perform complete CV analysis.

        Args:
            cv_text: CV/Resume text
            job_description: Job description text

        Returns:
            Dictionary with all analysis results
        """
        results = {}

        # 1. Extract skills from CV
        results['skills'] = self.ner.extract_skills(cv_text)

        # 2. Match CV to job description
        results['match'] = self.matcher.match(cv_text, job_description)

        # 3. Calculate ATS score
        results['ats'] = self.ats.score(cv_text)

        return results


# Singleton instance for the web app
_analyzer_instance: Optional[CVAnalyzer] = None


def get_analyzer() -> CVAnalyzer:
    """Get or create the CVAnalyzer singleton instance."""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = CVAnalyzer()
    return _analyzer_instance
