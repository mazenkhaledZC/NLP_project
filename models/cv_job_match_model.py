"""
CV-Job Match Model Architectures
Semantic matching between CVs and Job Descriptions.

Models:
1. LSTM Siamese Network: Baseline with learned embeddings
2. BERT for Sequence Classification: Fine-tuned BERT for text pairs
3. Sentence-BERT: Efficient semantic similarity with bi-encoders
4. RoBERTa Cross-Encoder: State-of-the-art text pair classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    BertForSequenceClassification,
    RobertaForSequenceClassification,
    DistilBertForSequenceClassification,
    BertModel,
    RobertaModel
)


class LSTMSiameseNetwork(nn.Module):
    """
    Siamese Network with LSTM encoders for CV-Job matching.

    Architecture:
    - Shared LSTM encoder for both CV and job description
    - Concatenate/subtract/multiply embeddings
    - Fully connected classifier
    """

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        embedding_dim: int = 300,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        pretrained_embeddings=None,
        pooling: str = 'mean'
    ):
        """
        Args:
            vocab_size: Size of vocabulary
            num_classes: Number of output classes (3: No Fit, Potential Fit, Good Fit)
            embedding_dim: Dimension of word embeddings
            hidden_dim: Hidden dimension of LSTM
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            pretrained_embeddings: Optional pretrained embeddings
            pooling: Pooling strategy ('mean', 'max', 'last')
        """
        super(LSTMSiameseNetwork, self).__init__()

        self.hidden_dim = hidden_dim
        self.pooling = pooling

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            self.embedding.weight.requires_grad = True

        # Shared LSTM encoder
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.dropout = nn.Dropout(dropout)

        # Feature dimension after concatenation
        # We concatenate: [cv_embedding; job_embedding; abs(cv - job); cv * job]
        feature_dim = hidden_dim * 2 * 4  # Bidirectional LSTM

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def encode(self, input_ids, attention_mask):
        """
        Encode a sequence using shared LSTM encoder.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            Sequence embedding [batch_size, hidden_dim * 2]
        """
        # Embedding
        embedded = self.embedding(input_ids)  # [batch_size, seq_len, embedding_dim]

        # LSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)  # lstm_out: [batch_size, seq_len, hidden_dim * 2]

        # Pooling
        if self.pooling == 'mean':
            # Mean pooling with attention mask
            mask = attention_mask.unsqueeze(-1).expand(lstm_out.size()).float()
            masked_lstm_out = lstm_out * mask
            sum_lstm = torch.sum(masked_lstm_out, dim=1)
            sum_mask = torch.sum(mask, dim=1).clamp(min=1e-9)
            encoding = sum_lstm / sum_mask
        elif self.pooling == 'max':
            # Max pooling
            encoding, _ = torch.max(lstm_out, dim=1)
        else:  # 'last'
            # Last hidden state
            encoding = torch.cat([hidden[-2], hidden[-1]], dim=1)  # Concatenate forward and backward

        return encoding

    def forward(self, cv_input_ids, cv_attention_mask, job_input_ids, job_attention_mask, labels=None):
        """
        Forward pass.

        Args:
            cv_input_ids: CV token IDs [batch_size, seq_len]
            cv_attention_mask: CV attention mask
            job_input_ids: Job description token IDs
            job_attention_mask: Job description attention mask
            labels: Ground truth labels [batch_size]

        Returns:
            Dictionary with loss and/or logits
        """
        # Encode CV and job description
        cv_embedding = self.encode(cv_input_ids, cv_attention_mask)
        job_embedding = self.encode(job_input_ids, job_attention_mask)

        # Concatenate features: [cv; job; |cv - job|; cv * job]
        features = torch.cat([
            cv_embedding,
            job_embedding,
            torch.abs(cv_embedding - job_embedding),
            cv_embedding * job_embedding
        ], dim=1)

        features = self.dropout(features)

        # Classification
        logits = self.classifier(features)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return {
            'loss': loss,
            'logits': logits,
            'cv_embedding': cv_embedding,
            'job_embedding': job_embedding
        }


class BERTForCVJobMatch(nn.Module):
    """
    BERT-based model for CV-Job matching.
    Uses BERT's sentence pair classification capability.
    """

    def __init__(
        self,
        num_classes: int,
        model_name: str = 'bert-base-uncased',
        dropout: float = 0.1,
        freeze_bert: bool = False
    ):
        """
        Args:
            num_classes: Number of output classes
            model_name: Pretrained BERT model name
            dropout: Dropout probability
            freeze_bert: Whether to freeze BERT parameters
        """
        super(BERTForCVJobMatch, self).__init__()

        # Load pretrained BERT
        self.bert = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout
        )

        # Optionally freeze BERT layers
        if freeze_bert:
            for param in self.bert.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        """
        Forward pass.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask
            token_type_ids: Segment IDs (0 for CV, 1 for job description)
            labels: Ground truth labels

        Returns:
            Dictionary with loss and/or logits
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels
        )

        return {
            'loss': outputs.loss if labels is not None else None,
            'logits': outputs.logits
        }


class RoBERTaForCVJobMatch(nn.Module):
    """
    RoBERTa-based model for CV-Job matching.
    RoBERTa is an optimized variant of BERT with better performance.
    """

    def __init__(
        self,
        num_classes: int,
        model_name: str = 'roberta-base',
        dropout: float = 0.1,
        freeze_encoder: bool = False
    ):
        """
        Args:
            num_classes: Number of output classes
            model_name: Pretrained RoBERTa model name
            dropout: Dropout probability
            freeze_encoder: Whether to freeze RoBERTa parameters
        """
        super(RoBERTaForCVJobMatch, self).__init__()

        # Load pretrained RoBERTa
        self.roberta = RobertaForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout
        )

        # Optionally freeze encoder layers
        if freeze_encoder:
            for param in self.roberta.roberta.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        Forward pass.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask
            labels: Ground truth labels

        Returns:
            Dictionary with loss and/or logits
        """
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        return {
            'loss': outputs.loss if labels is not None else None,
            'logits': outputs.logits
        }


class SentenceBERTBiEncoder(nn.Module):
    """
    Sentence-BERT Bi-Encoder for efficient semantic similarity.

    Architecture:
    - Separate BERT encoders for CV and job description (can share weights)
    - Mean pooling to get sentence embeddings
    - Cosine similarity or concatenation for classification
    """

    def __init__(
        self,
        num_classes: int,
        model_name: str = 'bert-base-uncased',
        pooling: str = 'mean',
        use_cosine_similarity: bool = False
    ):
        """
        Args:
            num_classes: Number of output classes
            model_name: Pretrained BERT model name
            pooling: Pooling strategy ('mean', 'cls', 'max')
            use_cosine_similarity: Use cosine similarity for classification
        """
        super(SentenceBERTBiEncoder, self).__init__()

        self.pooling = pooling
        self.use_cosine_similarity = use_cosine_similarity

        # Shared BERT encoder
        self.bert = BertModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size

        # Classifier
        if use_cosine_similarity:
            # Map similarity score to classes
            self.classifier = nn.Linear(1, num_classes)
        else:
            # Concatenate embeddings
            self.classifier = nn.Sequential(
                nn.Linear(hidden_size * 3, hidden_size),  # [cv; job; |cv-job|]
                nn.Tanh(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size, num_classes)
            )

    def mean_pooling(self, model_output, attention_mask):
        """
        Mean pooling - take attention mask into account for correct averaging.
        """
        token_embeddings = model_output[0]  # First element contains token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = input_mask_expanded.sum(dim=1).clamp(min=1e-9)
        return sum_embeddings / sum_mask

    def encode(self, input_ids, attention_mask):
        """
        Encode a sequence to a fixed-size embedding.
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        if self.pooling == 'mean':
            embedding = self.mean_pooling(outputs, attention_mask)
        elif self.pooling == 'cls':
            embedding = outputs[0][:, 0]  # CLS token
        elif self.pooling == 'max':
            embedding, _ = torch.max(outputs[0], dim=1)
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling}")

        # Normalize embeddings
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding

    def forward(self, cv_input_ids, cv_attention_mask, job_input_ids, job_attention_mask, labels=None):
        """
        Forward pass.

        Args:
            cv_input_ids: CV token IDs
            cv_attention_mask: CV attention mask
            job_input_ids: Job description token IDs
            job_attention_mask: Job description attention mask
            labels: Ground truth labels

        Returns:
            Dictionary with loss and/or logits
        """
        # Encode CV and job description
        cv_embedding = self.encode(cv_input_ids, cv_attention_mask)
        job_embedding = self.encode(job_input_ids, job_attention_mask)

        if self.use_cosine_similarity:
            # Compute cosine similarity
            similarity = F.cosine_similarity(cv_embedding, job_embedding, dim=1, eps=1e-8)
            logits = self.classifier(similarity.unsqueeze(-1))
        else:
            # Concatenate embeddings
            features = torch.cat([
                cv_embedding,
                job_embedding,
                torch.abs(cv_embedding - job_embedding)
            ], dim=1)
            logits = self.classifier(features)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return {
            'loss': loss,
            'logits': logits,
            'cv_embedding': cv_embedding,
            'job_embedding': job_embedding
        }


def get_cv_job_match_model(model_type: str, num_classes: int, **kwargs):
    """
    Factory function to get CV-Job matching model by type.

    Args:
        model_type: One of ['lstm-siamese', 'bert', 'roberta', 'sbert']
        num_classes: Number of output classes
        **kwargs: Additional model-specific arguments

    Returns:
        CV-Job matching model instance
    """
    if model_type.lower() == 'lstm-siamese':
        return LSTMSiameseNetwork(num_classes=num_classes, **kwargs)
    elif model_type.lower() == 'bert':
        return BERTForCVJobMatch(num_classes=num_classes, **kwargs)
    elif model_type.lower() == 'roberta':
        return RoBERTaForCVJobMatch(num_classes=num_classes, **kwargs)
    elif model_type.lower() == 'sbert':
        return SentenceBERTBiEncoder(num_classes=num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose from ['lstm-siamese', 'bert', 'roberta', 'sbert']")


if __name__ == "__main__":
    # Example usage
    print("CV-Job Match Model Architectures")
    print("=" * 80)

    num_classes = 3  # No Fit, Potential Fit, Good Fit

    # LSTM Siamese
    print("\n1. LSTM Siamese Network:")
    lstm_model = LSTMSiameseNetwork(
        vocab_size=30000,
        num_classes=num_classes,
        embedding_dim=300,
        hidden_dim=256,
        num_layers=2
    )
    print(f"   Parameters: {sum(p.numel() for p in lstm_model.parameters()):,}")

    # BERT
    print("\n2. BERT for CV-Job Match:")
    bert_model = BERTForCVJobMatch(num_classes=num_classes)
    print(f"   Parameters: {sum(p.numel() for p in bert_model.parameters()):,}")

    # RoBERTa
    print("\n3. RoBERTa for CV-Job Match:")
    roberta_model = RoBERTaForCVJobMatch(num_classes=num_classes)
    print(f"   Parameters: {sum(p.numel() for p in roberta_model.parameters()):,}")

    # Sentence-BERT
    print("\n4. Sentence-BERT Bi-Encoder:")
    sbert_model = SentenceBERTBiEncoder(num_classes=num_classes)
    print(f"   Parameters: {sum(p.numel() for p in sbert_model.parameters()):,}")

    print("\n" + "=" * 80)
