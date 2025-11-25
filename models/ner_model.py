"""
NER Model Architectures
Named Entity Recognition for CV field extraction.

Models:
1. BiLSTM-CRF: Baseline model with word embeddings
2. BERT for Token Classification: State-of-the-art transformer model
3. RoBERTa for Token Classification: Optimized BERT variant
"""

import torch
import torch.nn as nn
from transformers import (
    BertForTokenClassification,
    RobertaForTokenClassification,
    DistilBertForTokenClassification,
    BertConfig,
    RobertaConfig,
    DistilBertConfig
)
from torchcrf import CRF


class BiLSTMCRF(nn.Module):
    """
    BiLSTM-CRF model for Named Entity Recognition.

    Architecture:
    - Embedding layer
    - Bidirectional LSTM
    - Fully connected layer
    - CRF layer for sequence labeling
    """

    def __init__(
        self,
        vocab_size: int,
        num_labels: int,
        embedding_dim: int = 300,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        pretrained_embeddings=None
    ):
        """
        Args:
            vocab_size: Size of vocabulary
            num_labels: Number of BIO tags
            embedding_dim: Dimension of word embeddings
            hidden_dim: Hidden dimension of LSTM
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            pretrained_embeddings: Optional pretrained embeddings (e.g., GloVe)
        """
        super(BiLSTMCRF, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_labels = num_labels

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            self.embedding.weight.requires_grad = True  # Fine-tune embeddings

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim // 2,  # Divide by 2 because bidirectional
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Fully connected layer to map LSTM output to tag space
        self.fc = nn.Linear(hidden_dim, num_labels)

        # CRF layer
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        Forward pass.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Ground truth labels [batch_size, seq_len] (optional)

        Returns:
            If labels provided: Negative log-likelihood loss
            Otherwise: Decoded sequence
        """
        # Embedding
        embedded = self.embedding(input_ids)  # [batch_size, seq_len, embedding_dim]

        # LSTM
        lstm_out, _ = self.lstm(embedded)  # [batch_size, seq_len, hidden_dim]
        lstm_out = self.dropout(lstm_out)

        # Fully connected
        emissions = self.fc(lstm_out)  # [batch_size, seq_len, num_labels]

        # Create mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)

        if labels is not None:
            # Training: compute negative log-likelihood and get predictions
            loss = -self.crf(emissions, labels, mask=attention_mask.bool(), reduction='mean')
            # Also decode predictions for metrics computation
            predictions = self.crf.decode(emissions, mask=attention_mask.bool())
            return {'loss': loss, 'predictions': predictions}
        else:
            # Inference: decode best sequence
            predictions = self.crf.decode(emissions, mask=attention_mask.bool())
            return {'predictions': predictions}


class BERTForNER(nn.Module):
    """
    BERT-based model for Named Entity Recognition.
    Uses HuggingFace's BertForTokenClassification.
    """

    def __init__(
        self,
        num_labels: int,
        model_name: str = 'bert-base-uncased',
        dropout: float = 0.1,
        freeze_bert: bool = False
    ):
        """
        Args:
            num_labels: Number of BIO tags
            model_name: Pretrained BERT model name
            dropout: Dropout probability
            freeze_bert: Whether to freeze BERT parameters
        """
        super(BERTForNER, self).__init__()

        # Load pretrained BERT
        self.bert = BertForTokenClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout
        )

        # Optionally freeze BERT layers
        if freeze_bert:
            for param in self.bert.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        Forward pass.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Ground truth labels [batch_size, seq_len]

        Returns:
            Dictionary with loss and/or logits
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        return {
            'loss': outputs.loss if labels is not None else None,
            'logits': outputs.logits,
            'hidden_states': outputs.hidden_states if hasattr(outputs, 'hidden_states') else None
        }


class RoBERTaForNER(nn.Module):
    """
    RoBERTa-based model for Named Entity Recognition.
    RoBERTa is an optimized variant of BERT with better performance.
    """

    def __init__(
        self,
        num_labels: int,
        model_name: str = 'roberta-base',
        dropout: float = 0.1,
        freeze_encoder: bool = False
    ):
        """
        Args:
            num_labels: Number of BIO tags
            model_name: Pretrained RoBERTa model name
            dropout: Dropout probability
            freeze_encoder: Whether to freeze RoBERTa parameters
        """
        super(RoBERTaForNER, self).__init__()

        # Load pretrained RoBERTa
        self.roberta = RobertaForTokenClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
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
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Ground truth labels [batch_size, seq_len]

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
            'logits': outputs.logits,
            'hidden_states': outputs.hidden_states if hasattr(outputs, 'hidden_states') else None
        }


class DistilBERTForNER(nn.Module):
    """
    DistilBERT-based model for Named Entity Recognition.
    DistilBERT is a distilled version of BERT - faster and lighter with 97% of BERT's performance.
    """

    def __init__(
        self,
        num_labels: int,
        model_name: str = 'distilbert-base-uncased',
        dropout: float = 0.1
    ):
        """
        Args:
            num_labels: Number of BIO tags
            model_name: Pretrained DistilBERT model name
            dropout: Dropout probability
        """
        super(DistilBERTForNER, self).__init__()

        # Load pretrained DistilBERT
        self.distilbert = DistilBertForTokenClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            dropout=dropout
        )

    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        Forward pass.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Ground truth labels [batch_size, seq_len]

        Returns:
            Dictionary with loss and/or logits
        """
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        return {
            'loss': outputs.loss if labels is not None else None,
            'logits': outputs.logits,
            'hidden_states': outputs.hidden_states if hasattr(outputs, 'hidden_states') else None
        }


def get_ner_model(model_type: str, num_labels: int, **kwargs):
    """
    Factory function to get NER model by type.

    Args:
        model_type: One of ['bilstm-crf', 'bert', 'roberta', 'distilbert']
        num_labels: Number of BIO tags
        **kwargs: Additional model-specific arguments

    Returns:
        NER model instance
    """
    if model_type.lower() == 'bilstm-crf':
        return BiLSTMCRF(num_labels=num_labels, **kwargs)
    elif model_type.lower() == 'bert':
        return BERTForNER(num_labels=num_labels, **kwargs)
    elif model_type.lower() == 'roberta':
        return RoBERTaForNER(num_labels=num_labels, **kwargs)
    elif model_type.lower() == 'distilbert':
        return DistilBERTForNER(num_labels=num_labels, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose from ['bilstm-crf', 'bert', 'roberta', 'distilbert']")


if __name__ == "__main__":
    # Example usage
    print("NER Model Architectures")
    print("=" * 80)

    num_labels = 3  # B-SKILL, I-SKILL, O

    # BiLSTM-CRF
    print("\n1. BiLSTM-CRF Model:")
    bilstm_model = BiLSTMCRF(
        vocab_size=10000,
        num_labels=num_labels,
        embedding_dim=300,
        hidden_dim=256,
        num_layers=2,
        dropout=0.3
    )
    print(f"   Parameters: {sum(p.numel() for p in bilstm_model.parameters()):,}")

    # BERT
    print("\n2. BERT for NER:")
    bert_model = BERTForNER(num_labels=num_labels)
    print(f"   Parameters: {sum(p.numel() for p in bert_model.parameters()):,}")

    # RoBERTa
    print("\n3. RoBERTa for NER:")
    roberta_model = RoBERTaForNER(num_labels=num_labels)
    print(f"   Parameters: {sum(p.numel() for p in roberta_model.parameters()):,}")

    # DistilBERT
    print("\n4. DistilBERT for NER:")
    distilbert_model = DistilBERTForNER(num_labels=num_labels)
    print(f"   Parameters: {sum(p.numel() for p in distilbert_model.parameters()):,}")

    print("\n" + "=" * 80)
