"""
ATS Scoring Model Architectures
Predicting ATS (Applicant Tracking System) compatibility scores.

Models:
1. LSTM Regression: Baseline with learned embeddings
2. BERT for Regression: Fine-tuned BERT for score prediction
3. Sentence-BERT for Similarity: Efficient semantic similarity scoring
4. RoBERTa for Regression: State-of-the-art transformer regression
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    BertModel,
    RobertaModel,
    DistilBertModel,
    BertConfig,
    RobertaConfig,
    DistilBertConfig
)


class LSTMRegression(nn.Module):
    """
    LSTM-based model for ATS score regression.

    Architecture:
    - Embedding layer
    - Bidirectional LSTM
    - Attention mechanism
    - Fully connected regressor
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 300,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        pretrained_embeddings=None,
        use_attention: bool = True
    ):
        """
        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of word embeddings
            hidden_dim: Hidden dimension of LSTM
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            pretrained_embeddings: Optional pretrained embeddings
            use_attention: Whether to use attention mechanism
        """
        super(LSTMRegression, self).__init__()

        self.hidden_dim = hidden_dim
        self.use_attention = use_attention

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            self.embedding.weight.requires_grad = True

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim // 2,  # Divide by 2 because bidirectional
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Attention mechanism
        if use_attention:
            self.attention = nn.Linear(hidden_dim, 1)

        self.dropout = nn.Dropout(dropout)

        # Regression head
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1)  # Output: single score
        )

    def attention_mechanism(self, lstm_out, attention_mask):
        """
        Apply attention mechanism to LSTM outputs.

        Args:
            lstm_out: LSTM outputs [batch_size, seq_len, hidden_dim]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            Context vector [batch_size, hidden_dim]
        """
        # Compute attention scores
        attention_scores = self.attention(lstm_out).squeeze(-1)  # [batch_size, seq_len]

        # Mask padded tokens
        attention_scores = attention_scores.masked_fill(attention_mask == 0, -1e9)

        # Softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=1)  # [batch_size, seq_len]

        # Weighted sum of LSTM outputs
        context = torch.bmm(attention_weights.unsqueeze(1), lstm_out).squeeze(1)  # [batch_size, hidden_dim]

        return context

    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        Forward pass.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Ground truth scores [batch_size] (optional)

        Returns:
            Dictionary with loss and/or predictions
        """
        # Embedding
        embedded = self.embedding(input_ids)  # [batch_size, seq_len, embedding_dim]

        # LSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)  # [batch_size, seq_len, hidden_dim]
        lstm_out = self.dropout(lstm_out)

        # Pooling
        if self.use_attention:
            encoding = self.attention_mechanism(lstm_out, attention_mask)
        else:
            # Mean pooling
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).expand(lstm_out.size()).float()
                masked_lstm_out = lstm_out * mask
                sum_lstm = torch.sum(masked_lstm_out, dim=1)
                sum_mask = torch.sum(mask, dim=1).clamp(min=1e-9)
                encoding = sum_lstm / sum_mask
            else:
                encoding = torch.mean(lstm_out, dim=1)

        # Regression
        predictions = self.regressor(encoding).squeeze(-1)  # [batch_size]

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss_fct = nn.MSELoss()
            loss = loss_fct(predictions, labels)

        return {
            'loss': loss,
            'predictions': predictions
        }


class BERTForRegression(nn.Module):
    """
    BERT-based model for ATS score regression.
    Uses BERT encoder with a regression head.
    """

    def __init__(
        self,
        model_name: str = 'bert-base-uncased',
        dropout: float = 0.1,
        freeze_bert: bool = False,
        use_pooler: bool = False
    ):
        """
        Args:
            model_name: Pretrained BERT model name
            dropout: Dropout probability
            freeze_bert: Whether to freeze BERT parameters
            use_pooler: Whether to use BERT's pooler output (CLS token)
        """
        super(BERTForRegression, self).__init__()

        self.use_pooler = use_pooler

        # Load pretrained BERT
        self.bert = BertModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size

        # Optionally freeze BERT layers
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        self.dropout = nn.Dropout(dropout)

        # Regression head
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, 1)
        )

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        """
        Forward pass.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask
            token_type_ids: Segment IDs (optional)
            labels: Ground truth scores [batch_size]

        Returns:
            Dictionary with loss and/or predictions
        """
        # BERT encoding
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        if self.use_pooler:
            # Use pooler output (CLS token representation)
            encoding = outputs.pooler_output
        else:
            # Use mean pooling over all tokens
            token_embeddings = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
            mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            masked_embeddings = token_embeddings * mask
            sum_embeddings = torch.sum(masked_embeddings, dim=1)
            sum_mask = torch.sum(mask, dim=1).clamp(min=1e-9)
            encoding = sum_embeddings / sum_mask

        encoding = self.dropout(encoding)

        # Regression
        predictions = self.regressor(encoding).squeeze(-1)  # [batch_size]

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss_fct = nn.MSELoss()
            loss = loss_fct(predictions, labels)

        return {
            'loss': loss,
            'predictions': predictions,
            'encoding': encoding
        }


class RoBERTaForRegression(nn.Module):
    """
    RoBERTa-based model for ATS score regression.
    RoBERTa is an optimized variant of BERT with better performance.
    """

    def __init__(
        self,
        model_name: str = 'roberta-base',
        dropout: float = 0.1,
        freeze_encoder: bool = False
    ):
        """
        Args:
            model_name: Pretrained RoBERTa model name
            dropout: Dropout probability
            freeze_encoder: Whether to freeze RoBERTa parameters
        """
        super(RoBERTaForRegression, self).__init__()

        # Load pretrained RoBERTa
        self.roberta = RobertaModel.from_pretrained(model_name)
        hidden_size = self.roberta.config.hidden_size

        # Optionally freeze encoder layers
        if freeze_encoder:
            for param in self.roberta.parameters():
                param.requires_grad = False

        self.dropout = nn.Dropout(dropout)

        # Regression head
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 4, 1)
        )

    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        Forward pass.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask
            labels: Ground truth scores [batch_size]

        Returns:
            Dictionary with loss and/or predictions
        """
        # RoBERTa encoding
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Mean pooling
        token_embeddings = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        masked_embeddings = token_embeddings * mask
        sum_embeddings = torch.sum(masked_embeddings, dim=1)
        sum_mask = torch.sum(mask, dim=1).clamp(min=1e-9)
        encoding = sum_embeddings / sum_mask

        encoding = self.dropout(encoding)

        # Regression
        predictions = self.regressor(encoding).squeeze(-1)  # [batch_size]

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss_fct = nn.MSELoss()
            loss = loss_fct(predictions, labels)

        return {
            'loss': loss,
            'predictions': predictions,
            'encoding': encoding
        }


class SentenceBERTForSimilarity(nn.Module):
    """
    Sentence-BERT for computing semantic similarity scores.

    Architecture:
    - Separate BERT encoders for resume and job description
    - Cosine similarity between embeddings
    - Optional regression head to map similarity to ATS score
    """

    def __init__(
        self,
        model_name: str = 'bert-base-uncased',
        pooling: str = 'mean',
        use_regression_head: bool = True
    ):
        """
        Args:
            model_name: Pretrained BERT model name
            pooling: Pooling strategy ('mean', 'cls', 'max')
            use_regression_head: Use a regression head on top of cosine similarity
        """
        super(SentenceBERTForSimilarity, self).__init__()

        self.pooling = pooling
        self.use_regression_head = use_regression_head

        # Shared BERT encoder
        self.bert = BertModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size

        # Optional regression head
        if use_regression_head:
            self.regressor = nn.Sequential(
                nn.Linear(hidden_size * 3 + 1, hidden_size),  # [resume; job; |resume-job|; cosine_sim]
                nn.Tanh(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.Tanh(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size // 2, 1)
            )

    def mean_pooling(self, model_output, attention_mask):
        """Mean pooling with attention mask."""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = input_mask_expanded.sum(dim=1).clamp(min=1e-9)
        return sum_embeddings / sum_mask

    def encode(self, input_ids, attention_mask):
        """Encode a sequence to a fixed-size embedding."""
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        if self.pooling == 'mean':
            embedding = self.mean_pooling(outputs, attention_mask)
        elif self.pooling == 'cls':
            embedding = outputs[0][:, 0]
        elif self.pooling == 'max':
            embedding, _ = torch.max(outputs[0], dim=1)
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling}")

        # Normalize embeddings
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding

    def forward(self, resume_input_ids, resume_attention_mask, job_input_ids, job_attention_mask, labels=None):
        """
        Forward pass.

        Args:
            resume_input_ids: Resume token IDs
            resume_attention_mask: Resume attention mask
            job_input_ids: Job description token IDs
            job_attention_mask: Job description attention mask
            labels: Ground truth scores [batch_size]

        Returns:
            Dictionary with loss and/or predictions
        """
        # Encode resume and job description
        resume_embedding = self.encode(resume_input_ids, resume_attention_mask)
        job_embedding = self.encode(job_input_ids, job_attention_mask)

        # Compute cosine similarity
        cosine_sim = F.cosine_similarity(resume_embedding, job_embedding, dim=1, eps=1e-8)

        if self.use_regression_head:
            # Use regression head
            features = torch.cat([
                resume_embedding,
                job_embedding,
                torch.abs(resume_embedding - job_embedding),
                cosine_sim.unsqueeze(-1)
            ], dim=1)
            predictions = self.regressor(features).squeeze(-1)
        else:
            # Use cosine similarity directly as score
            predictions = cosine_sim

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss_fct = nn.MSELoss()
            loss = loss_fct(predictions, labels)

        return {
            'loss': loss,
            'predictions': predictions,
            'cosine_similarity': cosine_sim,
            'resume_embedding': resume_embedding,
            'job_embedding': job_embedding
        }


def get_ats_model(model_type: str, **kwargs):
    """
    Factory function to get ATS scoring model by type.

    Args:
        model_type: One of ['lstm', 'bert', 'roberta', 'sbert']
        **kwargs: Additional model-specific arguments

    Returns:
        ATS scoring model instance
    """
    if model_type.lower() == 'lstm':
        return LSTMRegression(**kwargs)
    elif model_type.lower() == 'bert':
        return BERTForRegression(**kwargs)
    elif model_type.lower() == 'roberta':
        return RoBERTaForRegression(**kwargs)
    elif model_type.lower() == 'sbert':
        return SentenceBERTForSimilarity(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose from ['lstm', 'bert', 'roberta', 'sbert']")


if __name__ == "__main__":
    # Example usage
    print("ATS Scoring Model Architectures")
    print("=" * 80)

    # LSTM Regression
    print("\n1. LSTM Regression:")
    lstm_model = LSTMRegression(
        vocab_size=30000,
        embedding_dim=300,
        hidden_dim=256,
        num_layers=2,
        use_attention=True
    )
    print(f"   Parameters: {sum(p.numel() for p in lstm_model.parameters()):,}")

    # BERT Regression
    print("\n2. BERT for Regression:")
    bert_model = BERTForRegression()
    print(f"   Parameters: {sum(p.numel() for p in bert_model.parameters()):,}")

    # RoBERTa Regression
    print("\n3. RoBERTa for Regression:")
    roberta_model = RoBERTaForRegression()
    print(f"   Parameters: {sum(p.numel() for p in roberta_model.parameters()):,}")

    # Sentence-BERT
    print("\n4. Sentence-BERT for Similarity:")
    sbert_model = SentenceBERTForSimilarity(use_regression_head=True)
    print(f"   Parameters: {sum(p.numel() for p in sbert_model.parameters()):,}")

    print("\n" + "=" * 80)
