# NLP CV Evaluation System

An end-to-end NLP project for automated CV evaluation using transformer-based models.

## Project Overview

This project implements an HR agent system that evaluates applicant CVs through three key tasks:
1. **Named Entity Recognition (NER)**: Extract structured fields from CVs (skills, experience, education, etc.)
2. **CV-Job Matching**: Evaluate how well a CV matches a job description
3. **ATS Scoring**: Predict Applicant Tracking System compatibility scores

## Directory Structure

```
NLP_Project/
├── data/                           # Original datasets
│   ├── ner/                       # NER dataset (5,029 resumes)
│   ├── cv_job_match/              # CV-Job matching dataset (8,000 pairs)
│   └── ats/                       # ATS scoring dataset (6,374 samples)
│
├── notebooks/                      # Data analysis notebooks
│   ├── ner_analysis.ipynb         # NER dataset exploration
│   ├── cv_job_match_analysis.ipynb # CV-Job matching analysis
│   └── ats_analysis.ipynb         # ATS dataset exploration
│
├── pre-processing/                 # Preprocessing scripts
│   ├── ner_preprocessing.py       # NER data preprocessing
│   ├── cv_job_match_preprocessing.py  # CV-Job match preprocessing
│   └── ats_preprocessing.py       # ATS data preprocessing
│
├── pre-processed-data/            # Preprocessed datasets (445MB)
│   ├── ner/                       # BIO-tagged NER data
│   ├── cv_job_match/              # Cleaned CV-Job pairs
│   └── ats/                       # Normalized ATS scores
│
├── models/                        # Model architectures
│   ├── ner_model.py              # NER models (BiLSTM-CRF, BERT, RoBERTa, DistilBERT)
│   ├── cv_job_match_model.py    # Matching models (Siamese LSTM, BERT, RoBERTa, SBERT)
│   └── ats_model.py              # Regression models (LSTM, BERT, RoBERTa, SBERT)
│
├── model_training/                # Training scripts
│   ├── train_ner.py              # Train NER models
│   ├── train_cv_job_match.py    # Train matching models
│   └── train_ats.py              # Train ATS models
│
└── trained_models/                # Saved trained models (will be created during training)
    ├── ner/
    │   ├── bert/
    │   │   ├── model.pth
    │   │   ├── config.json
    │   │   ├── tokenizer_config.json
    │   │   └── training_history.json
    │   ├── roberta/
    │   │   └── ...
    │   ├── distilbert/
    │   │   └── ...
    │   └── bilstm_crf/
    │       └── ...
    ├── cv_job_match/
    │   ├── bert/
    │   │   ├── model.pth
    │   │   ├── config.json
    │   │   ├── tokenizer_config.json
    │   │   └── training_history.json
    │   ├── roberta/
    │   │   └── ...
    │   ├── sbert/
    │   │   └── ...
    │   └── lstm_siamese/
    │       └── ...
    └── ats/
        ├── bert/
        │   ├── model.pth
        │   ├── config.json
        │   ├── tokenizer_config.json
        │   └── training_history.json
        ├── roberta/
        │   └── ...
        ├── sbert/
        │   └── ...
        └── lstm_attention/
            └── ...
```

## Dataset Statistics

### 1. NER Dataset
- **Total Resumes**: 4,971 (after filtering)
- **Train/Val/Test Split**: 3,479 / 746 / 746
- **Entity Types**: SKILL (399,182 annotations)
- **BIO Tags**: B-SKILL, I-SKILL, O
- **Average Tokens**: ~580 per resume

### 2. CV-Job Match Dataset
- **Total Pairs**: 8,000
- **Train/Val/Test Split**: 5,304 / 937 / 1,759
- **Classes**:
  - No Fit: 50.4%
  - Potential Fit: 24.9%
  - Good Fit: 24.7%
- **Average Text Length**:
  - Resume: ~723 words
  - Job Description: ~371 words

### 3. ATS Scoring Dataset
- **Total Samples**: 6,374
- **Train/Val Split**: 5,099 / 1,275
- **Score Range**: 19.16 - 90.05
- **Task Type**: Regression (continuous score prediction)
- **Average Text Length**: ~1,099 words (combined)

## Dataset Sources

The datasets used in this project are available on Hugging Face:

1. **ATS Scoring Dataset**: https://huggingface.co/datasets/0xnbk/resume-ats-score-v1-en/tree/main
2. **CV-Job Match Dataset**: https://huggingface.co/datasets/cnamuangtoun/resume-job-description-fit/tree/main
3. **NER Dataset**: https://huggingface.co/datasets/Mehyaar/Annotated_NER_PDF_Resumes

## Model Architectures

### NER Models
1. **BiLSTM-CRF**: Baseline model with word embeddings and CRF layer
2. **BERT**: Transformer-based token classification
3. **RoBERTa**: Optimized BERT variant
4. **DistilBERT**: Lightweight, fast model (97% of BERT performance)

### CV-Job Match Models
1. **LSTM Siamese Network**: Dual encoders with feature engineering
2. **BERT**: Fine-tuned for text pair classification
3. **RoBERTa**: Enhanced transformer for sequence classification
4. **Sentence-BERT**: Efficient bi-encoder architecture

### ATS Scoring Models
1. **LSTM Regression**: With attention mechanism
2. **BERT Regression**: Fine-tuned with regression head
3. **RoBERTa Regression**: Optimized transformer regression
4. **Sentence-BERT**: Cosine similarity + regression head

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchvision torchaudio
pip install transformers
pip install pandas numpy scikit-learn
pip install tqdm matplotlib seaborn
pip install wordcloud jupyter
pip install pytorch-crf
```

## Usage

### 1. Data Preprocessing

```bash
# Preprocess NER data
python pre-processing/ner_preprocessing.py

# Preprocess CV-Job Match data
python pre-processing/cv_job_match_preprocessing.py

# Preprocess ATS data
python pre-processing/ats_preprocessing.py
```

### 2. Model Training

#### Train NER Model
```bash
# Train BERT for NER
python model_training/train_ner.py \
    --model_type bert \
    --epochs 10 \
    --batch_size 16 \
    --learning_rate 2e-5

# Train RoBERTa for NER
python model_training/train_ner.py \
    --model_type roberta \
    --epochs 10 \
    --batch_size 16

# Train BiLSTM-CRF
python model_training/train_ner.py \
    --model_type bilstm-crf \
    --epochs 20 \
    --batch_size 32 \
    --vocab_size 30000
```

#### Train CV-Job Match Model
```bash
# Train BERT for matching
python model_training/train_cv_job_match.py \
    --model_type bert \
    --epochs 5 \
    --batch_size 8 \
    --max_length 256

# Train Sentence-BERT
python model_training/train_cv_job_match.py \
    --model_type sbert \
    --epochs 5 \
    --batch_size 8
```

#### Train ATS Model
```bash
# Train BERT for regression
python model_training/train_ats.py \
    --model_type bert \
    --epochs 5 \
    --batch_size 8 \
    --use_normalized_scores

# Train RoBERTa for regression
python model_training/train_ats.py \
    --model_type roberta \
    --epochs 5 \
    --batch_size 8
```

### 3. Explore Data

```bash
# Launch Jupyter Notebook
jupyter notebook

# Open and run analysis notebooks:
# - notebooks/ner_analysis.ipynb
# - notebooks/cv_job_match_analysis.ipynb
# - notebooks/ats_analysis.ipynb
```

## Model Selection Justification

### Why BERT, RoBERTa, and DistilBERT?

1. **BERT (Bidirectional Encoder Representations from Transformers)**
   - Pre-trained on massive text corpora
   - Bidirectional context understanding
   - State-of-the-art for NLP tasks
   - Good balance of performance and efficiency

2. **RoBERTa (Robustly Optimized BERT)**
   - Improved training methodology over BERT
   - Better performance on benchmarks
   - More robust to hyperparameters
   - Recommended for highest accuracy

3. **DistilBERT**
   - 40% smaller than BERT
   - 60% faster inference
   - Retains 97% of BERT's performance
   - Ideal for production deployment

4. **BiLSTM-CRF (for NER)**
   - Proven architecture for sequence labeling
   - CRF layer captures label dependencies
   - Good baseline for comparison
   - Lower computational cost

5. **Sentence-BERT (for matching/similarity)**
   - Specialized for semantic similarity
   - Efficient bi-encoder architecture
   - Fast inference for large-scale matching
   - Pre-trained on sentence pairs

## Training Features

- **Automatic GPU detection** (CUDA support)
- **Mixed precision training** (optional)
- **Learning rate scheduling** with warmup
- **Gradient clipping** for stability
- **Class weighting** for imbalanced data
- **Checkpointing** every N epochs
- **Best model saving** based on validation metrics
- **Training history logging** (JSON format)
- **Progress bars** with tqdm

## Evaluation Metrics

### NER
- Precision, Recall, F1-score (weighted)
- Per-class metrics for each entity type
- Token-level accuracy

### CV-Job Matching
- Accuracy
- Precision, Recall, F1-score (weighted)
- Confusion matrix
- Per-class classification report

### ATS Scoring
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R² Score

## Hyperparameter Tuning

Key hyperparameters to tune:
- Learning rate: `[1e-5, 2e-5, 3e-5, 5e-5]`
- Batch size: `[8, 16, 32]`
- Max sequence length: `[128, 256, 512]`
- Dropout: `[0.1, 0.2, 0.3]`
- Number of epochs: `[3, 5, 10]`
- Weight decay: `[0.01, 0.001]`

## Future Enhancements

1. **PDF Scanning**: Add OCR capabilities for extracting text from PDF CVs
2. **Multi-task Learning**: Joint training across all three tasks
3. **Active Learning**: Human-in-the-loop for difficult examples
4. **Model Ensembling**: Combine predictions from multiple models
5. **API Development**: REST API for production deployment
6. **Web Interface**: User-friendly UI for HR agents
7. **Explainability**: LIME/SHAP for model interpretation
8. **Continuous Learning**: Update models with new data

## Requirements

- Python 3.8+
- PyTorch 1.10+
- Transformers 4.20+
- CUDA 11.0+ (optional, for GPU acceleration)
- 16GB+ RAM recommended
- GPU with 8GB+ VRAM recommended for training

## Project Timeline

- **Step 1**: Data Analysis (Completed)
- **Step 2**: Data Preprocessing (Completed)
- **Step 3**: Model Architecture Design (Completed)
- **Step 4**: Model Training Infrastructure (Completed)
- **Step 5**: Model Training (Next)
- **Step 6**: Model Evaluation (Next)
- **Step 7**: Deployment (Future)

## License

This project is for educational purposes.

## Contact

For questions or issues, please open an issue in the repository.

---

**Note**: This is a comprehensive NLP project demonstrating end-to-end machine learning pipeline development, including data preprocessing, model architecture design, training infrastructure, and evaluation.
