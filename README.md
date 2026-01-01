# NLP Paper Analyzer

A comprehensive toolkit for analyzing academic papers using state-of-the-art NLP techniques.

## Features

### 8 Integrated Phases

1. **Data Processing**: Load and preprocess PeerRead dataset from Kaggle
2. **Embeddings**: Multiple embedding techniques (TF-IDF, Word2Vec, BERT, GloVe, FastText)
3. **Section Classification**: Deep learning models (LSTM, BiLSTM, CNN-BiLSTM, GRU) for automatic section detection
4. **OCR**: Document parsing with Nougat for PDF/image conversion
5. **Grammar Correction**: T5-based grammar fixing trained on JFLEG dataset
6. **Fact Checking**: FEVER-based claim verification for consistency analysis
7. **Unified Pipeline**: End-to-end document processing workflow
8. **Quality Scoring**: Automated paper quality assessment with 5 criteria

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (optional, for faster training)
- Poppler (for PDF processing): `sudo apt-get install poppler-utils` (Linux) or download from [poppler-windows](https://github.com/oschwartz10612/poppler-windows/releases/)

### Install Package

```bash
# Install in development mode
pip install -e .

# Or install dependencies only
pip install -r requirements.txt
```

## Quick Start

### 1. Load and Process Dataset

```python
from nlp_paper_analyzer.data import load_dataset_from_kaggle
import pandas as pd

# Download and load PeerRead dataset
df = load_dataset_from_kaggle(save_csv=True)
print(f"Loaded {len(df)} papers")
```

### 2. Preprocess Text

```python
from nlp_paper_analyzer.data import preprocess_column

# Tokenize reviews
df['reviews_tokens'] = df['reviews'].apply(lambda x: preprocess_column(str(x)))
df['reviews_text'] = df['reviews_tokens'].apply(lambda x: " ".join(x))
```

### 3. Train Section Classifier

```python
from nlp_paper_analyzer.models.section_classifier import train_section_classifier

# Train model (see examples/train_models.py for full workflow)
model, tokenizer, encoder = train_section_classifier(df)
```

### 4. Run Full Pipeline

```python
from nlp_paper_analyzer.pipeline.unified_pipeline import UnifiedDocumentPipeline

# Initialize pipeline with trained models
pipeline = UnifiedDocumentPipeline(
    section_model=model,
    section_tokenizer=tokenizer,
    section_encoder=encoder
)

# Process a document
results = pipeline.run_full_pipeline("path/to/paper.pdf")
```

## Project Structure

```
nlp_paper_analyzer/
├── data/               # Phase 1: Data ingestion and preprocessing
├── embeddings/         # Phase 2: TF-IDF, Word2Vec, BERT, GloVe, FastText
├── models/             # Phase 3: Deep learning architectures
├── ocr/                # Phase 4: Nougat document parsing
├── grammar/            # Phase 5: T5 grammar correction
├── fact_check/         # Phase 6: FEVER fact verification
├── pipeline/           # Phase 7: Unified end-to-end pipeline
├── scoring/            # Phase 8: Paper quality assessment
├── utils/              # Logging and helper utilities
└── config.py           # Centralized configuration
```

## Examples

See the `examples/` directory for complete workflows:

- `run_full_pipeline.py`: End-to-end document analysis
- `train_models.py`: Training workflow for all models
- `analyze_paper.py`: Single paper analysis

## Configuration

All hyperparameters and paths are centralized in `nlp_paper_analyzer/config.py`:

```python
from nlp_paper_analyzer.config import Config

# Access configuration
print(Config.DEVICE)  # cuda or cpu
print(Config.MAX_LEN)  # 300
print(Config.BATCH_SIZE)  # 32

# Ensure directories exist
Config.ensure_dirs()
```

## Model Training

### Section Classification (Phase 3)

Trains LSTM, BiLSTM, CNN-BiLSTM, and GRU models for section detection:

```python
from nlp_paper_analyzer.models import train_section_classifier

model, tokenizer, encoder = train_section_classifier(
    df,
    architecture='cnn_bilstm',  # or 'lstm', 'bilstm', 'gru'
    epochs=10,
    batch_size=32
)
```

### Grammar Correction (Phase 5)

Fine-tune T5 on JFLEG dataset:

```python
from nlp_paper_analyzer.grammar import GrammarCorrector

corrector = GrammarCorrector()
corrector.train(epochs=10)
corrector.save_model("models/t5-grammar/best_model")
```

### Fact Checking (Phase 6)

Train on FEVER dataset:

```python
from nlp_paper_analyzer.fact_check import FactVerifier

verifier = FactVerifier()
verifier.train(epochs=1)
verifier.save_model("models/t5_fever/best_model")
```

## Paper Quality Scoring

The system evaluates papers on 5 criteria (each scored 0-10):

1. **Structure**: Presence of essential sections (Introduction, Methodology, etc.)
2. **Section Order**: Logical flow of sections
3. **Clarity**: Classification confidence (how clearly sections are defined)
4. **Grammar**: Writing quality based on corrections needed
5. **Consistency**: Internal consistency via fact-checking

**Final Score** = Weighted average of all criteria

### Grade Scale

| Score | Grade | Description |
|-------|-------|-------------|
| 9.0-10.0 | A+ | Excellent |
| 8.0-8.9 | A | Very Good |
| 7.0-7.9 | B+ | Good |
| 6.0-6.9 | B | Above Average |
| 5.0-5.9 | C | Average |
| 4.0-4.9 | D | Below Average |
| 0.0-3.9 | F | Needs Improvement |

## API Reference

### Data Module

- `load_dataset_from_kaggle()`: Download and load PeerRead dataset
- `preprocess_column()`: Tokenize and clean text
- `normalize_section_label()`: Standardize section headers

### Pipeline Module

- `UnifiedDocumentPipeline`: End-to-end processing class
  - `step1_ocr()`: Convert PDF/image to text
  - `step2_classify_sections()`: Classify sections
  - `step3_correct_grammar()`: Fix grammar errors
  - `step4_fact_check()`: Verify claims
  - `run_full_pipeline()`: Execute all steps

### Scoring Module

- `PaperQualityScorer`: Quality assessment class
  - `calculate_all_scores()`: Compute all metrics
  - `generate_report()`: Create text report
  - `get_grade()`: Convert score to letter grade

## Citation

If you use this toolkit, please cite the original PeerRead dataset:

```bibtex
@inproceedings{kang2018dataset,
  title={A Dataset of Peer Reviews (PeerRead): Collection, Insights and NLP Applications},
  author={Kang, Dongyeop and Ammar, Waleed and Dalvi, Bhavana and van Zuylen, Madeleine and Kohlmeier, Sebastian and Hovy, Eduard and Schwartz, Roy},
  booktitle={NAACL},
  year={2018}
}
```

## License

This project is for educational and research purposes.

## Acknowledgments

- PeerRead Dataset by AllenAI
- Nougat OCR by Meta
- Transformers library by Hugging Face
- JFLEG and FEVER datasets
