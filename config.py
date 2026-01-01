"""
Configuration management for NLP Paper Analyzer.
"""

import os
import torch
from pathlib import Path


class Config:
    """Central configuration for the NLP Paper Analyzer."""
    
    # Project paths
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    MODELS_DIR = PROJECT_ROOT / "models"
    OUTPUTS_DIR = PROJECT_ROOT / "outputs"
    
    # Device configuration
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Phase 1: Data Processing
    KAGGLE_DATASET = "jonauskis/asap-review"
    
    # Phase 2: Embeddings
    TFIDF_MAX_FEATURES = 500
    W2V_VECTOR_SIZE = 100
    W2V_WINDOW = 5
    W2V_MIN_COUNT = 5
    BERT_MODEL = "all-MiniLM-L6-v2"
    GLOVE_MODEL = "glove-wiki-gigaword-100"
    FASTTEXT_VECTOR_SIZE = 100
    
    # Phase 3: Section Classification
    MAX_WORDS = 20000
    MAX_LEN = 300
    EMBEDDING_DIM = 128
    BATCH_SIZE = 32
    EPOCHS = 10
    VALIDATION_SPLIT = 0.2
    TARGET_BALANCE_SIZE = 3000
    
    # Expected academic sections
    EXPECTED_SECTIONS = [
        'Introduction',
        'Related Work',
        'Methodology',
        'Experiments',
        'Conclusion',
        'Appendix'
    ]
    
    # Phase 4: OCR
    NOUGAT_MODEL = "facebook/nougat-base"
    OCR_MAX_TOKENS = 4000
    
    # Phase 5: Grammar Correction
    GRAMMAR_MODEL = "t5-base"
    GRAMMAR_MAX_INPUT_LEN = 256
    GRAMMAR_MAX_TARGET_LEN = 256
    GRAMMAR_LEARNING_RATE = 5e-5
    GRAMMAR_MODEL_PATH = MODELS_DIR / "t5-grammar" / "best_model"
    
    # Phase 6: Fact Checking
    FACT_MODEL = "t5-small"
    FACT_MAX_LEN = 512
    FACT_LEARNING_RATE = 5e-5
    FACT_MODEL_PATH = MODELS_DIR / "t5_fever" / "best_model"
    
    # Phase 7: Pipeline
    PIPELINE_MAX_GRAMMAR_CHARS = 500
    
    # Phase 8: Scoring
    SCORING_WEIGHTS = {
        'structure': 1.0,
        'section_order': 1.0,
        'classification_confidence': 1.0,
        'grammar_quality': 1.0,
        'consistency': 1.0
    }
    
    @classmethod
    def ensure_dirs(cls):
        """Create necessary directories if they don't exist."""
        for dir_path in [cls.DATA_DIR, cls.MODELS_DIR, cls.OUTPUTS_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_model_path(cls, phase: str) -> Path:
        """Get the path for a specific phase's model."""
        paths = {
            'grammar': cls.GRAMMAR_MODEL_PATH,
            'fact_check': cls.FACT_MODEL_PATH,
        }
        return paths.get(phase, cls.MODELS_DIR / phase)
