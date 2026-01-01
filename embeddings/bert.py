"""
BERT embeddings using Sentence-BERT for Phase 2.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List

from nlp_paper_analyzer.config import Config
from nlp_paper_analyzer.utils import logger


def load_bert_model(model_name: str = None) -> SentenceTransformer:
    """
    Load pre-trained Sentence-BERT model.
    
    Args:
        model_name: Model identifier (default from config)
        
    Returns:
        Loaded SentenceTransformer model
    """
    if model_name is None:
        model_name = Config.BERT_MODEL
    
    logger.info(f"Loading BERT model: {model_name}")
    model = SentenceTransformer(model_name)
    logger.info("✓ BERT model loaded")
    
    return model


def encode_texts(texts: List[str], 
                 model: SentenceTransformer = None,
                 show_progress: bool = True) -> np.ndarray:
    """
    Encode texts using BERT.
    
    Args:
        texts: List of text strings
        model: SentenceTransformer model (loads default if None)
        show_progress: Show progress bar
        
    Returns:
        Array of embeddings
    """
    if model is None:
        model = load_bert_model()
    
    logger.info(f"Encoding {len(texts)} texts with BERT")
    
    embeddings = model.encode(texts, show_progress_bar=show_progress)
    
    logger.info(f"✓ BERT embeddings shape: {embeddings.shape}")
    
    return embeddings
