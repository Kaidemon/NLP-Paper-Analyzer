"""
Word2Vec embeddings for Phase 2.
"""

from gensim.models import Word2Vec
from pathlib import Path
from typing import List

from nlp_paper_analyzer.config import Config
from nlp_paper_analyzer.utils import logger


def train_word2vec(sentences: List[List[str]], 
                   vector_size: int = None,
                   window: int = None,
                   min_count: int = None,
                   workers: int = 4) -> Word2Vec:
    """
    Train Word2Vec model on tokenized sentences.
    
    Args:
        sentences: List of tokenized sentences
        vector_size: Dimensionality of word vectors
        window: Maximum distance between current and predicted word
        min_count: Ignores words with frequency lower than this
        workers: Number of worker threads
        
    Returns:
        Trained Word2Vec model
    """
    if vector_size is None:
        vector_size = Config.W2V_VECTOR_SIZE
    if window is None:
        window = Config.W2V_WINDOW
    if min_count is None:
        min_count = Config.W2V_MIN_COUNT
    
    logger.info(f"Training Word2Vec: vector_size={vector_size}, window={window}, min_count={min_count}")
    
    model = Word2Vec(
        sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers
    )
    
    vocab_size = len(model.wv.key_to_index)
    logger.info(f"✓ Word2Vec training complete. Vocabulary size: {vocab_size}")
    
    return model


def save_word2vec(model: Word2Vec, save_path: Path):
    """Save Word2Vec model."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(save_path))
    logger.info(f"Saved Word2Vec model to {save_path}")


def load_word2vec(model_path: Path) -> Word2Vec:
    """Load Word2Vec model."""
    model = Word2Vec.load(str(model_path))
    logger.info(f"Loaded Word2Vec model from {model_path}")
    return model
