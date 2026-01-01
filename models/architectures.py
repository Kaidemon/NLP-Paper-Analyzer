"""
Neural network architectures for section classification (Phase 3).
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding, Bidirectional, LSTM, Dense, Dropout, 
    Conv1D, MaxPooling1D, Input, GRU
)

from nlp_paper_analyzer.config import Config
from nlp_paper_analyzer.utils import logger


def build_bilstm(num_classes: int,
                 max_words: int = None,
                 max_len: int = None,
                 embedding_dim: int = None) -> Sequential:
    """
    Build Bidirectional LSTM architecture.
    
    Args:
        num_classes: Number of output classes
        max_words: Vocabulary size
        max_len: Maximum sequence length
        embedding_dim: Embedding dimension
        
    Returns:
        Compiled Keras model
    """
    if max_words is None:
        max_words = Config.MAX_WORDS
    if max_len is None:
        max_len = Config.MAX_LEN
    if embedding_dim is None:
        embedding_dim = Config.EMBEDDING_DIM
    
    logger.info(f"Building BiLSTM: classes={num_classes}, vocab={max_words}, max_len={max_len}")
    
    model = Sequential([
        Input(shape=(max_len,)),
        Embedding(input_dim=max_words, output_dim=embedding_dim),
        Bidirectional(LSTM(64, return_sequences=False)),
        Dropout(0.5),
        Dense(num_classes, activation="softmax")
    ])
    
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )
    
    return model


def build_cnn_bilstm(num_classes: int,
                     max_words: int = None,
                     max_len: int = None,
                     embedding_dim: int = None) -> Sequential:
    """
    Build CNN + BiLSTM hybrid architecture.
    
    Args:
        num_classes: Number of output classes
        max_words: Vocabulary size
        max_len: Maximum sequence length
        embedding_dim: Embedding dimension
        
    Returns:
        Compiled Keras model
    """
    if max_words is None:
        max_words = Config.MAX_WORDS
    if max_len is None:
        max_len = Config.MAX_LEN
    if embedding_dim is None:
        embedding_dim = Config.EMBEDDING_DIM
    
    logger.info(f"Building CNN-BiLSTM: classes={num_classes}, vocab={max_words}, max_len={max_len}")
    
    model = Sequential([
        Input(shape=(max_len,)),
        Embedding(input_dim=max_words, output_dim=embedding_dim),
        Conv1D(filters=64, kernel_size=5, activation='relu'),
        MaxPooling1D(pool_size=2),
        Bidirectional(LSTM(64)),
        Dropout(0.5),
        Dense(num_classes, activation="softmax")
    ])
    
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )
    
    return model


def build_lstm(num_classes: int,
               max_words: int = None,
               max_len: int = None,
               embedding_dim: int = None) -> Sequential:
    """
    Build standard LSTM architecture.
    
    Args:
        num_classes: Number of output classes
        max_words: Vocabulary size
        max_len: Maximum sequence length
        embedding_dim: Embedding dimension
        
    Returns:
        Compiled Keras model
    """
    if max_words is None:
        max_words = Config.MAX_WORDS
    if max_len is None:
        max_len = Config.MAX_LEN
    if embedding_dim is None:
        embedding_dim = Config.EMBEDDING_DIM
    
    logger.info(f"Building LSTM: classes={num_classes}, vocab={max_words}, max_len={max_len}")
    
    model = Sequential([
        Input(shape=(max_len,)),
        Embedding(input_dim=max_words, output_dim=embedding_dim),
        LSTM(128),
        Dropout(0.5),
        Dense(num_classes, activation="softmax")
    ])
    
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )
    
    return model


def build_gru(num_classes: int,
              max_words: int = None,
              max_len: int = None,
              embedding_dim: int = None) -> Sequential:
    """
    Build GRU architecture.
    
    Args:
        num_classes: Number of output classes
        max_words: Vocabulary size
        max_len: Maximum sequence length
        embedding_dim: Embedding dimension
        
    Returns:
        Compiled Keras model
    """
    if max_words is None:
        max_words = Config.MAX_WORDS
    if max_len is None:
        max_len = Config.MAX_LEN
    if embedding_dim is None:
        embedding_dim = Config.EMBEDDING_DIM
    
    logger.info(f"Building GRU: classes={num_classes}, vocab={max_words}, max_len={max_len}")
    
    model = Sequential([
        Input(shape=(max_len,)),
        Embedding(input_dim=max_words, output_dim=embedding_dim),
        GRU(128),
        Dropout(0.5),
        Dense(num_classes, activation="softmax")
    ])
    
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )
    
    return model


# Architecture registry
ARCHITECTURES = {
    'bilstm': build_bilstm,
    'cnn_bilstm': build_cnn_bilstm,
    'lstm': build_lstm,
    'gru': build_gru
}


def get_architecture(name: str):
    """Get architecture builder by name."""
    if name not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture: {name}. Available: {list(ARCHITECTURES.keys())}")
    return ARCHITECTURES[name]
