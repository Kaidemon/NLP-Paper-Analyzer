"""
Section classification training and utilities (Phase 3).
"""

import pandas as pd
import numpy as np
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from typing import Tuple

from nlp_paper_analyzer.config import Config
from nlp_paper_analyzer.data import extract_and_flatten_sections
from nlp_paper_analyzer.models.architectures import get_architecture
from nlp_paper_analyzer.utils import logger


def balance_dataset(section_df: pd.DataFrame, 
                    target_size: int = None) -> pd.DataFrame:
    """
    Balance dataset using over/undersampling.
    
    Args:
        section_df: DataFrame with 'label' column
        target_size: Target size for each class
        
    Returns:
        Balanced DataFrame
    """
    if target_size is None:
        target_size = Config.TARGET_BALANCE_SIZE
    
    logger.info(f"Balancing dataset to {target_size} samples per class")
    logger.info(f"Original distribution:\n{section_df['label'].value_counts()}")
    
    balanced_dfs = []
    
    for label in section_df['label'].unique():
        df_subset = section_df[section_df['label'] == label]
        
        if len(df_subset) > target_size:
            # Undersample
            resampled = resample(df_subset, replace=False, n_samples=target_size, random_state=42)
        else:
            # Oversample
            resampled = resample(df_subset, replace=True, n_samples=target_size, random_state=42)
        
        balanced_dfs.append(resampled)
    
    final_df = pd.concat(balanced_dfs).sample(frac=1, random_state=42).reset_index(drop=True)
    
    logger.info(f"✓ Balanced dataset size: {len(final_df)}")
    logger.info(f"New distribution:\n{final_df['label'].value_counts()}")
    
    return final_df


def prepare_data(df: pd.DataFrame,
                 max_words: int = None,
                 max_len: int = None,
                 test_size: float = None) -> Tuple:
    """
    Prepare data for training: extract sections, balance, tokenize, split.
    
    Args:
        df: DataFrame with sections column
        max_words: Maximum vocabulary size
        max_len: Maximum sequence length
        test_size: Test set proportion
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, tokenizer, encoder, num_classes)
    """
    if max_words is None:
        max_words = Config.MAX_WORDS
    if max_len is None:
        max_len = Config.MAX_LEN
    if test_size is None:
        test_size = Config.VALIDATION_SPLIT
    
    logger.info("Preparing data for section classification")
    
    # Extract and flatten sections
    flat_data = extract_and_flatten_sections(df)
    section_df = pd.DataFrame(flat_data)
    
    # Balance dataset
    balanced_df = balance_dataset(section_df)
    
    # Prepare X and y
    X = balanced_df['text'].astype(str)
    y = balanced_df['label']
    
    # Encode labels
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    num_classes = len(encoder.classes_)
    
    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Classes: {list(encoder.classes_)}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
    )
    
    # Tokenization
    logger.info(f"Tokenizing with max_words={max_words}, max_len={max_len}")
    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)
    
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    
    X_train_pad = pad_sequences(X_train_seq, maxlen=max_len, padding='post')
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_len, padding='post')
    
    logger.info(f"✓ Data prepared: train={len(X_train_pad)}, test={len(X_test_pad)}")
    
    return X_train_pad, X_test_pad, y_train, y_test, tokenizer, encoder, num_classes


def train_section_classifier(df: pd.DataFrame,
                             architecture: str = 'cnn_bilstm',
                             epochs: int = None,
                             batch_size: int = None,
                             save_path: str = None):
    """
    Complete training pipeline for section classification.
    
    Args:
        df: DataFrame with sections
        architecture: Model architecture ('lstm', 'bilstm', 'cnn_bilstm', 'gru')
        epochs: Number of training epochs
        batch_size: Batch size
        save_path: Path to save best model
        
    Returns:
        Tuple of (model, tokenizer, encoder)
    """
    if epochs is None:
        epochs = Config.EPOCHS
    if batch_size is None:
        batch_size = Config.BATCH_SIZE
    
    logger.info(f"Training section classifier with {architecture} architecture")
    
    # Prepare data
    X_train, X_test, y_train, y_test, tokenizer, encoder, num_classes = prepare_data(df)
    
    # Build model
    arch_builder = get_architecture(architecture)
    model = arch_builder(num_classes)
    
    logger.info(f"Model architecture:\n{model.summary()}")
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
    ]
    
    if save_path:
        callbacks.append(
            ModelCheckpoint(save_path, monitor="val_loss", save_best_only=True, mode="min")
        )
    
    # Train
    logger.info(f"Training for {epochs} epochs with batch_size={batch_size}")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks
    )
    
    # Evaluate
    y_pred = model.predict(X_test).argmax(axis=1)
    accuracy = (y_pred == y_test).mean()
    
    logger.info(f"✓ Training complete. Test accuracy: {accuracy:.4f}")
    
    return model, tokenizer, encoder, history
