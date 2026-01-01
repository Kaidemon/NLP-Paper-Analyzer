"""
GloVe embeddings for Phase 2.
"""

import numpy as np
import matplotlib.pyplot as plt
import gensim.downloader as api
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from typing import List, Tuple

from nlp_paper_analyzer.config import Config
from nlp_paper_analyzer.utils import logger


def load_glove_model(model_name: str = None):
    """
    Load pre-trained GloVe model.
    
    Args:
        model_name: GloVe model identifier (default from config)
        
    Returns:
        Loaded GloVe model
    """
    if model_name is None:
        model_name = Config.GLOVE_MODEL
    
    logger.info(f"Loading GloVe model: {model_name}")
    model = api.load(model_name)
    logger.info(f"✓ GloVe loaded. Vocabulary size: {len(model.key_to_index)}")
    
    return model


def get_word_vectors(words: List[str], glove_model) -> Tuple[List[str], np.ndarray]:
    """
    Get GloVe vectors for words that exist in vocabulary.
    
    Args:
        words: List of words
        glove_model: Loaded GloVe model
        
    Returns:
        Tuple of (valid_words, vectors)
    """
    valid_words = [w for w in words if w in glove_model.key_to_index]
    vectors = np.array([glove_model[w] for w in valid_words])
    
    logger.info(f"Found {len(valid_words)}/{len(words)} words in GloVe vocabulary")
    
    return valid_words, vectors


def visualize_glove_tsne(words: List[str], 
                         vectors: np.ndarray,
                         save_path: str = None):
    """
    Visualize GloVe vectors using t-SNE.
    
    Args:
        words: List of words
        vectors: Word vectors
        save_path: Optional path to save figure
    """
    logger.info("Creating t-SNE visualization of GloVe vectors")
    
    perplexity = min(30, len(words) - 1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, init='pca')
    tsne_emb = tsne.fit_transform(vectors)
    
    plt.figure(figsize=(12, 10))
    plt.scatter(tsne_emb[:, 0], tsne_emb[:, 1], color='purple', alpha=0.7)
    
    for i, w in enumerate(words):
        plt.annotate(w, (tsne_emb[i, 0], tsne_emb[i, 1]), fontsize=8)
    
    plt.title('t-SNE: GloVe on Tokens')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved visualization to {save_path}")
    
    plt.show()


def visualize_glove_pca(words: List[str], 
                        vectors: np.ndarray,
                        save_path: str = None):
    """
    Visualize GloVe vectors using PCA.
    
    Args:
        words: List of words
        vectors: Word vectors
        save_path: Optional path to save figure
    """
    logger.info("Creating PCA visualization of GloVe vectors")
    
    pca = PCA(n_components=2)
    pca_emb = pca.fit_transform(vectors)
    
    plt.figure(figsize=(12, 10))
    plt.scatter(pca_emb[:, 0], pca_emb[:, 1], color='orange', alpha=0.7)
    
    for i, w in enumerate(words):
        plt.annotate(w, (pca_emb[i, 0], pca_emb[i, 1]), fontsize=8)
    
    plt.title('PCA: GloVe on Tokens')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved visualization to {save_path}")
    
    plt.show()
