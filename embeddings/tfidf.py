"""
TF-IDF vectorization and visualization for Phase 2.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from typing import Tuple, List

from nlp_paper_analyzer.config import Config
from nlp_paper_analyzer.utils import logger


def create_tfidf_matrix(texts: List[str], max_features: int = None) -> Tuple[np.ndarray, np.ndarray, TfidfVectorizer]:
    """
    Create TF-IDF matrix from texts.
    
    Args:
        texts: List of text documents
        max_features: Maximum number of features (default from config)
        
    Returns:
        Tuple of (tfidf_matrix, feature_names, vectorizer)
    """
    if max_features is None:
        max_features = Config.TFIDF_MAX_FEATURES
    
    logger.info(f"Creating TF-IDF matrix with max_features={max_features}")
    
    vectorizer = TfidfVectorizer(max_features=max_features)
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    
    logger.info(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
    
    return tfidf_matrix, feature_names, vectorizer


def visualize_tfidf_tsne(feature_names: np.ndarray, 
                         vectorizer: TfidfVectorizer,
                         n_labels: int = 50,
                         save_path: str = None):
    """
    Visualize TF-IDF word vectors using t-SNE.
    
    Args:
        feature_names: Array of feature names
        vectorizer: Fitted TfidfVectorizer
        n_labels: Number of words to label
        save_path: Optional path to save figure
    """
    logger.info("Creating t-SNE visualization of TF-IDF vectors")
    
    # Get term vectors
    term_vecs = vectorizer.transform(feature_names).toarray()
    
    # Apply t-SNE
    perplexity = min(30, len(feature_names) - 1)
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, init='pca')
    tsne_emb = tsne.fit_transform(term_vecs)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.scatter(tsne_emb[:, 0], tsne_emb[:, 1], c='darkgreen', alpha=0.6)
    
    # Label first n words
    for i, term in enumerate(feature_names[:n_labels]):
        plt.annotate(term, (tsne_emb[i, 0], tsne_emb[i, 1]), fontsize=9)
    
    plt.title('t-SNE: TF-IDF Word Vectors')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved visualization to {save_path}")
    
    plt.show()
    
    logger.info("✓ TF-IDF visualization complete")
