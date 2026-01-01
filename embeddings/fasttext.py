"""
FastText embeddings for Phase 2.
"""

import numpy as np
import matplotlib.pyplot as plt
from gensim.models import FastText
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from pathlib import Path
from typing import List

from nlp_paper_analyzer.config import Config
from nlp_paper_analyzer.utils import logger


def train_fasttext(sentences: List[List[str]],
                   vector_size: int = None,
                   window: int = None,
                   min_count: int = None,
                   workers: int = 4,
                   sg: int = 0,
                   epochs: int = 10) -> FastText:
    """
    Train FastText model on tokenized sentences.
    
    Args:
        sentences: List of tokenized sentences
        vector_size: Dimensionality of word vectors
        window: Maximum distance between current and predicted word
        min_count: Ignores words with frequency lower than this
        workers: Number of worker threads
        sg: Training algorithm (0=CBOW, 1=skip-gram)
        epochs: Number of training epochs
        
    Returns:
        Trained FastText model
    """
    if vector_size is None:
        vector_size = Config.FASTTEXT_VECTOR_SIZE
    if window is None:
        window = Config.W2V_WINDOW
    if min_count is None:
        min_count = Config.W2V_MIN_COUNT
    
    logger.info(f"Training FastText: vector_size={vector_size}, window={window}, epochs={epochs}")
    
    model = FastText(
        sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=sg,
        epochs=epochs
    )
    
    vocab_size = len(model.wv.key_to_index)
    logger.info(f"✓ FastText training complete. Vocabulary size: {vocab_size}")
    
    return model


def save_fasttext(model: FastText, save_path: Path):
    """Save FastText model."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(save_path))
    logger.info(f"Saved FastText model to {save_path}")


def load_fasttext(model_path: Path) -> FastText:
    """Load FastText model."""
    model = FastText.load(str(model_path))
    logger.info(f"Loaded FastText model from {model_path}")
    return model


def visualize_fasttext_tsne(model: FastText, 
                            n_words: int = 200,
                            save_path: str = None):
    """
    Visualize FastText vectors using t-SNE.
    
    Args:
        model: Trained FastText model
        n_words: Number of words to visualize
        save_path: Optional path to save figure
    """
    logger.info(f"Creating t-SNE visualization of FastText vectors ({n_words} words)")
    
    vocab = list(model.wv.key_to_index.keys())[:n_words]
    vectors = np.array([model.wv[w] for w in vocab])
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, init='pca')
    tsne_emb = tsne.fit_transform(vectors)
    
    plt.figure(figsize=(12, 10))
    plt.scatter(tsne_emb[:, 0], tsne_emb[:, 1], color='magenta', alpha=0.7)
    
    for i, w in enumerate(vocab):
        plt.annotate(w, (tsne_emb[i, 0], tsne_emb[i, 1]), fontsize=8)
    
    plt.title(f't-SNE: FastText (trained on tokens)')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved visualization to {save_path}")
    
    plt.show()


def visualize_fasttext_pca(model: FastText, 
                           n_words: int = 200,
                           save_path: str = None):
    """
    Visualize FastText vectors using PCA.
    
    Args:
        model: Trained FastText model
        n_words: Number of words to visualize
        save_path: Optional path to save figure
    """
    logger.info(f"Creating PCA visualization of FastText vectors ({n_words} words)")
    
    vocab = list(model.wv.key_to_index.keys())[:n_words]
    vectors = np.array([model.wv[w] for w in vocab])
    
    pca = PCA(n_components=2)
    pca_emb = pca.fit_transform(vectors)
    
    plt.figure(figsize=(12, 10))
    plt.scatter(pca_emb[:, 0], pca_emb[:, 1], color='lime', alpha=0.7)
    
    for i, w in enumerate(vocab):
        plt.annotate(w, (pca_emb[i, 0], pca_emb[i, 1]), fontsize=8)
    
    plt.title(f'PCA: FastText (trained on tokens)')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved visualization to {save_path}")
    
    plt.show()
