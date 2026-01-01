"""Embeddings module for Phase 2."""

from nlp_paper_analyzer.embeddings.tfidf import (
    create_tfidf_matrix,
    visualize_tfidf_tsne
)

from nlp_paper_analyzer.embeddings.word2vec import (
    train_word2vec,
    save_word2vec,
    load_word2vec
)

from nlp_paper_analyzer.embeddings.bert import (
    load_bert_model,
    encode_texts
)

from nlp_paper_analyzer.embeddings.glove import (
    load_glove_model,
    get_word_vectors,
    visualize_glove_tsne,
    visualize_glove_pca
)

from nlp_paper_analyzer.embeddings.fasttext import (
    train_fasttext,
    save_fasttext,
    load_fasttext,
    visualize_fasttext_tsne,
    visualize_fasttext_pca
)

__all__ = [
    # TF-IDF
    'create_tfidf_matrix',
    'visualize_tfidf_tsne',
    # Word2Vec
    'train_word2vec',
    'save_word2vec',
    'load_word2vec',
    # BERT
    'load_bert_model',
    'encode_texts',
    # GloVe
    'load_glove_model',
    'get_word_vectors',
    'visualize_glove_tsne',
    'visualize_glove_pca',
    # FastText
    'train_fasttext',
    'save_fasttext',
    'load_fasttext',
    'visualize_fasttext_tsne',
    'visualize_fasttext_pca'
]
