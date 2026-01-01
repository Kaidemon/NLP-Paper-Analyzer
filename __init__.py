"""
NLP Paper Analyzer
==================

A comprehensive toolkit for analyzing academic papers using NLP techniques.

Phases:
    1. Data Processing: Load and preprocess PeerRead dataset
    2. Embeddings: TF-IDF, Word2Vec, BERT, GloVe, FastText
    3. Section Classification: Deep learning models for section detection
    4. OCR: Document parsing with Nougat
    5. Grammar Correction: T5-based grammar fixing
    6. Fact Checking: FEVER-based claim verification
    7. Unified Pipeline: End-to-end document processing
    8. Quality Scoring: Automated paper quality assessment
"""

__version__ = "1.0.0"
__author__ = "NLP Project Team"

from nlp_paper_analyzer.config import Config

__all__ = ['Config']
