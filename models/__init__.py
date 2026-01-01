"""Deep learning models module for Phase 3."""

from nlp_paper_analyzer.models.architectures import (
    build_bilstm,
    build_cnn_bilstm,
    build_lstm,
    build_gru,
    get_architecture,
    ARCHITECTURES
)

from nlp_paper_analyzer.models.section_classifier import (
    balance_dataset,
    prepare_data,
    train_section_classifier
)


from nlp_paper_analyzer.models.training import (
    plot_training_history,
    evaluate_model,
    compare_models,
    plot_model_comparison
)

from nlp_paper_analyzer.models.grammar import GrammarCorrector
from nlp_paper_analyzer.models.fact_checker import FactVerifier

__all__ = [
    # Architectures
    'build_bilstm',
    'build_cnn_bilstm',
    'build_lstm',
    'build_gru',
    'get_architecture',
    'ARCHITECTURES',
    # Section Classifier
    'balance_dataset',
    'prepare_data',
    'train_section_classifier',
    # Training Utilities
    'plot_training_history',
    'evaluate_model',
    'compare_models',
    'plot_model_comparison',
    # Grammar & Fact Check
    'GrammarCorrector',
    'FactVerifier'
]
