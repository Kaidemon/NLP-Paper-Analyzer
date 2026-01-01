"""Scoring module for Phase 8."""

from nlp_paper_analyzer.scoring.scorer import PaperQualityScorer
from nlp_paper_analyzer.scoring.visualizations import (
    visualize_scores,
    display_detailed_scores
)

__all__ = [
    'PaperQualityScorer',
    'visualize_scores',
    'display_detailed_scores'
]
