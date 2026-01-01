"""OCR module for Phase 4."""

from nlp_paper_analyzer.ocr.nougat_parser import (
    NougatParser,
    convert_to_latex,
    parse_mmd_to_sections
)

__all__ = [
    'NougatParser',
    'convert_to_latex',
    'parse_mmd_to_sections'
]
