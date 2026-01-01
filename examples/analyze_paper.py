"""
Example: Analyze a single academic paper.

This script demonstrates how to:
1. Load a PDF or image of a paper
2. Run OCR to extract text
3. Classify sections
4. Check grammar
5. Verify facts
6. Generate quality scores
"""

import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from nlp_paper_analyzer.config import Config
from nlp_paper_analyzer.utils import logger

def analyze_paper(paper_path: str):
    """
    Analyze a single paper through the complete pipeline.
    
    Args:
        paper_path: Path to PDF or image file
    """
    logger.info("="*70)
    logger.info(f"Analyzing Paper: {paper_path}")
    logger.info("="*70)
    
    # Check if file exists
    if not Path(paper_path).exists():
        logger.error(f"File not found: {paper_path}")
        logger.info("\nPlease provide a valid path to a PDF or image file.")
        logger.info("Example: python analyze_paper.py path/to/paper.pdf")
        return
    
    # Initialize pipeline (requires trained models)
    logger.info("\n[Step 1] Initializing pipeline...")
    logger.info("Note: This requires pre-trained models.")
    logger.info("See examples/train_models.py for training workflow.")
    
    # TODO: Load models and initialize pipeline
    # from nlp_paper_analyzer.pipeline import UnifiedDocumentPipeline
    # pipeline = UnifiedDocumentPipeline(...)
    
    # Run pipeline
    logger.info("\n[Step 2] Running OCR...")
    logger.info("Converting document to text using Nougat...")
    
    logger.info("\n[Step 3] Classifying sections...")
    logger.info("Identifying Introduction, Methodology, etc...")
    
    logger.info("\n[Step 4] Checking grammar...")
    logger.info("Analyzing writing quality...")
    
    logger.info("\n[Step 5] Verifying facts...")
    logger.info("Checking internal consistency...")
    
    logger.info("\n[Step 6] Calculating quality scores...")
    logger.info("Generating final assessment...")
    
    # Display results
    logger.info("\n" + "="*70)
    logger.info("Analysis Complete!")
    logger.info("="*70)
    logger.info("\nResults would include:")
    logger.info("- Extracted sections with classifications")
    logger.info("- Grammar corrections")
    logger.info("- Fact-check verdicts")
    logger.info("- Overall quality score (0-10)")
    logger.info("- Grade (A+ to F)")
    logger.info("- Improvement recommendations")

def main():
    """Main entry point."""
    
    # Example usage
    example_paper = "path/to/your/paper.pdf"
    
    logger.info("NLP Paper Analyzer - Single Paper Analysis")
    logger.info("\nThis is a template script demonstrating the analysis workflow.")
    logger.info("\nTo use:")
    logger.info("1. Train models using examples/train_models.py")
    logger.info("2. Update this script to load your trained models")
    logger.info("3. Run: python analyze_paper.py path/to/paper.pdf")
    logger.info("\nFor now, running with example path...")
    
    analyze_paper(example_paper)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        paper_path = sys.argv[1]
        analyze_paper(paper_path)
    else:
        main()
