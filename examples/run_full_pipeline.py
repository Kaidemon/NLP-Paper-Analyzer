"""
Example: Complete pipeline execution for analyzing academic papers.

This script demonstrates the full end-to-end workflow:
1. Load dataset
2. Preprocess data
3. Train or load models
4. Run unified pipeline
5. Generate quality scores
"""

import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from nlp_paper_analyzer.config import Config
from nlp_paper_analyzer.data import load_dataset_from_kaggle, preprocess_column
from nlp_paper_analyzer.utils import logger

def main():
    """Run the complete pipeline."""
    
    logger.info("="*70)
    logger.info("NLP Paper Analyzer - Full Pipeline Example")
    logger.info("="*70)
    
    # Ensure directories exist
    Config.ensure_dirs()
    
    # Step 1: Load dataset
    logger.info("\n[Step 1] Loading PeerRead dataset from Kaggle...")
    try:
        df = load_dataset_from_kaggle(save_csv=True)
        logger.info(f"✓ Loaded {len(df)} papers")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        logger.info("Please ensure you have Kaggle credentials configured")
        return
    
    # Step 2: Preprocess
    logger.info("\n[Step 2] Preprocessing text...")
    df['reviews_tokens'] = df['reviews'].apply(lambda x: preprocess_column(str(x)))
    df['sections_tokens'] = df['sections'].apply(lambda x: preprocess_column(str(x)))
    df['reviews_text'] = df['reviews_tokens'].apply(lambda x: " ".join(x))
    logger.info(f"✓ Preprocessed {len(df)} papers")
    
    # Step 3: Train models (or load if available)
    logger.info("\n[Step 3] Model training/loading...")
    logger.info("Note: Full model training requires significant time and resources.")
    logger.info("See examples/train_models.py for the complete training workflow.")
    
    # Step 4: Pipeline execution
    logger.info("\n[Step 4] Pipeline execution...")
    logger.info("To run the unified pipeline on a specific document:")
    logger.info("  from nlp_paper_analyzer.pipeline import UnifiedDocumentPipeline")
    logger.info("  pipeline = UnifiedDocumentPipeline(...)")
    logger.info("  results = pipeline.run_full_pipeline('path/to/paper.pdf')")
    
    # Step 5: Quality scoring
    logger.info("\n[Step 5] Quality scoring...")
    logger.info("After pipeline execution, scores are automatically calculated")
    logger.info("See the README for scoring criteria details")
    
    logger.info("\n" + "="*70)
    logger.info("Pipeline setup complete!")
    logger.info("="*70)
    logger.info("\nNext steps:")
    logger.info("1. Train models using examples/train_models.py")
    logger.info("2. Analyze a paper using examples/analyze_paper.py")
    logger.info("3. Check the README for detailed API documentation")

if __name__ == "__main__":
    main()
