"""
Data ingestion module for loading PeerRead dataset from Kaggle.
"""

import os
import glob
import json
import pandas as pd
import kagglehub
from pathlib import Path
from typing import List, Dict, Optional

from nlp_paper_analyzer.config import Config
from nlp_paper_analyzer.utils import logger, progress_bar


def download_dataset(dataset_name: str = None) -> Path:
    """
    Download the ASAP review dataset from Kaggle.
    
    Args:
        dataset_name: Kaggle dataset identifier (default from config)
        
    Returns:
        Path to downloaded dataset
    """
    if dataset_name is None:
        dataset_name = Config.KAGGLE_DATASET
    
    logger.info(f"Downloading dataset: {dataset_name}")
    try:
        path = kagglehub.dataset_download(dataset_name)
        logger.info(f"Dataset downloaded to: {path}")
        return Path(path)
    except Exception as e:
        logger.error(f"Error downloading dataset: {e}")
        raise


def load_papers(base_path: Path) -> List[Dict]:
    """
    Load all papers from the dataset directory structure.
    
    Args:
        base_path: Root directory of the dataset
        
    Returns:
        List of paper dictionaries with journal, title, reviews, and sections
    """
    all_papers_data = []
    journal_dirs = glob.glob(str(base_path / "*"))
    
    logger.info(f"Found {len(journal_dirs)} journal directories")
    
    for journal_path in progress_bar(journal_dirs, desc="Processing journals"):
        if not os.path.isdir(journal_path):
            continue
        
        journal_name = os.path.basename(journal_path)
        content_dir = os.path.join(journal_path, f"{journal_name}_content")
        review_dir = os.path.join(journal_path, f"{journal_name}_review")
        
        if not os.path.exists(content_dir):
            continue
            
        content_files = glob.glob(os.path.join(content_dir, "*.json"))
        
        for content_file_path in content_files:
            paper_data = {}
            try:
                base_filename = os.path.basename(content_file_path)
                paper_id = base_filename.replace("_content.json", "")
                review_file_path = os.path.join(review_dir, f"{paper_id}_review.json")
                
                paper_data['journal'] = journal_name
                
                # Load content
                with open(content_file_path, 'r', encoding='utf-8') as f:
                    content_data = json.load(f)
                    paper_data['title'] = content_data.get('metadata', {}).get('title', None)
                    paper_data['sections'] = content_data.get('metadata', {}).get('sections', None)
                
                # Load reviews
                paper_data['reviews'] = None
                if os.path.exists(review_file_path):
                    with open(review_file_path, 'r', encoding='utf-8') as f:
                        review_data = json.load(f)
                        # Handle key inconsistencies
                        paper_data['reviews'] = review_data.get('reviews') or review_data.get('reveiws')
                
                all_papers_data.append(paper_data)
                
            except Exception as e:
                logger.warning(f"Error processing {base_filename}: {e}")
                continue
    
    logger.info(f"Loaded {len(all_papers_data)} papers")
    return all_papers_data


def compile_dataset(papers_data: List[Dict], output_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Compile papers into a DataFrame and optionally save to CSV.
    
    Args:
        papers_data: List of paper dictionaries
        output_path: Optional path to save CSV
        
    Returns:
        DataFrame with compiled papers
    """
    df = pd.DataFrame(papers_data)
    df = df[['journal', 'title', 'reviews', 'sections']].dropna(subset=['title'])
    
    logger.info(f"Compiled dataset: {len(df)} rows")
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved to: {output_path}")
    
    return df


def load_dataset_from_kaggle(save_csv: bool = True) -> pd.DataFrame:
    """
    Complete pipeline to download and load dataset from Kaggle.
    
    Args:
        save_csv: Whether to save compiled dataset to CSV
        
    Returns:
        DataFrame with all papers
    """
    # Download
    base_path = download_dataset()
    
    # Load papers
    papers = load_papers(base_path)
    
    # Compile
    output_path = Config.DATA_DIR / "asap_dataset_compiled.csv" if save_csv else None
    df = compile_dataset(papers, output_path)
    
    return df
