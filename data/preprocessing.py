"""
Text preprocessing utilities for NLP Paper Analyzer.
"""

import re
import ast
import nltk
from typing import List, Union
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from nlp_paper_analyzer.utils import logger

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

STOP_WORDS = set(stopwords.words('english'))


def simple_preprocess_nltk(text: str, min_len: int = 2, max_len: int = 15) -> List[str]:
    """
    Preprocess text using NLTK: tokenize, lowercase, remove stopwords.
    
    Args:
        text: Input text
        min_len: Minimum token length
        max_len: Maximum token length
        
    Returns:
        List of preprocessed tokens
    """
    # Remove special characters
    text = re.sub(r'[^a-zA-Z0-9\s\-\']', '', text)
    
    # Tokenize and lowercase
    tokens = word_tokenize(text.lower())
    
    # Filter by length and stopwords
    tokens = [
        t for t in tokens 
        if min_len <= len(t) <= max_len and t not in STOP_WORDS
    ]
    
    return tokens


def preprocess_column(text_list_str: Union[str, List]) -> List[str]:
    """
    Parse stringified lists and tokenize content.
    
    Args:
        text_list_str: String representation of list or actual list
        
    Returns:
        List of preprocessed tokens
    """
    # Parse if string
    if isinstance(text_list_str, str):
        try:
            text_list = ast.literal_eval(text_list_str)
        except (ValueError, SyntaxError):
            return []
    else:
        text_list = text_list_str
    
    processed_tokens = []
    
    if isinstance(text_list, list):
        for item in text_list:
            if isinstance(item, dict):
                # Handle different key names in JSON structure
                text_content = item.get('review') or item.get('text')
                if text_content:
                    processed_tokens.extend(simple_preprocess_nltk(text_content))
            elif isinstance(item, str):
                processed_tokens.extend(simple_preprocess_nltk(item))
    
    return processed_tokens


def normalize_section_label(header: str) -> str:
    """
    Normalize diverse section headers into standard categories.
    
    Args:
        header: Section header text
        
    Returns:
        Standardized section label
    """
    if not isinstance(header, str):
        return "Unknown"
    
    # Clean header
    clean = re.sub(r'^[\d\.\s]+', '', header).lower().strip()
    
    # Mapping rules
    mapping = {
        'Appendix': ['acknowledgement', 'appendix', 'reference'],
        'Introduction': ['introduction', 'motivation', 'preface'],
        'Conclusion': ['conclusion', 'summary', 'future work', 'discussion'],
        'Related Work': ['related work', 'background', 'literature', 'prior work'],
        'Experiments': ['experiment', 'result', 'evaluation', 'performance', 'dataset'],
        'Methodology': []  # Default/Fallthrough
    }
    
    # Match against patterns
    for label, keywords in mapping.items():
        if keywords:
            pattern = r'\b(' + '|'.join(keywords) + r')\b'
            if re.search(pattern, clean):
                return label
    
    return 'Methodology'


def extract_and_flatten_sections(df, section_column: str = 'sections') -> List[dict]:
    """
    Extract and flatten sections from DataFrame.
    
    Args:
        df: DataFrame with sections column
        section_column: Name of the sections column
        
    Returns:
        List of flattened section dictionaries
    """
    flat_data = []
    
    for idx, row in df.iterrows():
        try:
            sections = ast.literal_eval(row[section_column]) if isinstance(row[section_column], str) else row[section_column]
            
            if isinstance(sections, list):
                for s in sections:
                    if isinstance(s, dict) and s.get('text'):
                        flat_data.append({
                            'text': s['text'],
                            'raw_label': s.get('heading', ''),
                            'label': normalize_section_label(s.get('heading', ''))
                        })
        except Exception as e:
            logger.warning(f"Error processing row {idx}: {e}")
            continue
    
    return flat_data
