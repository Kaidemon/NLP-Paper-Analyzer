"""Helper utilities for NLP Paper Analyzer."""

import os
import json
from pathlib import Path
from typing import Any, Dict, List
from tqdm import tqdm


def ensure_dir(path: Path) -> Path:
    """Create directory if it doesn't exist."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: Any, filepath: Path):
    """Save data to JSON file."""
    filepath = Path(filepath)
    ensure_dir(filepath.parent)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(filepath: Path) -> Any:
    """Load data from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def progress_bar(iterable, desc: str = None, **kwargs):
    """Create a progress bar."""
    return tqdm(iterable, desc=desc, **kwargs)


def count_files(directory: Path, pattern: str = "*") -> int:
    """Count files matching pattern in directory."""
    directory = Path(directory)
    if not directory.exists():
        return 0
    return len(list(directory.glob(pattern)))


def get_file_size_mb(filepath: Path) -> float:
    """Get file size in megabytes."""
    return Path(filepath).stat().st_size / (1024 * 1024)
