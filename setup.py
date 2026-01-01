"""Setup script for NLP Paper Analyzer package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="nlp-paper-analyzer",
    version="1.0.0",
    author="NLP Project Team",
    description="A comprehensive toolkit for analyzing academic papers using NLP",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "nltk>=3.6.0",
        "gensim>=4.0.0",
        "sentence-transformers>=2.0.0",
        "torch>=2.0.0",
        "tensorflow>=2.10.0",
        "transformers>=4.30.0",
        "datasets>=2.0.0",
        "evaluate>=0.4.0",
        "sentencepiece>=0.1.99",
        "sacrebleu>=2.0.0",
        "accelerate>=0.20.0",
        "pdf2image>=1.16.0",
        "pillow>=9.0.0",
        "python-Levenshtein>=0.20.0",
        "kagglehub>=0.1.0",
        "scikit-learn>=1.0.0",
        "tqdm>=4.62.0",
        "seaborn>=0.11.0",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
