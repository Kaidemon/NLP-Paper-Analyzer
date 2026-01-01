"""
Nougat OCR parser for Phase 4.
Converts PDF/images to Markdown using Meta's Nougat model.
"""

import os
import re
import torch
from PIL import Image
from pdf2image import convert_from_path
from transformers import NougatProcessor, VisionEncoderDecoderModel
from pathlib import Path
from typing import List, Dict

from nlp_paper_analyzer.config import Config
from nlp_paper_analyzer.utils import logger


class NougatParser:
    """Nougat OCR parser for academic documents."""
    
    def __init__(self, model_name: str = None):
        """
        Initialize Nougat parser.
        
        Args:
            model_name: Nougat model identifier
        """
        if model_name is None:
            model_name = Config.NOUGAT_MODEL
        
        self.model_name = model_name
        self.device = Config.DEVICE
        self.processor = None
        self.model = None
        
        logger.info(f"Initializing Nougat parser: {model_name}")
    
    def load_model(self):
        """Load Nougat model and processor."""
        if self.model is not None:
            return
        
        logger.info(f"Loading Nougat model: {self.model_name}")
        self.processor = NougatProcessor.from_pretrained(self.model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        logger.info(f"✓ Nougat model loaded on {self.device}")
    
    def convert_to_latex(self, file_path: str, output_file: str = "output.mmd") -> str:
        """
        Convert PDF or image to Markdown/LaTeX.
        
        Args:
            file_path: Path to PDF or image file
            output_file: Output file path
            
        Returns:
            Path to output file
        """
        self.load_model()
        
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        logger.info(f"Converting {file_path} to Markdown")
        
        # Load images
        images = []
        if file_path.suffix.lower() == ".pdf":
            logger.info("Rendering PDF pages to images...")
            images = convert_from_path(str(file_path))
        else:
            logger.info("Loading image file...")
            images = [Image.open(file_path).convert("RGB")]
        
        # Process each page
        full_text = ""
        logger.info(f"Processing {len(images)} page(s)...")
        
        for i, img in enumerate(images):
            pixel_values = self.processor(img, return_tensors="pt").pixel_values.to(self.device)
            
            outputs = self.model.generate(
                pixel_values,
                min_length=1,
                max_new_tokens=Config.OCR_MAX_TOKENS,
                bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
            )
            
            seq = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
            seq = self.processor.post_process_generation(seq, fix_markdown=False)
            
            full_text += f"\n% Page {i+1}\n{seq}\n"
            logger.info(f"  Page {i+1}/{len(images)} complete")
        
        # Save output
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(full_text)
        
        logger.info(f"✓ Conversion complete. Saved to: {output_path}")
        
        return str(output_path)
    
    def parse_mmd_to_sections(self, mmd_content: str) -> List[Dict]:
        """
        Parse Nougat Markdown output into sections.
        
        Args:
            mmd_content: Markdown content from Nougat
            
        Returns:
            List of section dictionaries
        """
        lines = mmd_content.split('\n')
        sections = []
        current_heading = "Abstract"
        current_text = []
        
        header_pattern = re.compile(r'^(#+)\s+(.*)')
        
        for line in lines:
            match = header_pattern.match(line)
            if match:
                # Save previous section
                if current_text:
                    sections.append({
                        'heading': current_heading,
                        'text': ' '.join(current_text).strip()
                    })
                
                # Start new section
                current_heading = match.group(2).strip()
                current_text = []
            else:
                # Add to current section (skip page markers)
                if not line.strip().startswith('% Page') and line.strip():
                    current_text.append(line.strip())
        
        # Save last section
        if current_text:
            sections.append({
                'heading': current_heading,
                'text': ' '.join(current_text).strip()
            })
        
        logger.info(f"✓ Parsed {len(sections)} sections from Markdown")
        
        return sections


def convert_to_latex(file_path: str, output_file: str = "output.mmd") -> str:
    """
    Convenience function to convert file to LaTeX/Markdown.
    
    Args:
        file_path: Path to PDF or image
        output_file: Output file path
        
    Returns:
        Path to output file
    """
    parser = NougatParser()
    return parser.convert_to_latex(file_path, output_file)


def parse_mmd_to_sections(mmd_content: str) -> List[Dict]:
    """
    Convenience function to parse Markdown to sections.
    
    Args:
        mmd_content: Markdown content
        
    Returns:
        List of sections
    """
    parser = NougatParser()
    return parser.parse_mmd_to_sections(mmd_content)
