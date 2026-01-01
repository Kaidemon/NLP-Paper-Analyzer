"""
Unified end-to-end pipeline for Phase 7.
Integrates OCR, classification, grammar correction, and fact checking.
"""

import os
import torch
from pathlib import Path
from typing import List, Dict, Optional


from nlp_paper_analyzer.config import Config
from nlp_paper_analyzer.data.ocr import convert_to_latex, parse_mmd_to_sections
from nlp_paper_analyzer.utils import logger



class UnifiedDocumentPipeline:
    """End-to-end pipeline for processing academic documents."""
    
    def __init__(self,
                 section_model=None,
                 section_tokenizer=None,
                 section_encoder=None,
                 grammar_model=None,
                 grammar_tokenizer=None,
                 fact_model=None,
                 fact_tokenizer=None):
        """
        Initialize pipeline with pre-trained models.
        
        Args:
            section_model: Section classification model
            section_tokenizer: Tokenizer for section classification
            section_encoder: Label encoder for sections
            grammar_model: Grammar correction model
            grammar_tokenizer: Tokenizer for grammar correction
            fact_model: Fact checking model
            fact_tokenizer: Tokenizer for fact checking
        """
        self.section_model = section_model
        self.section_tokenizer = section_tokenizer
        self.section_encoder = section_encoder
        
        self.grammar_model = grammar_model
        self.grammar_tokenizer = grammar_tokenizer
        
        self.fact_model = fact_model
        self.fact_tokenizer = fact_tokenizer
        
        self.max_len = Config.MAX_LEN
        self.max_grammar_chars = Config.PIPELINE_MAX_GRAMMAR_CHARS
        self.device = Config.DEVICE
        
        logger.info("✓ Unified pipeline initialized")
    
    def step1_ocr(self, file_path: str, output_file: str = "pipeline_output.mmd") -> List[Dict]:
        """
        Step 1: Convert PDF/Image to text using Nougat OCR.
        
        Args:
            file_path: Path to PDF or image
            output_file: Output file path
            
        Returns:
            List of parsed sections
        """
        logger.info("\n" + "="*70)
        logger.info("STEP 1: OCR Processing (Nougat)")
        logger.info("="*70)
        
        if not os.path.exists(file_path):
            logger.error(f"❌ File not found: {file_path}")
            return None
        
        # Convert to Markdown
        output_path = convert_to_latex(file_path, output_file)
        
        # Read and parse
        with open(output_path, "r", encoding="utf-8") as f:
            mmd_content = f.read()
        
        parsed_sections = parse_mmd_to_sections(mmd_content)
        
        logger.info(f"✓ Extracted {len(parsed_sections)} sections from document")
        return parsed_sections
    
    def step2_classify_sections(self, parsed_sections: List[Dict]) -> List[Dict]:
        """
        Step 2: Classify each section using the trained model.
        
        Args:
            parsed_sections: List of section dictionaries
            
        Returns:
            List of classified sections
        """
        logger.info("\n" + "="*70)
        logger.info("STEP 2: Section Classification")
        logger.info("="*70)
        
        if self.section_model is None or self.section_tokenizer is None:
            logger.warning("⚠ Section classification model not loaded. Skipping...")
            return [{'heading': s['heading'], 'text': s['text'], 'predicted_label': 'Unknown', 'confidence': 0.0}
                    for s in parsed_sections]
        
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        
        results = []
        texts = [s['text'] for s in parsed_sections]
        headings = [s['heading'] for s in parsed_sections]
        
        # Tokenize and predict
        seqs = self.section_tokenizer.texts_to_sequences(texts)
        padded_seqs = pad_sequences(seqs, maxlen=self.max_len, padding='post')
        
        predictions = self.section_model.predict(padded_seqs, verbose=0)
        predicted_indices = predictions.argmax(axis=1)
        predicted_labels = self.section_encoder.inverse_transform(predicted_indices)
        confidences = [max(p) for p in predictions]
        
        for i, (heading, text, label, conf) in enumerate(zip(headings, texts, predicted_labels, confidences)):
            results.append({
                'heading': heading,
                'text': text,
                'predicted_label': label,
                'confidence': conf
            })
            logger.info(f"  [{label}] {heading} (confidence: {conf:.2%})")
        
        logger.info(f"✓ Classified {len(results)} sections")
        return results
    
    def step3_correct_grammar(self, classified_sections: List[Dict]) -> List[Dict]:
        """
        Step 3: Apply grammar correction to each section.
        
        Args:
            classified_sections: List of classified sections
            
        Returns:
            List of grammar-corrected sections
        """
        logger.info("\n" + "="*70)
        logger.info("STEP 3: Grammar Correction")
        logger.info("="*70)
        
        if self.grammar_model is None or self.grammar_tokenizer is None:
            logger.warning("⚠ Grammar model not loaded. Skipping correction...")
            return [dict(s, corrected_text=s['text']) for s in classified_sections]
        
        results = []
        for i, section in enumerate(classified_sections):
            raw_text = section['text'][:self.max_grammar_chars]
            
            # Apply grammar correction
            input_text = "grammar: " + raw_text
            inputs = self.grammar_tokenizer(
                input_text,
                return_tensors="pt",
                max_length=256,
                truncation=True
            ).to(self.device)
            
            outputs = self.grammar_model.generate(
                **inputs,
                num_beams=5,
                max_length=256,
                early_stopping=True
            )
            corrected_text = self.grammar_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            result = dict(section)
            result['raw_text'] = raw_text
            result['corrected_text'] = corrected_text
            results.append(result)
            
            logger.info(f"  ✓ Corrected section: {section['heading']}")
        
        logger.info(f"✓ Grammar corrected {len(results)} sections")
        return results
    
    def step4_fact_check(self, corrected_sections: List[Dict], claims: List[Dict] = None) -> List[Dict]:
        """
        Step 4: Verify claims against the processed content.
        
        Args:
            corrected_sections: List of corrected sections
            claims: Optional list of claims to verify
            
        Returns:
            List of fact-check results
        """
        logger.info("\n" + "="*70)
        logger.info("STEP 4: Fact Checking & Consistency Verification")
        logger.info("="*70)
        
        if self.fact_model is None or self.fact_tokenizer is None:
            logger.warning("⚠ Fact-checking model not loaded. Skipping...")
            return []
        
        # Generate default claims if none provided
        if claims is None:
            claims = []
            for i, section in enumerate(corrected_sections[:3]):
                claims.append({
                    'claim': f"This paper has a section about {section['predicted_label'].lower()}.",
                    'evidence_idx': i
                })
        
        results = []
        for claim_info in claims:
            claim = claim_info['claim']
            evidence_idx = claim_info.get('evidence_idx', 0)
            
            if evidence_idx >= len(corrected_sections):
                continue
            
            evidence = corrected_sections[evidence_idx]['corrected_text']
            
            # Verify using fact-checking model
            input_text = f"verify: {evidence} </s> claim: {claim}"
            inputs = self.fact_tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.fact_model.generate(**inputs, max_new_tokens=10)
            
            verdict = self.fact_tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            results.append({
                'claim': claim,
                'evidence_section': corrected_sections[evidence_idx]['heading'],
                'verdict': verdict
            })
            
            status_icon = "✓" if "support" in verdict.lower() else "✗" if "refute" in verdict.lower() else "?"
            logger.info(f"  {status_icon} \"{claim[:50]}...\" → {verdict}")
        
        logger.info(f"✓ Verified {len(results)} claims")
        return results
    
    def run_full_pipeline(self, file_path: str, custom_claims: List[Dict] = None) -> Dict:
        """
        Execute the complete end-to-end pipeline.
        
        Args:
            file_path: Path to PDF or image
            custom_claims: Optional custom claims to verify
            
        Returns:
            Dictionary with complete results
        """
        logger.info("\n" + "█"*70)
        logger.info("        UNIFIED NLP PIPELINE - FULL EXECUTION")
        logger.info("█"*70)
        logger.info(f"Input File: {file_path}")
        logger.info(f"Device: {self.device.upper()}")
        
        # Step 1: OCR
        parsed_sections = self.step1_ocr(file_path)
        if not parsed_sections:
            return None
        
        # Step 2: Section Classification
        classified_sections = self.step2_classify_sections(parsed_sections)
        
        # Step 3: Grammar Correction
        corrected_sections = self.step3_correct_grammar(classified_sections)
        
        # Step 4: Fact Checking
        fact_results = self.step4_fact_check(corrected_sections, custom_claims)
        
        # Compile final results
        final_results = {
            'input_file': file_path,
            'num_sections': len(parsed_sections),
            'sections': corrected_sections,
            'fact_check_results': fact_results,
            'summary': self._generate_summary(corrected_sections, fact_results)
        }
        
        logger.info("\n" + "█"*70)
        logger.info("        PIPELINE COMPLETE")
        logger.info("█"*70)
        
        return final_results
    
    def _generate_summary(self, sections: List[Dict], fact_results: List[Dict]) -> Dict:
        """Generate a summary of pipeline results."""
        section_types = {}
        for s in sections:
            label = s.get('predicted_label', 'Unknown')
            section_types[label] = section_types.get(label, 0) + 1
        
        fact_summary = {'supports': 0, 'refutes': 0, 'not_enough_info': 0}
        for r in fact_results:
            verdict = r['verdict'].lower()
            if 'support' in verdict:
                fact_summary['supports'] += 1
            elif 'refute' in verdict:
                fact_summary['refutes'] += 1
            else:
                fact_summary['not_enough_info'] += 1
        
        return {
            'section_distribution': section_types,
            'fact_check_summary': fact_summary
        }
