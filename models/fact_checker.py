"""
Fact verification using FEVER dataset for Phase 6.
"""

import torch
from datasets import load_dataset, Dataset
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Trainer,
    TrainingArguments
)
from collections import defaultdict
from pathlib import Path
from typing import List, Dict

from nlp_paper_analyzer.config import Config
from nlp_paper_analyzer.utils import logger


class FactVerifier:
    """T5-based fact verification model trained on FEVER."""
    
    def __init__(self, model_name: str = None, model_path: Path = None):
        """
        Initialize fact verifier.
        
        Args:
            model_name: Base model name (for training)
            model_path: Path to trained model (for inference)
        """
        self.device = Config.DEVICE
        
        if model_path and Path(model_path).exists():
            logger.info(f"Loading trained model from {model_path}")
            self.tokenizer = T5Tokenizer.from_pretrained(str(model_path), legacy=False)
            self.model = T5ForConditionalGeneration.from_pretrained(str(model_path))
        else:
            if model_name is None:
                model_name = Config.FACT_MODEL
            logger.info(f"Initializing new model: {model_name}")
            self.tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
            self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        
        self.model.to(self.device)
        logger.info(f"✓ Fact verifier ready on {self.device}")
    
    def train(self, epochs: int = 1, batch_size: int = 8, output_dir: str = None):
        """
        Train fact verification model on FEVER dataset.
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size
            output_dir: Output directory for checkpoints
        """
        if output_dir is None:
            output_dir = str(Config.FACT_MODEL_PATH.parent)
        
        logger.info("Loading FEVER dataset...")
        ds = load_dataset("fever", revision="refs/convert/parquet")
        
        # Attach evidence text
        def attach_evidence_text(example):
            example["evidence_text"] = f"Content regarding {example['evidence_wiki_url']}."
            return example
        
        ds_with_text = ds.map(attach_evidence_text)
        
        # Aggregate evidence
        def aggregate_evidence(dataset):
            grouped = defaultdict(list)
            for ex in dataset:
                grouped[ex["id"]].append(ex)
            
            aggregated = []
            for claim_id, examples in grouped.items():
                label = examples[0]["label"]
                claim = examples[0]["claim"]
                
                evidence_texts = [ex["evidence_text"] for ex in examples if ex.get("evidence_text")]
                premise = " ".join(evidence_texts) if evidence_texts else "No evidence available."
                
                aggregated.append({
                    "premise": premise,
                    "hypothesis": claim,
                    "label": label
                })
            return aggregated
        
        train_data = aggregate_evidence(ds_with_text["train"])
        val_data = aggregate_evidence(ds_with_text["validation"])
        
        logger.info(f"Aggregated into {len(train_data)} training examples")
        
        # Format for T5
        def to_t5_format(example):
            return {
                "input_text": f"verify: {example['premise']} </s> claim: {example['hypothesis']}",
                "target_text": example["label"]
            }
        
        train_ds = Dataset.from_list(train_data).map(to_t5_format)
        val_ds = Dataset.from_list(val_data).map(to_t5_format)
        
        # Tokenize
        def tokenize_and_prepare_labels(example):
            model_inputs = self.tokenizer(
                example["input_text"],
                max_length=Config.FACT_MAX_LEN,
                truncation=True,
                padding="max_length",
            )
            
            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(
                    example["target_text"],
                    max_length=10,
                    truncation=True,
                    padding="max_length",
                )
            
            model_inputs["labels"] = [
                (l if l != self.tokenizer.pad_token_id else -100) for l in labels["input_ids"]
            ]
            return model_inputs
        
        train_dataset = train_ds.map(tokenize_and_prepare_labels, remove_columns=train_ds.column_names)
        val_dataset = val_ds.map(tokenize_and_prepare_labels, remove_columns=val_ds.column_names)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            do_eval=True,
            eval_steps=500,
            logging_steps=100,
            save_steps=1000,
            save_total_limit=2,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            learning_rate=Config.FACT_LEARNING_RATE,
            remove_unused_columns=False,
            report_to="none"
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer
        )
        
        # Train
        logger.info(f"Starting training for {epochs} epochs...")
        trainer.train()
        
        logger.info("✓ Training complete")
        
        return trainer
    
    def verify_claim(self, claim: str, evidence: str) -> str:
        """
        Verify a claim against evidence.
        
        Args:
            claim: Claim to verify
            evidence: Evidence text
            
        Returns:
            Verdict ('supports', 'refutes', or 'not enough info')
        """
        input_text = f"verify: {evidence} </s> claim: {claim}"
        inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, max_length=Config.FACT_MAX_LEN).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=10)
        
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return result
    
    def verify_sections(self, sections: List[Dict], claims: List[Dict] = None) -> List[Dict]:
        """
        Verify claims against sections.
        
        Args:
            sections: List of section dictionaries
            claims: List of claim dictionaries with 'claim' and 'evidence_idx'
            
        Returns:
            List of verification results
        """
        if claims is None:
            # Auto-generate claims
            claims = []
            for i, section in enumerate(sections[:3]):
                claims.append({
                    'claim': f"This section discusses {section.get('predicted_label', section.get('heading', 'content')).lower()}.",
                    'evidence_idx': i
                })
        
        results = []
        
        for claim_info in claims:
            claim = claim_info['claim']
            evidence_idx = claim_info.get('evidence_idx', 0)
            
            if evidence_idx < len(sections):
                evidence = sections[evidence_idx].get('corrected_text') or sections[evidence_idx].get('text', '')
                verdict = self.verify_claim(claim, evidence)
                
                results.append({
                    'claim': claim,
                    'evidence_section': sections[evidence_idx].get('heading', 'Unknown'),
                    'verdict': verdict
                })
                
                logger.info(f"  ✓ Verified: \"{claim[:50]}...\" → {verdict}")
        
        logger.info(f"✓ Verified {len(results)} claims")
        
        return results
    
    def save_model(self, save_path: Path):
        """Save model and tokenizer."""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        self.model.save_pretrained(str(save_path))
        self.tokenizer.save_pretrained(str(save_path))
        
        logger.info(f"✓ Model saved to {save_path}")
