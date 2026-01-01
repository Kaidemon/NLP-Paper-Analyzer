"""
Grammar correction using T5 for Phase 5.
"""

import torch
import random
from datasets import load_dataset
from transformers import (
    T5TokenizerFast,
    T5ForConditionalGeneration,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback
)
import evaluate
import numpy as np
from pathlib import Path
from typing import List

from nlp_paper_analyzer.config import Config
from nlp_paper_analyzer.utils import logger


class GrammarCorrector:
    """T5-based grammar correction model."""
    
    def __init__(self, model_name: str = None, model_path: Path = None):
        """
        Initialize grammar corrector.
        
        Args:
            model_name: Base model name (for training)
            model_path: Path to trained model (for inference)
        """
        self.device = Config.DEVICE
        
        if model_path and Path(model_path).exists():
            logger.info(f"Loading trained model from {model_path}")
            self.tokenizer = T5TokenizerFast.from_pretrained(str(model_path))
            self.model = T5ForConditionalGeneration.from_pretrained(str(model_path))
        else:
            if model_name is None:
                model_name = Config.GRAMMAR_MODEL
            logger.info(f"Initializing new model: {model_name}")
            self.tokenizer = T5TokenizerFast.from_pretrained(model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        
        self.model.to(self.device)
        logger.info(f"✓ Grammar corrector ready on {self.device}")
    
    def train(self, epochs: int = 10, batch_size: int = 4, output_dir: str = None):
        """
        Train grammar correction model on JFLEG dataset.
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size
            output_dir: Output directory for checkpoints
        """
        if output_dir is None:
            output_dir = str(Config.GRAMMAR_MODEL_PATH.parent)
        
        logger.info("Loading JFLEG dataset...")
        raw_train = load_dataset("jfleg", split="validation")
        raw_eval = load_dataset("jfleg", split="test")
        
        # Preprocessing function
        def preprocess_multi_ref(example):
            input_text = "grammar: " + example["sentence"]
            target_text = random.choice(example["corrections"])
            
            model_inputs = self.tokenizer(
                input_text,
                max_length=Config.GRAMMAR_MAX_INPUT_LEN,
                truncation=True
            )
            
            labels = self.tokenizer(
                target_text,
                max_length=Config.GRAMMAR_MAX_TARGET_LEN,
                truncation=True
            )
            
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs
        
        # Prepare datasets
        train_dataset = raw_train.map(preprocess_multi_ref, remove_columns=raw_train.column_names)
        eval_dataset = raw_eval.map(preprocess_multi_ref, remove_columns=raw_eval.column_names)
        
        data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer, model=self.model)
        
        # Metrics
        gleu = evaluate.load("google_bleu")
        
        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            
            if isinstance(preds, tuple) or preds.ndim == 3:
                preds = np.argmax(preds, axis=-1)
            
            labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
            
            decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
            decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
            
            references = [[lbl.strip()] for lbl in decoded_labels]
            predictions = [pred.strip() for pred in decoded_preds]
            
            return gleu.compute(predictions=predictions, references=references)
        
        # Training arguments
        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=Config.GRAMMAR_LEARNING_RATE,
            lr_scheduler_type="linear",
            warmup_ratio=0.1,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            weight_decay=0.01,
            predict_with_generate=True,
            generation_max_length=128,
            load_best_model_at_end=True,
            metric_for_best_model="eval_google_bleu",
            greater_is_better=True,
            save_total_limit=2,
            fp16=torch.cuda.is_available(),
            logging_steps=100,
            report_to="none"
        )
        
        # Trainer
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]
        )
        
        # Train
        logger.info(f"Starting training for {epochs} epochs...")
        trainer.train()
        
        # Evaluate
        metrics = trainer.evaluate()
        logger.info(f"✓ Training complete. Final GLEU Score: {metrics['eval_google_bleu']:.4f}")
        
        return trainer
    
    def correct_sentence(self, text: str, num_beams: int = 5, max_length: int = 128) -> str:
        """
        Correct grammar in a sentence.
        
        Args:
            text: Input text
            num_beams: Number of beams for generation
            max_length: Maximum output length
            
        Returns:
            Corrected text
        """
        input_text = "grammar: " + text
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.device)
        
        outputs = self.model.generate(
            **inputs,
            num_beams=num_beams,
            max_length=max_length,
            early_stopping=True
        )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def correct_sections(self, sections: List[Dict], max_chars: int = None) -> List[Dict]:
        """
        Correct grammar in multiple sections.
        
        Args:
            sections: List of section dictionaries
            max_chars: Maximum characters per section
            
        Returns:
            List of corrected sections
        """
        if max_chars is None:
            max_chars = Config.PIPELINE_MAX_GRAMMAR_CHARS
        
        corrected_sections = []
        
        for i, section in enumerate(sections):
            heading = section.get('heading', 'Unknown')
            raw_text = section.get('text', '')[:max_chars]
            
            if raw_text:
                corrected_text = self.correct_sentence(raw_text)
                corrected_sections.append({
                    'heading': heading,
                    'raw_text': raw_text,
                    'corrected_text': corrected_text
                })
                
                logger.info(f"  ✓ Corrected section {i+1}: {heading}")
        
        logger.info(f"✓ Corrected {len(corrected_sections)} sections")
        
        return corrected_sections
    
    def save_model(self, save_path: Path):
        """Save model and tokenizer."""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        self.model.save_pretrained(str(save_path))
        self.tokenizer.save_pretrained(str(save_path))
        
        logger.info(f"✓ Model saved to {save_path}")
