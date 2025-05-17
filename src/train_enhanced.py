"""
Enhanced training script for the fast spell correction system.
Implements fine-tuning procedures for masked language models with advanced
features such as learning rate scheduling, gradient clipping, and detailed metrics.
"""

import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    AdamW,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import accuracy_score
import logging
import random
import time
import json
from Levenshtein import distance as levenshtein_distance

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class SpellCorrectionDataset(Dataset):
    """Dataset for training spell correction models"""
    
    def __init__(self, tokenizer, clean_file, noisy_file, max_length=128):
        """
        Initialize the dataset from parallel clean and noisy text files
        
        Args:
            tokenizer: The tokenizer to use
            clean_file: Path to the file with clean text
            noisy_file: Path to the file with noisy text
            max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load data
        with open(clean_file, 'r', encoding='utf-8') as f:
            self.clean_texts = [line.strip() for line in f if line.strip()]
        
        with open(noisy_file, 'r', encoding='utf-8') as f:
            self.noisy_texts = [line.strip() for line in f if line.strip()]
        
        # Ensure same length
        assert len(self.clean_texts) == len(self.noisy_texts), \
            "Clean and noisy text files must have the same number of lines"
        
        logger.info(f"Loaded {len(self.clean_texts)} examples from {clean_file} and {noisy_file}")
    
    def __len__(self):
        return len(self.clean_texts)
    
    def __getitem__(self, idx):
        clean_text = self.clean_texts[idx]
        noisy_text = self.noisy_texts[idx]
        
        # Tokenize both texts
        clean_tokens = self.tokenizer(
            clean_text, 
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        noisy_tokens = self.tokenizer(
            noisy_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Extract values and squeeze batch dimension
        input_ids = noisy_tokens["input_ids"].squeeze()
        attention_mask = noisy_tokens["attention_mask"].squeeze()
        
        # Create labels where tokens differ, use clean tokens as targets
        # Where they're the same, use -100 to ignore in loss calculation
        labels = torch.where(
            input_ids == clean_tokens["input_ids"].squeeze(),
            torch.tensor(-100),
            clean_tokens["input_ids"].squeeze()
        )
        
        # Also replace padding token ids in the labels with -100
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "clean_text": clean_text,
            "noisy_text": noisy_text
        }

def compute_word_correction_rate(predictions, targets):
    """
    Compute Word Correction Rate (WCR) - the percentage of words correctly fixed
    """
    total_words = 0
    correct_words = 0
    
    for pred, target in zip(predictions, targets):
        pred_words = pred.split()
        target_words = target.split()
        
        # Count matching words
        for p_word, t_word in zip(pred_words, target_words):
            if p_word == t_word:
                correct_words += 1
            total_words += 1
    
    return correct_words / total_words if total_words > 0 else 0.0

def compute_character_error_rate(predictions, targets):
    """
    Compute Character Error Rate (CER) using Levenshtein distance
    """
    total_chars = sum(len(t) for t in targets)
    total_distance = sum(levenshtein_distance(p, t) for p, t in zip(predictions, targets))
    
    return total_distance / total_chars if total_chars > 0 else 0.0

def decode_batch(tokenizer, batch, outputs):
    """
    Decode model outputs to text for evaluation
    """
    # Get predictions
    logits = outputs.logits
    preds = torch.argmax(logits, dim=-1)
    
    # Convert to lists of token IDs
    preds = preds.cpu().numpy()
    input_ids = batch["input_ids"].cpu().numpy()
    
    # Decode and return texts
    pred_texts = []
    for i in range(len(preds)):
        # Replace input tokens with predicted tokens where attention mask is 1
        input_id = input_ids[i]
        pred = preds[i]
        attention_mask = batch["attention_mask"][i].cpu().numpy()
        
        # Combine input and predictions
        combined = []
        for j in range(len(input_id)):
            if attention_mask[j] == 1:  # Only use tokens where attention is 1
                combined.append(pred[j])
        
        # Decode to text
        text = tokenizer.decode(combined, skip_special_tokens=True)
        pred_texts.append(text)
    
    return pred_texts

def train(args):
    """
    Train the model
    
    Args:
        args: Command-line arguments
    """
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load tokenizer and model
    logger.info(f"Loading tokenizer and model from {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForMaskedLM.from_pretrained(args.model_name)
    model.to(device)
    
    # Create datasets
    train_dataset = SpellCorrectionDataset(
        tokenizer,
        os.path.join(args.data_dir, "train_clean.txt"),
        os.path.join(args.data_dir, "train_noisy.txt"),
        max_length=args.max_length
    )
    
    val_dataset = SpellCorrectionDataset(
        tokenizer,
        os.path.join(args.data_dir, "val_clean.txt"),
        os.path.join(args.data_dir, "val_noisy.txt"),
        max_length=args.max_length
    )
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )
    
    # Prepare optimizer and schedule
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        eps=args.adam_epsilon
    )
    
    total_steps = len(train_dataloader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Training loop
    logger.info("***** Starting training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num epochs = {args.epochs}")
    logger.info(f"  Batch size = {args.batch_size}")
    logger.info(f"  Total optimization steps = {total_steps}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Track metrics
    best_val_loss = float('inf')
    training_stats = []
    
    for epoch in range(args.epochs):
        logger.info(f"Starting epoch {epoch+1}/{args.epochs}")
        
        # Training
        model.train()
        train_loss = 0.0
        train_steps = 0
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")
        for batch in progress_bar:
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )
            
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            
            # Update weights
            optimizer.step()
            scheduler.step()
            
            # Update stats
            train_loss += loss.item()
            train_steps += 1
            
            # Update progress bar
            progress_bar.set_postfix({"loss": loss.item()})
        
        avg_train_loss = train_loss / train_steps
        logger.info(f"Average training loss: {avg_train_loss:.4f}")
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_steps = 0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validation"):
                # Move batch to device
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                # Forward pass
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"]
                )
                
                loss = outputs.loss
                
                # Update stats
                val_loss += loss.item()
                val_steps += 1
                
                # Decode outputs for metrics
                pred_texts = decode_batch(tokenizer, batch, outputs)
                val_preds.extend(pred_texts)
                val_targets.extend(batch["clean_text"])
        
        avg_val_loss = val_loss / val_steps
        
        # Compute metrics
        wcr = compute_word_correction_rate(val_preds, val_targets)
        cer = compute_character_error_rate(val_preds, val_targets)
        
        logger.info(f"Validation loss: {avg_val_loss:.4f}")
        logger.info(f"Word Correction Rate: {wcr:.4f}")
        logger.info(f"Character Error Rate: {cer:.4f}")
        
        # Save stats
        epoch_stats = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "word_correction_rate": wcr,
            "character_error_rate": cer,
            "learning_rate": scheduler.get_last_lr()[0]
        }
        training_stats.append(epoch_stats)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            logger.info(f"New best validation loss: {best_val_loss:.4f}")
            logger.info(f"Saving model to {args.output_dir}/best_model")
            
            # Save model
            model.save_pretrained(os.path.join(args.output_dir, "best_model"))
            tokenizer.save_pretrained(os.path.join(args.output_dir, "best_model"))
            
            # Save training arguments
            with open(os.path.join(args.output_dir, "best_model", "training_args.json"), "w") as f:
                json.dump(vars(args), f, indent=2)
        
        # Save training stats
        with open(os.path.join(args.output_dir, "training_stats.json"), "w") as f:
            json.dump(training_stats, f, indent=2)
    
    # Save final model
    logger.info(f"Saving final model to {args.output_dir}/final_model")
    model.save_pretrained(os.path.join(args.output_dir, "final_model"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "final_model"))
    
    logger.info("Training complete!")

def main():
    parser = argparse.ArgumentParser(description="Train a spell correction model")
    
    # Required parameters
    parser.add_argument("--model_name", type=str, required=True,
                        help="Path to pre-trained model or model identifier from huggingface.co/models")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing train_clean.txt, train_noisy.txt, etc.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the model")
    
    # Training parameters
    parser.add_argument("--max_length", type=int, default=128,
                        help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Learning rate")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--warmup_ratio", type=float, default=0.1,
                        help="Ratio of warmup steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Maximum gradient norm")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8,
                        help="Epsilon for Adam optimizer")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    
    train(args)

if __name__ == "__main__":
    main()
