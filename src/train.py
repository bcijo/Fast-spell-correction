"""
Training script for the fast spell correction model.
This script handles fine-tuning a pre-trained masked language model for spell correction.
"""

import os
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForMaskedLM, AdamW
from tqdm import tqdm
import random
import numpy as np

class SpellCorrectionDataset(Dataset):
    def __init__(self, noisy_texts, clean_texts, tokenizer, max_length=128):
        self.noisy_texts = noisy_texts
        self.clean_texts = clean_texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.noisy_texts)
    
    def __getitem__(self, idx):
        noisy_text = self.noisy_texts[idx]
        clean_text = self.clean_texts[idx]
        
        # Tokenize both texts
        noisy_encoding = self.tokenizer(
            noisy_text, 
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        clean_encoding = self.tokenizer(
            clean_text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        # Squeeze to remove batch dimension
        noisy_ids = noisy_encoding['input_ids'].squeeze()
        noisy_mask = noisy_encoding['attention_mask'].squeeze()
        clean_ids = clean_encoding['input_ids'].squeeze()
        
        # Create labels: where tokens differ, use clean tokens as targets
        # Where they're the same, use -100 to ignore in loss calculation
        labels = torch.where(
            noisy_ids == clean_ids,
            torch.tensor(-100),
            clean_ids
        )
        
        return {
            'input_ids': noisy_ids,
            'attention_mask': noisy_mask,
            'labels': labels
        }

def load_data(data_dir, split):
    """
    Load data from the given directory and split
    """
    clean_path = os.path.join(data_dir, f"{split}_clean.txt")
    noisy_path = os.path.join(data_dir, f"{split}_noisy.txt")
    
    with open(clean_path, 'r', encoding='utf-8') as f:
        clean_texts = [line.strip() for line in f if line.strip()]
    
    with open(noisy_path, 'r', encoding='utf-8') as f:
        noisy_texts = [line.strip() for line in f if line.strip()]
    
    assert len(clean_texts) == len(noisy_texts), "Clean and noisy text files must have the same number of lines"
    
    print(f"Loaded {len(clean_texts)} {split} examples")
    return noisy_texts, clean_texts

def set_seed(seed):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train(args):
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load the pre-trained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForMaskedLM.from_pretrained(args.model_name)
    model.to(device)
    
    # Load data
    train_noisy, train_clean = load_data(args.data_dir, "train")
    val_noisy, val_clean = load_data(args.data_dir, "val")
    
    # Create datasets
    train_dataset = SpellCorrectionDataset(train_noisy, train_clean, tokenizer, args.max_length)
    val_dataset = SpellCorrectionDataset(val_noisy, val_clean, tokenizer, args.max_length)
    
    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    # Setup optimizer
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_steps = 0
        
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_steps += 1
        
        avg_train_loss = train_loss / train_steps
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_steps = 0
        
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                val_loss += loss.item()
                val_steps += 1
        
        avg_val_loss = val_loss / val_steps
        
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
        
        # Save the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            
            # Create output directory if it doesn't exist
            os.makedirs(args.output_dir, exist_ok=True)
            
            # Save model and tokenizer
            model.save_pretrained(os.path.join(args.output_dir, "best_model"))
            tokenizer.save_pretrained(os.path.join(args.output_dir, "best_model"))
            
            print(f"New best model saved with val loss: {best_val_loss:.4f}")
    
    # Save the final model
    model.save_pretrained(os.path.join(args.output_dir, "final_model"))
    tokenizer.save_pretrained(os.path.join(args.output_dir, "final_model"))
    
    print("Training complete!")

def main():
    parser = argparse.ArgumentParser(description="Train a spell correction model")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing the train/val/test data")
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased",
                        help="Name or path of the pre-trained model")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the fine-tuned model")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training")
    parser.add_argument("--max_length", type=int, default=128,
                        help="Maximum sequence length")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Learning rate")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    train(args)

if __name__ == "__main__":
    main()
