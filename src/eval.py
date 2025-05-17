"""
Evaluation script for the fast spell correction model.
This script handles evaluation of the model on test data and computes various metrics.
"""

import os
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForMaskedLM
from tqdm import tqdm
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import time
import json

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
        
        return {
            'input_ids': noisy_ids,
            'attention_mask': noisy_mask,
            'target_ids': clean_ids,
            'noisy_text': noisy_text,
            'clean_text': clean_text
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

def compute_word_correction_rate(predicted_texts, noisy_texts, clean_texts):
    """
    Compute the Word Correction Rate (WCR)
    """
    total_errors = 0
    corrected_errors = 0
    
    for noisy, clean, pred in zip(noisy_texts, clean_texts, predicted_texts):
        noisy_words = noisy.split()
        clean_words = clean.split()
        pred_words = pred.split()
        
        # Make sure we have the same number of words
        min_len = min(len(noisy_words), len(clean_words), len(pred_words))
        noisy_words = noisy_words[:min_len]
        clean_words = clean_words[:min_len]
        pred_words = pred_words[:min_len]
        
        for i in range(min_len):
            if noisy_words[i] != clean_words[i]:
                total_errors += 1
                if pred_words[i] == clean_words[i]:
                    corrected_errors += 1
    
    wcr = corrected_errors / total_errors if total_errors > 0 else 1.0
    return wcr, total_errors, corrected_errors

def evaluate(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load the fine-tuned model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForMaskedLM.from_pretrained(args.model_path)
    model.to(device)
    model.eval()
    
    # Load test data
    test_noisy, test_clean = load_data(args.data_dir, "test")
    
    # Create dataset and dataloader
    test_dataset = SpellCorrectionDataset(test_noisy, test_clean, tokenizer, args.max_length)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    # Prepare metrics
    total_tokens = 0
    correct_tokens = 0
    
    # Lists to store predictions and targets for further analysis
    all_predictions = []
    all_targets = []
    predicted_texts = []
    
    # Measure inference time
    total_latency = 0
    num_sentences = 0
    
    for batch in tqdm(test_dataloader, desc="Evaluating"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        target_ids = batch['target_ids'].to(device)
        
        # Measure inference time
        start_time = time.time()
        
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        
        end_time = time.time()
        batch_latency = end_time - start_time
        total_latency += batch_latency
        num_sentences += input_ids.size(0)
        
        # Get predictions
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        
        # Mask out padding tokens
        valid_mask = attention_mask.bool()
        
        # Convert predictions to text
        for i in range(predictions.size(0)):
            pred_ids = predictions[i][valid_mask[i]]
            pred_text = tokenizer.decode(pred_ids, skip_special_tokens=True)
            predicted_texts.append(pred_text)
        
        # Calculate token-level accuracy
        for i in range(predictions.size(0)):
            for j in range(predictions.size(1)):
                if attention_mask[i, j] == 1:  # Only consider non-padding tokens
                    total_tokens += 1
                    if predictions[i, j] == target_ids[i, j]:
                        correct_tokens += 1
        
        # Store predictions and targets for further analysis
        all_predictions.extend(predictions[valid_mask].cpu().numpy())
        all_targets.extend(target_ids[valid_mask].cpu().numpy())
    
    # Calculate metrics
    token_accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0
    
    # Calculate Word Correction Rate (WCR)
    wcr, total_errors, corrected_errors = compute_word_correction_rate(
        predicted_texts, test_noisy, test_clean
    )
    
    # Calculate average latency
    avg_latency_ms = (total_latency / num_sentences) * 1000  # Convert to ms
    throughput = num_sentences / total_latency  # sentences per second
    
    # Calculate model size
    model_size_params = sum(p.numel() for p in model.parameters())
    model_size_mb = model_size_params * 4 / (1024 * 1024)  # Assuming 4 bytes per parameter
    
    # Print results
    print("\n===== Evaluation Results =====")
    print(f"Token-level Accuracy: {token_accuracy:.4f}")
    print(f"Word Correction Rate (WCR): {wcr:.4f} ({corrected_errors}/{total_errors} errors corrected)")
    print(f"Average Latency: {avg_latency_ms:.2f} ms per sentence")
    print(f"Throughput: {throughput:.2f} sentences/second")
    print(f"Model Size: {model_size_mb:.2f} MB ({model_size_params} parameters)")
    
    # Save results to file
    results = {
        "token_accuracy": token_accuracy,
        "word_correction_rate": wcr,
        "errors_corrected": corrected_errors,
        "total_errors": total_errors,
        "avg_latency_ms": avg_latency_ms,
        "throughput": throughput,
        "model_size_mb": model_size_mb,
        "model_parameters": model_size_params,
        "device": str(device)
    }
    
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "eval_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {os.path.join(args.output_dir, 'eval_results.json')}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate a spell correction model")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing the test data")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the fine-tuned model")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save evaluation results")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for evaluation")
    parser.add_argument("--max_length", type=int, default=128,
                        help="Maximum sequence length")
    
    args = parser.parse_args()
    
    evaluate(args)

if __name__ == "__main__":
    main()
