"""
Comprehensive evaluation and benchmarking for the fast spell correction system.
Calculates quality metrics (WCR, CER) and performance metrics (latency, throughput, memory).
"""

import os
import argparse
import torch
import time
import json
import numpy as np
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForMaskedLM, AutoTokenizer
from sklearn.metrics import precision_recall_fscore_support
import logging
import psutil
import matplotlib.pyplot as plt
from Levenshtein import distance as levenshtein_distance

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class SpellCorrectionDataset(Dataset):
    """Dataset for evaluating spell correction models"""
    
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
        target_ids = clean_tokens["input_ids"].squeeze()
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "target_ids": target_ids,
            "clean_text": clean_text,
            "noisy_text": noisy_text
        }

def compute_word_correction_rate(predicted_texts, noisy_texts, clean_texts):
    """
    Compute the Word Correction Rate (WCR)
    
    WCR measures the percentage of erroneous words that are correctly fixed.
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

def compute_character_error_rate(predicted_texts, clean_texts):
    """
    Compute Character Error Rate (CER) using Levenshtein distance
    
    CER = sum(levenshtein_distance(pred, target)) / sum(len(target))
    """
    total_chars = sum(len(t) for t in clean_texts)
    total_distance = sum(levenshtein_distance(p, t) for p, t in zip(predicted_texts, clean_texts))
    
    cer = total_distance / total_chars if total_chars > 0 else 0.0
    return cer

def compute_error_reduction_rate(predicted_texts, noisy_texts, clean_texts):
    """
    Compute Error Reduction Rate (ERR)
    
    ERR measures how much the model reduces the error rate compared to the noisy input.
    ERR = 1 - (error_rate_after / error_rate_before)
    """
    # Calculate error rate before correction
    before_distance = sum(levenshtein_distance(n, t) for n, t in zip(noisy_texts, clean_texts))
    before_chars = sum(len(t) for t in clean_texts)
    error_rate_before = before_distance / before_chars if before_chars > 0 else 0.0
    
    # Calculate error rate after correction
    after_distance = sum(levenshtein_distance(p, t) for p, t in zip(predicted_texts, clean_texts))
    error_rate_after = after_distance / before_chars if before_chars > 0 else 0.0
    
    # Calculate error reduction rate
    err = 1.0 - (error_rate_after / error_rate_before) if error_rate_before > 0 else 0.0
    return err

def decode_model_outputs(tokenizer, input_ids, attention_mask, predictions):
    """
    Decode model predictions to text
    """
    predicted_texts = []
    
    for i in range(predictions.size(0)):
        # Get mask for valid tokens
        valid_mask = attention_mask[i].bool()
        
        # Get predicted token IDs for valid tokens
        pred_ids = predictions[i][valid_mask]
        
        # Decode to text
        pred_text = tokenizer.decode(pred_ids, skip_special_tokens=True)
        predicted_texts.append(pred_text)
    
    return predicted_texts

def plot_latency_vs_length(latencies, text_lengths, output_path):
    """
    Plot latency vs text length
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(text_lengths, latencies, alpha=0.6)
    plt.xlabel('Text Length (characters)')
    plt.ylabel('Latency (ms)')
    plt.title('Latency vs Text Length')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add trend line
    z = np.polyfit(text_lengths, latencies, 1)
    p = np.poly1d(z)
    plt.plot(text_lengths, p(text_lengths), "r--", alpha=0.8)
    
    plt.tight_layout()
    plt.savefig(output_path)
    logger.info(f"Latency plot saved to {output_path}")

def evaluate(args):
    """
    Evaluate the model on test data
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load model and tokenizer
    logger.info(f"Loading model from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForMaskedLM.from_pretrained(args.model_path)
    model.to(device)
    model.eval()
    
    # Create dataset and dataloader
    test_dataset = SpellCorrectionDataset(
        tokenizer,
        os.path.join(args.data_dir, "test_clean.txt"),
        os.path.join(args.data_dir, "test_noisy.txt"),
        max_length=args.max_length
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )
    
    # Lists to store results
    predicted_texts = []
    clean_texts = []
    noisy_texts = []
    latencies = []
    text_lengths = []
    
    # Memory peak usage
    process = psutil.Process(os.getpid())
    baseline_memory = process.memory_info().rss / (1024 * 1024)  # MB
    peak_memory = baseline_memory
    
    # Process batches
    logger.info("Evaluating model...")
    for batch in tqdm(test_dataloader):
        # Move batch to device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        # Store clean and noisy texts for later
        clean_texts.extend(batch["clean_text"])
        noisy_texts.extend(batch["noisy_text"])
        
        # Record text lengths
        batch_text_lengths = [len(text) for text in batch["clean_text"]]
        text_lengths.extend(batch_text_lengths)
        
        # Measure inference time
        start_time = time.time()
        
        with torch.no_grad():
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )
        
        end_time = time.time()
        batch_latency = (end_time - start_time) * 1000  # Convert to ms
        latencies.extend([batch_latency / len(batch["clean_text"])] * len(batch["clean_text"]))
        
        # Check memory usage
        current_memory = process.memory_info().rss / (1024 * 1024)  # MB
        peak_memory = max(peak_memory, current_memory)
        
        # Get predictions
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        
        # Decode predictions to text
        batch_predicted_texts = decode_model_outputs(
            tokenizer,
            batch["input_ids"],
            batch["attention_mask"],
            predictions
        )
        
        predicted_texts.extend(batch_predicted_texts)
    
    # Calculate metrics
    logger.info("Calculating metrics...")
    
    # Quality metrics
    wcr, total_errors, corrected_errors = compute_word_correction_rate(
        predicted_texts, noisy_texts, clean_texts
    )
    
    cer = compute_character_error_rate(predicted_texts, clean_texts)
    err = compute_error_reduction_rate(predicted_texts, noisy_texts, clean_texts)
    
    # Performance metrics
    avg_latency = np.mean(latencies)
    p95_latency = np.percentile(latencies, 95)
    p99_latency = np.percentile(latencies, 99)
    throughput = 1000 / avg_latency  # sentences per second
    memory_overhead = peak_memory - baseline_memory
    
    # Calculate model size
    model_size_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    model_size_mb = model_size_bytes / (1024 * 1024)
    
    # Create results directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save latency plot
    plot_path = os.path.join(args.output_dir, "latency_vs_length.png")
    plot_latency_vs_length(latencies, text_lengths, plot_path)
    
    # Save all results to file
    results = {
        "model_path": args.model_path,
        "data_dir": args.data_dir,
        "quality_metrics": {
            "word_correction_rate": wcr,
            "total_errors": total_errors,
            "corrected_errors": corrected_errors,
            "character_error_rate": cer,
            "error_reduction_rate": err
        },
        "performance_metrics": {
            "avg_latency_ms": avg_latency,
            "p95_latency_ms": p95_latency,
            "p99_latency_ms": p99_latency,
            "throughput_sentences_per_second": throughput,
            "model_size_mb": model_size_mb,
            "peak_memory_usage_mb": peak_memory,
            "memory_overhead_mb": memory_overhead
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    results_path = os.path.join(args.output_dir, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    logger.info("\n===== Evaluation Results =====")
    logger.info(f"Word Correction Rate (WCR): {wcr:.4f} ({corrected_errors}/{total_errors} errors corrected)")
    logger.info(f"Character Error Rate (CER): {cer:.4f}")
    logger.info(f"Error Reduction Rate: {err:.4f}")
    logger.info(f"Average Latency: {avg_latency:.2f} ms per sentence")
    logger.info(f"P95 Latency: {p95_latency:.2f} ms")
    logger.info(f"Throughput: {throughput:.2f} sentences per second")
    logger.info(f"Model Size: {model_size_mb:.2f} MB")
    logger.info(f"Memory Overhead: {memory_overhead:.2f} MB")
    logger.info(f"Results saved to {results_path}")
    
    # Save some example corrections
    examples = []
    for i in range(min(10, len(clean_texts))):
        examples.append({
            "noisy": noisy_texts[i],
            "predicted": predicted_texts[i],
            "target": clean_texts[i]
        })
    
    examples_path = os.path.join(args.output_dir, "correction_examples.json")
    with open(examples_path, "w") as f:
        json.dump(examples, f, indent=2)
    
    logger.info(f"Correction examples saved to {examples_path}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Evaluate a spell correction model")
    
    # Required parameters
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the model directory")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing test_clean.txt and test_noisy.txt")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save evaluation results")
    
    # Optional parameters
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for evaluation")
    parser.add_argument("--max_length", type=int, default=128,
                        help="Maximum sequence length")
    
    args = parser.parse_args()
    
    evaluate(args)

if __name__ == "__main__":
    main()
