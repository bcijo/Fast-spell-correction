"""
Inference script for the fast spell correction model.
This script provides functions for loading a model and performing spell correction.
"""

import os
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import time
import torch.quantization
import Levenshtein

class SpellCorrector:
    def __init__(self, model_path, device=None, quantized=False):
        """
        Initialize the spell corrector with a model and tokenizer
        
        Args:
            model_path: Path to the fine-tuned model
            device: Device to use (cuda/cpu)
            quantized: Whether to load a quantized model
        """
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load model
        self.model = AutoModelForMaskedLM.from_pretrained(model_path)
        
        # Quantize model if requested
        if quantized:
            print("Quantizing model...")
            self.model = torch.quantization.quantize_dynamic(
                self.model, {torch.nn.Linear}, dtype=torch.qint8
            )
        
        self.model.to(self.device)
        self.model.eval()
        
        # Build vocabulary set for quick lookup
        self.vocab = set(self.tokenizer.get_vocab().keys())
        
        print(f"Model loaded from {model_path}")
    
    def is_suspect(self, token, threshold_prob=0.01):
        """
        Detect if a token is potentially misspelled
        
        Args:
            token: The token to check
            threshold_prob: Probability threshold for suspicion
        
        Returns:
            bool: True if the token is suspect, False otherwise
        """
        # If token is not in vocabulary, it's suspect
        if token not in self.vocab:
            return True
        
        # If token is very short, it's less likely to be suspect
        if len(token) <= 2:
            return False
        
        # You could add more sophisticated detection here
        return False
    
    def correct_text(self, text, max_edit_distance=2):
        """
        Correct misspelled words in the text
        
        Args:
            text: The text to correct
            max_edit_distance: Maximum edit distance for accepting corrections
        
        Returns:
            str: The corrected text
        """
        start_time = time.time()
        
        # Tokenize the text
        tokens = self.tokenizer.tokenize(text)
        
        # Create a list to track suspect tokens
        suspect_indices = []
        for i, token in enumerate(tokens):
            if self.is_suspect(token):
                suspect_indices.append(i)
        
        corrected_tokens = tokens.copy()
        
        # Process each suspect token
        for i in suspect_indices:
            # Create a masked version of the input
            masked_tokens = tokens.copy()
            masked_tokens[i] = self.tokenizer.mask_token
            
            # Convert to input IDs
            inputs = self.tokenizer.convert_tokens_to_string(masked_tokens)
            inputs = self.tokenizer(inputs, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get model predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Get the predicted token
            mask_idx = torch.where(inputs["input_ids"][0] == self.tokenizer.mask_token_id)[0].item()
            probs = torch.nn.functional.softmax(outputs.logits[0, mask_idx], dim=-1)
            top_k_probs, top_k_indices = torch.topk(probs, 5)
            
            # Get the best candidate
            for idx, prob in zip(top_k_indices.tolist(), top_k_probs.tolist()):
                candidate = self.tokenizer.convert_ids_to_tokens(idx)
                
                # Skip if the candidate is a special token
                if candidate.startswith("[") and candidate.endswith("]"):
                    continue
                
                # Check edit distance
                orig_token = tokens[i]
                if Levenshtein.distance(orig_token, candidate) <= max_edit_distance:
                    corrected_tokens[i] = candidate
                    break
        
        # Convert tokens back to text
        corrected_text = self.tokenizer.convert_tokens_to_string(corrected_tokens)
        
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000
        
        return corrected_text, latency_ms
    
    def export_to_torchscript(self, output_path):
        """
        Export the model to TorchScript for faster inference
        
        Args:
            output_path: Path to save the exported model
        """
        self.model.eval()
        
        # Create an example input
        example = self.tokenizer("example sentence", return_tensors="pt")
        example = {k: v.to(self.device) for k, v in example.items()}
        
        # Trace the model
        try:
            traced_model = torch.jit.trace(self.model, (example["input_ids"],))
            torch.jit.save(traced_model, output_path)
            print(f"Model exported to {output_path}")
            return True
        except Exception as e:
            print(f"Failed to export model: {e}")
            return False
    
    def export_to_onnx(self, output_path):
        """
        Export the model to ONNX for cross-platform inference
        
        Args:
            output_path: Path to save the exported model
        """
        self.model.eval()
        
        # Create an example input
        example = self.tokenizer("example sentence", return_tensors="pt")
        example = {k: v.to(self.device) for k, v in example.items()}
        
        # Export to ONNX
        try:
            torch.onnx.export(
                self.model,
                (example["input_ids"],),
                output_path,
                input_names=["input_ids"],
                output_names=["logits"],
                opset_version=11
            )
            print(f"Model exported to {output_path}")
            return True
        except Exception as e:
            print(f"Failed to export model: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description="Inference with spell correction model")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the fine-tuned model")
    parser.add_argument("--input_text", type=str,
                        help="Text to correct")
    parser.add_argument("--input_file", type=str,
                        help="File containing text to correct")
    parser.add_argument("--output_file", type=str,
                        help="File to write corrected text to")
    parser.add_argument("--quantized", action="store_true",
                        help="Whether to use quantized model")
    parser.add_argument("--export_torchscript", type=str,
                        help="Export model to TorchScript and save to this path")
    parser.add_argument("--export_onnx", type=str,
                        help="Export model to ONNX and save to this path")
    
    args = parser.parse_args()
    
    # Initialize spell corrector
    corrector = SpellCorrector(args.model_path, quantized=args.quantized)
    
    # Export model if requested
    if args.export_torchscript:
        corrector.export_to_torchscript(args.export_torchscript)
    
    if args.export_onnx:
        corrector.export_to_onnx(args.export_onnx)
    
    # Process input text
    if args.input_text:
        corrected_text, latency_ms = corrector.correct_text(args.input_text)
        print(f"Input: {args.input_text}")
        print(f"Corrected: {corrected_text}")
        print(f"Latency: {latency_ms:.2f} ms")
    
    # Process input file
    if args.input_file:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        corrected_lines = []
        total_latency = 0
        
        for line in lines:
            corrected_line, latency_ms = corrector.correct_text(line)
            corrected_lines.append(corrected_line)
            total_latency += latency_ms
        
        # Write to output file if specified
        if args.output_file:
            with open(args.output_file, 'w', encoding='utf-8') as f:
                for line in corrected_lines:
                    f.write(line + '\n')
            
            print(f"Corrected text written to {args.output_file}")
        
        # Print statistics
        avg_latency = total_latency / len(lines) if lines else 0
        print(f"Processed {len(lines)} lines")
        print(f"Average latency: {avg_latency:.2f} ms per line")
        print(f"Throughput: {1000 * len(lines) / total_latency:.2f} lines/second")

if __name__ == "__main__":
    main()
