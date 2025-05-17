"""
Model optimization and compression for the fast spell correction system.
Implements techniques such as quantization, pruning, and model export.
"""

import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
import time
import json
from transformers import AutoModelForMaskedLM, AutoTokenizer
import logging

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class ModelOptimizer:
    """Class for optimizing and compressing transformer models"""
    
    def __init__(self, model_path, output_dir=None, device=None):
        """
        Initialize the model optimizer
        
        Args:
            model_path: Path to the pre-trained or fine-tuned model
            output_dir: Directory to save optimized models
            device: Device to use for optimization ('cpu' or 'cuda')
        """
        self.model_path = model_path
        self.output_dir = output_dir or os.path.join(model_path, "optimized")
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model and tokenizer
        logger.info(f"Loading model from {model_path}")
        self.model = AutoModelForMaskedLM.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
    
    def quantize_dynamic(self):
        """
        Apply dynamic quantization to the model
        
        Dynamic quantization converts weights to int8 precision on the fly during inference
        """
        logger.info("Applying dynamic quantization...")
        
        # Move model to CPU for quantization
        model_cpu = self.model.cpu()
        
        # Apply dynamic quantization
        quantized_model = torch.quantization.quantize_dynamic(
            model_cpu,  # model to quantize
            {torch.nn.Linear},  # specify layers to quantize
            dtype=torch.qint8  # target dtype for quantized weights
        )
        
        # Save quantized model
        output_path = os.path.join(self.output_dir, "quantized_dynamic")
        os.makedirs(output_path, exist_ok=True)
        
        # Save with PyTorch's native serialization
        torch.save(quantized_model.state_dict(), os.path.join(output_path, "pytorch_model.bin"))
        
        # Save tokenizer and config
        self.tokenizer.save_pretrained(output_path)
        
        # Save metadata
        metadata = {
            "original_model": self.model_path,
            "quantization_type": "dynamic",
            "quantization_dtype": "qint8",
            "device": "cpu",  # dynamic quantization only for CPU
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(os.path.join(output_path, "quantization_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Dynamic quantized model saved to {output_path}")
        
        return output_path
    
    def quantize_static(self, calibration_dataset=None):
        """
        Apply static quantization to the model
        
        Static quantization requires calibration data to determine optimal quantization parameters
        """
        logger.info("Static quantization requires calibration data and model modifications.")
        logger.info("For HuggingFace transformers, this is a complex process.")
        logger.info("Consider using dynamic quantization or export to ONNX for best results.")
        
        return None
    
    def prune_model(self, sparsity=0.3):
        """
        Apply weight pruning to the model
        
        Args:
            sparsity: Target sparsity (fraction of weights to prune)
        """
        logger.info(f"Applying weight pruning with sparsity {sparsity}...")
        
        # For transformers, we apply pruning to linear layers
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                # Simple magnitude pruning - set smallest weights to zero
                weight = module.weight.data.cpu().numpy()
                threshold = np.percentile(np.abs(weight), sparsity * 100)
                weight[np.abs(weight) < threshold] = 0
                sparsity_applied = np.sum(weight == 0) / weight.size
                logger.info(f"Applied {sparsity_applied:.4f} sparsity to {name}")
                module.weight.data = torch.from_numpy(weight).to(self.device)
        
        # Save pruned model
        output_path = os.path.join(self.output_dir, f"pruned_{int(sparsity * 100)}_pct")
        os.makedirs(output_path, exist_ok=True)
        
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        
        # Save metadata
        metadata = {
            "original_model": self.model_path,
            "pruning_type": "magnitude",
            "target_sparsity": sparsity,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(os.path.join(output_path, "pruning_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Pruned model saved to {output_path}")
        
        return output_path
    
    def export_to_torchscript(self):
        """
        Export the model to TorchScript format
        """
        logger.info("Exporting model to TorchScript...")
        
        # Move model to CPU for tracing
        model_cpu = self.model.cpu()
        
        # Create sample input
        sample_text = "This is a sample text to trace the model."
        inputs = self.tokenizer(
            sample_text, 
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        # Trace the model
        with torch.no_grad():
            traced_model = torch.jit.trace(
                model_cpu,
                (inputs["input_ids"], inputs["attention_mask"])
            )
        
        # Save traced model
        output_path = os.path.join(self.output_dir, "torchscript")
        os.makedirs(output_path, exist_ok=True)
        
        traced_model.save(os.path.join(output_path, "model.pt"))
        self.tokenizer.save_pretrained(output_path)
        
        # Save metadata
        metadata = {
            "original_model": self.model_path,
            "export_format": "torchscript",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "sample_input": sample_text
        }
        
        with open(os.path.join(output_path, "export_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"TorchScript model saved to {output_path}")
        
        return output_path
    
    def export_to_onnx(self):
        """
        Export the model to ONNX format
        """
        logger.info("Exporting model to ONNX...")
        
        try:
            # For ONNX export, we need to ensure torch.onnx is available
            import torch.onnx
        except ImportError:
            logger.error("torch.onnx not available. Please install the full PyTorch version.")
            return None
        
        # Create output directory
        output_path = os.path.join(self.output_dir, "onnx")
        os.makedirs(output_path, exist_ok=True)
        
        # Create sample input
        sample_text = "This is a sample text to export the model."
        inputs = self.tokenizer(
            sample_text, 
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        # Move model to CPU for export
        model_cpu = self.model.cpu()
        
        # Export to ONNX
        with torch.no_grad():
            torch.onnx.export(
                model_cpu,
                (inputs["input_ids"], inputs["attention_mask"]),
                os.path.join(output_path, "model.onnx"),
                input_names=["input_ids", "attention_mask"],
                output_names=["logits"],
                dynamic_axes={
                    "input_ids": {0: "batch_size", 1: "sequence_length"},
                    "attention_mask": {0: "batch_size", 1: "sequence_length"},
                    "logits": {0: "batch_size", 1: "sequence_length"}
                },
                opset_version=12
            )
        
        # Save tokenizer
        self.tokenizer.save_pretrained(output_path)
        
        # Save metadata
        metadata = {
            "original_model": self.model_path,
            "export_format": "onnx",
            "opset_version": 12,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "sample_input": sample_text
        }
        
        with open(os.path.join(output_path, "export_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"ONNX model saved to {output_path}")
        
        return output_path
    
    def benchmark(self, input_texts=None):
        """
        Benchmark the model for inference speed and memory usage
        
        Args:
            input_texts: List of sample texts to use for benchmarking
        """
        logger.info("Benchmarking model...")
        
        if input_texts is None:
            # Default benchmark texts of varying lengths
            input_texts = [
                "This is a short text.",
                "This is a medium length text with some spelling errors lik thsi and taht.",
                "This is a longer text that contains multiple sentences. It has some spelling errors and typos. " +
                "The purpse of this textt is to benchmark the model for inference speed and memory usage. " +
                "We want to make sure that the model can handle texts of different lengths efficiently."
            ]
        
        # Move model to evaluation mode on the target device
        self.model.to(self.device)
        self.model.eval()
        
        # Prepare inputs
        all_inputs = [
            self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            for text in input_texts
        ]
        
        # Warm-up
        logger.info("Warming up...")
        with torch.no_grad():
            for inputs in all_inputs:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                _ = self.model(**inputs)
        
        # Benchmark
        logger.info("Running benchmark...")
        results = []
        
        with torch.no_grad():
            for i, (text, inputs) in enumerate(zip(input_texts, all_inputs)):
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Measure latency
                start_time = time.time()
                for _ in range(10):  # Run multiple times for more accurate measurement
                    _ = self.model(**inputs)
                end_time = time.time()
                
                latency = (end_time - start_time) / 10 * 1000  # Convert to ms
                
                # Record results
                result = {
                    "sample_id": i,
                    "text_length": len(text),
                    "token_length": inputs["input_ids"].shape[1],
                    "latency_ms": latency,
                    "device": self.device
                }
                results.append(result)
                
                logger.info(f"Sample {i+1}: Length={len(text)}, Tokens={inputs['input_ids'].shape[1]}, Latency={latency:.2f}ms")
        
        # Save results
        output_path = os.path.join(self.output_dir, "benchmark_results.json")
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Benchmark results saved to {output_path}")
        
        return results

def main():
    parser = argparse.ArgumentParser(description="Optimize and compress models for spell correction")
    
    # Required parameters
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the model directory")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save optimized models (default: model_path/optimized)")
    
    # Optimization options
    parser.add_argument("--quantize", action="store_true",
                        help="Apply dynamic quantization")
    parser.add_argument("--prune", action="store_true",
                        help="Apply weight pruning")
    parser.add_argument("--sparsity", type=float, default=0.3,
                        help="Target sparsity for pruning (default: 0.3)")
    parser.add_argument("--torchscript", action="store_true",
                        help="Export to TorchScript format")
    parser.add_argument("--onnx", action="store_true",
                        help="Export to ONNX format")
    parser.add_argument("--benchmark", action="store_true",
                        help="Benchmark the model")
    
    # Device
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (default: cuda if available, else cpu)")
    
    args = parser.parse_args()
    
    # Create model optimizer
    optimizer = ModelOptimizer(args.model_path, args.output_dir, args.device)
    
    # Apply optimizations
    if args.quantize:
        optimizer.quantize_dynamic()
    
    if args.prune:
        optimizer.prune_model(args.sparsity)
    
    if args.torchscript:
        optimizer.export_to_torchscript()
    
    if args.onnx:
        optimizer.export_to_onnx()
    
    if args.benchmark:
        optimizer.benchmark()
    
    # If no optimization is selected, run all
    if not any([args.quantize, args.prune, args.torchscript, args.onnx, args.benchmark]):
        logger.info("No specific optimization selected, running all...")
        optimizer.quantize_dynamic()
        optimizer.prune_model(args.sparsity)
        optimizer.export_to_torchscript()
        optimizer.export_to_onnx()
        optimizer.benchmark()
    
    logger.info("Model optimization complete!")

if __name__ == "__main__":
    main()
