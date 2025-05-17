"""
Complete pipeline demonstration for the Fast Spell Correction system.
This script shows the entire workflow from data preparation to inference.
"""

import os
import argparse
import subprocess
import time
import logging
from pathlib import Path
import torch

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def run_command(command, description=None):
    """Run a command and log the output"""
    if description:
        logger.info(f"Step: {description}")
    
    logger.info(f"Running command: {command}")
    start_time = time.time()
    
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Stream output in real-time
    for line in process.stdout:
        print(line.strip())
    
    # Get return code
    process.wait()
    end_time = time.time()
    
    if process.returncode != 0:
        stderr = process.stderr.read()
        logger.error(f"Command failed with error: {stderr}")
        return False
    
    logger.info(f"Command completed in {end_time - start_time:.2f} seconds")
    return True

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run the complete fast spell correction pipeline")
    parser.add_argument("--data_dir", type=str, default="data", 
                        help="Directory for data files")
    parser.add_argument("--models_dir", type=str, default="models",
                        help="Directory for model files")
    parser.add_argument("--outputs_dir", type=str, default="outputs",
                        help="Directory for output files")
    parser.add_argument("--model_name", type=str, default="prajjwal1/bert-tiny",
                        help="Pre-trained model to use")
    parser.add_argument("--skip_data_prep", action="store_true",
                        help="Skip data preparation steps")
    parser.add_argument("--skip_training", action="store_true",
                        help="Skip model training steps")
    parser.add_argument("--skip_optimization", action="store_true",
                        help="Skip model optimization steps")
    parser.add_argument("--skip_evaluation", action="store_true",
                        help="Skip evaluation steps")
    parser.add_argument("--run_demo", action="store_true",
                        help="Run the web demo after pipeline completion")
    
    return parser.parse_args()

def setup_directories(args):
    """Set up directory structure"""
    # Create main directories
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(args.outputs_dir, exist_ok=True)
    
    # Create subdirectories
    os.makedirs(os.path.join(args.data_dir, "synthetic_errors"), exist_ok=True)
    os.makedirs(os.path.join(args.data_dir, "real_errors"), exist_ok=True)
    
    # Create model subdirectories
    model_name_short = args.model_name.split("/")[-1]
    base_model_dir = os.path.join(args.models_dir, model_name_short)
    os.makedirs(base_model_dir, exist_ok=True)
    
    spell_model_dir = os.path.join(args.models_dir, f"spell-{model_name_short}")
    os.makedirs(spell_model_dir, exist_ok=True)
    
    # Create output subdirectories
    os.makedirs(os.path.join(args.outputs_dir, "benchmark"), exist_ok=True)
    
    return {
        "base_model_dir": base_model_dir,
        "spell_model_dir": spell_model_dir
    }

def data_preparation(args):
    """Run data preparation steps"""
    logger.info("=== Data Preparation Phase ===")
    
    # Check if Shakespeare corpus exists, generate if it doesn't
    if not os.path.exists(os.path.join(args.data_dir, "clean_corpus.txt")):
        logger.info("Preparing Shakespeare corpus as a fallback dataset")
        if not run_command(f"python src/prepare_sentences.py --output_file {os.path.join(args.data_dir, 'clean_corpus.txt')}",
                        "Preparing Shakespeare corpus"):
            return False
    
    # Run enhanced data preparation
    synthetic_errors_dir = os.path.join(args.data_dir, "synthetic_errors")
    if not run_command(f"python src/enhanced_data_prep.py --clean_corpus {os.path.join(args.data_dir, 'clean_corpus.txt')} "
                    f"--output_dir {synthetic_errors_dir} --p_word 0.2 --p_sentence 0.1",
                    "Generating synthetic errors with enhanced techniques"):
        return False
    
    logger.info("Data preparation completed successfully")
    return True

def model_training(args, directories):
    """Run model training steps"""
    logger.info("=== Model Training Phase ===")
    
    # Download pre-trained model if it doesn't exist
    if not os.path.exists(os.path.join(directories["base_model_dir"], "pytorch_model.bin")):
        if not run_command(f"python src/download_model.py --model {args.model_name} --output_dir {directories['base_model_dir']}",
                        f"Downloading pre-trained model {args.model_name}"):
            return False
    
    # Train with enhanced training script
    synthetic_errors_dir = os.path.join(args.data_dir, "synthetic_errors")
    if not run_command(f"python src/train_enhanced.py --model_name {directories['base_model_dir']} "
                    f"--data_dir {synthetic_errors_dir} --output_dir {directories['spell_model_dir']} "
                    f"--epochs 3 --batch_size 32 --max_length 128",
                    "Training spell correction model"):
        return False
    
    logger.info("Model training completed successfully")
    return True

def model_optimization(args, directories):
    """Run model optimization steps"""
    logger.info("=== Model Optimization Phase ===")
    
    # Best model path
    best_model_path = os.path.join(directories["spell_model_dir"], "best_model")
    
    if not os.path.exists(best_model_path):
        logger.warning(f"Best model not found at {best_model_path}, falling back to final model")
        best_model_path = os.path.join(directories["spell_model_dir"], "final_model")
        
        if not os.path.exists(best_model_path):
            logger.error("No trained model found, optimization cannot proceed")
            return False
    
    # Run optimization
    if not run_command(f"python src/optimize_model.py --model_path {best_model_path} --quantize --prune --torchscript",
                    "Optimizing model with quantization, pruning, and TorchScript export"):
        return False
    
    logger.info("Model optimization completed successfully")
    return True

def model_evaluation(args, directories):
    """Run model evaluation steps"""
    logger.info("=== Model Evaluation Phase ===")
    
    # Best model path
    best_model_path = os.path.join(directories["spell_model_dir"], "best_model")
    
    if not os.path.exists(best_model_path):
        logger.warning(f"Best model not found at {best_model_path}, falling back to final model")
        best_model_path = os.path.join(directories["spell_model_dir"], "final_model")
        
        if not os.path.exists(best_model_path):
            logger.error("No trained model found, evaluation cannot proceed")
            return False
    
    # Run comprehensive benchmark
    benchmark_output = os.path.join(args.outputs_dir, "benchmark")
    synthetic_errors_dir = os.path.join(args.data_dir, "synthetic_errors")
    
    if not run_command(f"python src/benchmark.py --model_path {best_model_path} "
                    f"--data_dir {synthetic_errors_dir} --output_dir {benchmark_output}",
                    "Running comprehensive benchmark"):
        return False
    
    # Test inference
    if not run_command(f"python src/infer.py --model_path {best_model_path} "
                    f"--input_text \"Thsi is a smiple test of the spell corection systm.\"",
                    "Testing inference on a sample sentence"):
        return False
    
    logger.info("Model evaluation completed successfully")
    return True

def run_demo(args, directories):
    """Run the web demo"""
    logger.info("=== Starting Web Demo ===")
    
    # Best model path
    best_model_path = os.path.join(directories["spell_model_dir"], "best_model")
    
    if not os.path.exists(best_model_path):
        logger.warning(f"Best model not found at {best_model_path}, falling back to final model")
        best_model_path = os.path.join(directories["spell_model_dir"], "final_model")
        
        if not os.path.exists(best_model_path):
            logger.error("No trained model found, demo cannot proceed")
            return False
    
    # Set model path environment variable
    os.environ["MODEL_PATH"] = best_model_path
    
    # Check for optimized model
    optimized_path = os.path.join(best_model_path, "optimized", "quantized_dynamic")
    if os.path.exists(optimized_path):
        logger.info(f"Using optimized model from {optimized_path}")
        os.environ["MODEL_PATH"] = optimized_path
    
    # Run enhanced web demo
    logger.info("Starting web demo. Press Ctrl+C to stop.")
    logger.info("Once started, open your browser to http://localhost:8000/")
    
    try:
        run_command("python src/serve_enhanced.py", "Running enhanced web demo")
    except KeyboardInterrupt:
        logger.info("Web demo stopped by user")
    
    return True

def main():
    """Main function to run the pipeline"""
    args = parse_args()
    logger.info("Starting Fast Spell Correction pipeline")
    
    # Setup directories
    directories = setup_directories(args)
    
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Run pipeline phases
    if not args.skip_data_prep:
        if not data_preparation(args):
            logger.error("Data preparation failed, stopping pipeline")
            return
    
    if not args.skip_training:
        if not model_training(args, directories):
            logger.error("Model training failed, stopping pipeline")
            return
    
    if not args.skip_optimization:
        if not model_optimization(args, directories):
            logger.error("Model optimization failed, stopping pipeline")
            return
    
    if not args.skip_evaluation:
        if not model_evaluation(args, directories):
            logger.error("Model evaluation failed, stopping pipeline")
            return
    
    logger.info("Pipeline completed successfully!")
    
    if args.run_demo:
        run_demo(args, directories)

if __name__ == "__main__":
    main()
