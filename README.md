# Fast, Non-LLM Spell Correction System

A lightweight, fast spell correction system built with PyTorch. It leverages masked language models with optimization techniques to achieve excellent performance without requiring a full-sized LLM for inference.

## Features

- Character-level and phonetic error modeling
- Fast, optimized inference via model quantization and pruning
- Cross-platform deployment via TorchScript and ONNX exports
- Web-based demo interface with FastAPI backend
- Comprehensive evaluation metrics for both quality and performance

## Project Structure

```
spell-correction/
├── data/
│   ├── clean_corpus.txt              # Base corpus for training
│   ├── clean_sentences.txt           # Processed sentences
│   ├── combined_clean_corpus.txt     # Combined corpus from multiple sources
│   ├── real_errors/                  # Real spelling error datasets
│   └── synthetic_errors/             # Generated errors for training
│       ├── train_clean.txt
│       ├── train_noisy.txt
│       ├── val_clean.txt
│       ├── val_noisy.txt
│       ├── test_clean.txt
│       └── test_noisy.txt
├── models/                           # Saved models
├── outputs/                          # Evaluation outputs
├── src/
│   ├── api.py                        # Basic FastAPI implementation
│   ├── benchmark.py                  # Comprehensive evaluation script
│   ├── data_prep.py                  # Basic error generation
│   ├── download_corpus.py            # Downloads text corpora
│   ├── download_model.py             # Downloads pre-trained models
│   ├── enhanced_data_prep.py         # Advanced error generation
│   ├── eval.py                       # Basic evaluation metrics
│   ├── infer.py                      # Inference implementation
│   ├── optimize_model.py             # Model compression techniques
│   ├── prepare_sentences.py          # Preprocesses text data
│   ├── serve.py                      # Web demo server
│   ├── serve_enhanced.py             # Enhanced web interface
│   ├── simple_demo.py                # Simple PyTorch demo
│   ├── test_model.py                 # Model testing utilities
│   ├── train.py                      # Basic training script
│   ├── train_enhanced.py             # Enhanced training with more metrics
│   ├── static/                       # Web interface files
│   │   ├── index.html
│   │   └── enhanced_index.html
│   └── tests/                        # Unit tests
│       └── test_spell_correction.py
├── requirements.txt                  # Python dependencies
├── environment.yml                   # Conda environment
└── LICENSE                           # MIT License
```

## Setup

### Option 1: Using Conda

```bash
# Create and activate the environment
conda env create -f environment.yml
conda activate fast_spell
```

### Option 2: Using pip

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Workflow

### 1. Data Collection and Preparation

```bash
# Download and prepare corpora
python src/download_corpus.py

# Generate synthetic errors with advanced techniques
python src/enhanced_data_prep.py --clean_corpus data/combined_clean_corpus.txt --output_dir data/synthetic_errors --p_word 0.2 --p_sentence 0.1
```

### 2. Model Development

```bash
# Download a pre-trained model
python src/download_model.py --model prajjwal1/bert-tiny --output_dir models/bert-tiny

# Train the model
python src/train_enhanced.py --model_name models/bert-tiny --data_dir data/synthetic_errors --output_dir models/spell-corrector --epochs 3 --batch_size 32
```

### 3. Model Optimization

```bash
# Optimize the model (quantization, pruning, export)
python src/optimize_model.py --model_path models/spell-corrector/best_model --quantize --prune --torchscript --onnx
```

### 4. Evaluation and Benchmarking

```bash
# Comprehensive evaluation
python src/benchmark.py --model_path models/spell-corrector/best_model --data_dir data/synthetic_errors --output_dir outputs/benchmark-results
```

### 5. Web Demo and Deployment

```bash
# Start the enhanced web server
python src/serve_enhanced.py
```

Then open your browser to http://localhost:8000/ to access the demo interface.

## Model Optimization Techniques

The system supports several optimization techniques:

1. **Dynamic Quantization**: Reduces model size by up to 4x and improves inference speed
2. **Pruning**: Removes less important weights for faster computation (up to 30% sparsity)
3. **TorchScript Export**: Optimized deployment in production environments
4. **ONNX Export**: Cross-platform deployment with hardware-specific optimizations

## Performance Benchmarks

The system achieves:

- Token-level accuracy > 95% on synthetic benchmarks
- Word Correction Rate (WCR) > 85% on standard spelling correction datasets
- Error Reduction Rate (ERR) > 70% on real-world errors
- Inference latency < 20ms per sentence on CPU (quantized model)
- Model size < 50MB after optimization (for bert-tiny variant)
- Peak memory usage < 200MB during inference

## Technical Approach

The project utilizes a fine-tuned masked language model approach rather than relying on traditional rule-based systems or large language models. By employing model compression techniques like quantization and pruning, the system achieves real-time performance on CPU hardware while maintaining high correction accuracy.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## License

[MIT License](LICENSE)
