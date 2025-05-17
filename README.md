# Fast, Non-LLM Spell Correction System

This repository contains a lightweight, fast spell correction system built with PyTorch. It leverages the power of masked language models without requiring a full-sized LLM for inference.

## Features

- Character-level and phonetic error modeling
- Fast, optimized inference via model quantization and pruning
- Cross-platform deployment via TorchScript and ONNX exports
- Comprehensive evaluation metrics

## Project Structure

```
fast_spell/
├── data/
│   ├── clean_corpus.txt
│   ├── real_errors/          # e.g. BEA-60K, GitHub Typo
│   └── synthetic_errors/     # generated
├── src/
│   ├── data_prep.py
│   ├── train.py
│   ├── eval.py
│   └── infer.py
├── models/
├── outputs/
├── requirements.txt
└── environment.yml
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

## Usage

### 1. Data Preparation

```bash
python src/data_prep.py --clean_corpus data/clean_corpus.txt --output_dir data/synthetic_errors
```

### 2. Model Training

```bash
python src/train.py --data_dir data/synthetic_errors --output_dir models
```

### 3. Model Evaluation

```bash
python src/eval.py --data_dir data/synthetic_errors --model_path models/best_model --output_dir outputs
```

### 4. Inference

```bash
# Correct a single text
python src/infer.py --model_path models/best_model --input_text "thsi is a setnence with speliing errros"

# Correct a file
python src/infer.py --model_path models/best_model --input_file input.txt --output_file corrected.txt

# Export to TorchScript for deployment
python src/infer.py --model_path models/best_model --export_torchscript models/spell_traced.pt
```

## Model Optimization

The system supports several optimization techniques:

1. **Dynamic Quantization**: Reduces model size and improves inference speed
2. **Pruning**: Removes less important weights for faster computation
3. **TorchScript Export**: For deployment in production environments
4. **ONNX Export**: For cross-platform deployment

## Performance

The system is designed to achieve:

- Token-level accuracy > 95% on standard benchmarks
- Word Correction Rate (WCR) > 80% on real-world errors
- Inference latency < 50ms per sentence on CPU
- Model size < 300MB after optimization

## License

[MIT License](LICENSE)
