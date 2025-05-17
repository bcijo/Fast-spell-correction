"""
Model downloader and adapter for the fast spell correction system.
Downloads and adapts various lightweight transformer models.
"""

import os
import torch
import argparse
from tqdm import tqdm
from pathlib import Path
from transformers import (
    AutoModelForMaskedLM, 
    AutoTokenizer,
    BertForMaskedLM,
    AlbertForMaskedLM,
    DistilBertForMaskedLM
)

class SpellCorrectionModel:
    """Base class for spell correction models"""
    def __init__(self, model_name_or_path, device=None):
        self.model_name = model_name_or_path
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        
    def load(self):
        """Load the model and tokenizer"""
        raise NotImplementedError("Subclasses must implement load()")
    
    def save(self, output_dir):
        """Save the model and tokenizer"""
        if not self.model or not self.tokenizer:
            raise ValueError("Model and tokenizer must be loaded before saving")
        
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"Model and tokenizer saved to {output_dir}")
    
    def correct(self, text):
        """Correct spelling errors in text"""
        raise NotImplementedError("Subclasses must implement correct()")

class BertSpellCorrector(SpellCorrectionModel):
    """BERT-based spell corrector"""
    def load(self):
        """Load the BERT model and tokenizer"""
        print(f"Loading BERT model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode
        return self

class TinyBertSpellCorrector(SpellCorrectionModel):
    """TinyBERT-based spell corrector"""
    def load(self):
        """Load the TinyBERT model and tokenizer"""
        print(f"Loading TinyBERT model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        return self

class MobileBertSpellCorrector(SpellCorrectionModel):
    """MobileBERT-based spell corrector"""
    def load(self):
        """Load the MobileBERT model and tokenizer"""
        print(f"Loading MobileBERT model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        return self

class AlbertSpellCorrector(SpellCorrectionModel):
    """ALBERT-based spell corrector"""
    def load(self):
        """Load the ALBERT model and tokenizer"""
        print(f"Loading ALBERT model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        return self

def get_model_class(model_name):
    """Get the appropriate model class based on the model name"""
    model_name_lower = model_name.lower()
    
    if 'bert-tiny' in model_name_lower or 'tinybert' in model_name_lower:
        return TinyBertSpellCorrector
    elif 'mobilebert' in model_name_lower:
        return MobileBertSpellCorrector
    elif 'albert' in model_name_lower:
        return AlbertSpellCorrector
    elif 'distilbert' in model_name_lower or 'bert' in model_name_lower:
        return BertSpellCorrector
    else:
        # Default to BERT
        return BertSpellCorrector

def download_model(model_name, output_dir=None):
    """
    Download the specified model and tokenizer
    
    Args:
        model_name: Name of the model from Hugging Face model hub
        output_dir: Directory to save the model (optional)
    
    Returns:
        Model instance if output_dir is None, else None
    """
    model_class = get_model_class(model_name)
    model = model_class(model_name)
    model.load()
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        model.save(output_dir)
        return None
    else:
        return model

def list_available_models():
    """List available pre-trained models for spell correction"""
    models = [
        {
            "name": "distilbert-base-uncased",
            "description": "DistilBERT model (66M parameters)",
            "size": "265MB"
        },
        {
            "name": "prajjwal1/bert-tiny",
            "description": "Tiny BERT model (4.4M parameters)",
            "size": "17MB"
        },
        {
            "name": "google/mobilebert-uncased",
            "description": "MobileBERT model (25M parameters)",
            "size": "95MB"
        },
        {
            "name": "albert-base-v2",
            "description": "ALBERT base model (12M parameters)",
            "size": "45MB"
        }
    ]
    
    print("\nAvailable pre-trained models for spell correction:")
    print("-" * 80)
    print(f"{'Model Name':<30} {'Description':<30} {'Size':<10}")
    print("-" * 80)
    
    for model in models:
        print(f"{model['name']:<30} {model['description']:<30} {model['size']:<10}")
    
    print("\nUse the --model argument with one of the above model names.")

def main():
    parser = argparse.ArgumentParser(description="Download and adapt transformer models for spell correction")
    parser.add_argument("--model", type=str, 
                        help="Model name from Hugging Face (e.g., distilbert-base-uncased)")
    parser.add_argument("--output_dir", type=str, 
                        help="Directory to save the model")
    parser.add_argument("--list", action="store_true",
                        help="List available pre-trained models")
    
    args = parser.parse_args()
    
    if args.list:
        list_available_models()
        return
    
    if not args.model:
        parser.error("Please specify a model name with --model or use --list to see available models")
    
    # Download the model
    download_model(args.model, args.output_dir)
    print(f"Model {args.model} downloaded and saved to {args.output_dir}")

if __name__ == "__main__":
    main()
