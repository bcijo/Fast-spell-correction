"""
Data preparation script for the fast spell correction system.
This script handles:
1. Clean text corpus preparation
2. Real error dataset processing
3. Synthetic error generation
"""

import os
import random
import argparse
import torch
from tqdm import tqdm
from typing import List, Tuple, Dict, Optional

def load_clean_corpus(filepath: str) -> List[str]:
    """
    Load clean text from the given filepath
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(lines)} clean sentences from {filepath}")
    return lines

def character_noise(word: str, p: float = 0.2) -> str:
    """
    Apply character-level noise to a word with probability p per word
    Operations: deletion, insertion, substitution, swap
    """
    if random.random() > p or len(word) <= 1:
        return word
    
    operation = random.choice(['delete', 'insert', 'substitute', 'swap'])
    
    if operation == 'delete' and len(word) > 1:
        pos = random.randint(0, len(word) - 1)
        return word[:pos] + word[pos+1:]
    
    elif operation == 'insert':
        pos = random.randint(0, len(word))
        char = random.choice('abcdefghijklmnopqrstuvwxyz')
        return word[:pos] + char + word[pos:]
    
    elif operation == 'substitute' and len(word) > 0:
        pos = random.randint(0, len(word) - 1)
        chars = 'abcdefghijklmnopqrstuvwxyz'.replace(word[pos].lower(), '')
        char = random.choice(chars)
        return word[:pos] + char + word[pos+1:]
    
    elif operation == 'swap' and len(word) > 1:
        pos = random.randint(0, len(word) - 2)
        return word[:pos] + word[pos+1] + word[pos] + word[pos+2:]
    
    return word

def generate_noisy_text(text: str, p_word: float = 0.2) -> str:
    """
    Generate noisy text by applying character-level noise
    """
    words = text.split()
    noisy_words = [character_noise(word, p_word) for word in words]
    return ' '.join(noisy_words)

def create_synthetic_errors(clean_texts: List[str], 
                           output_dir: str,
                           p_word: float = 0.2,
                           split_ratio: Dict[str, float] = {'train': 0.8, 'val': 0.1, 'test': 0.1}):
    """
    Create synthetic errors from clean texts and save to output_dir
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Shuffle the data
    indices = list(range(len(clean_texts)))
    random.shuffle(indices)
    
    # Calculate split sizes
    train_size = int(len(indices) * split_ratio['train'])
    val_size = int(len(indices) * split_ratio['val'])
    
    # Split indices
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size+val_size]
    test_indices = indices[train_size+val_size:]
    
    # Process each split
    for split_name, split_indices in [('train', train_indices), 
                                      ('val', val_indices), 
                                      ('test', test_indices)]:
        clean_path = os.path.join(output_dir, f"{split_name}_clean.txt")
        noisy_path = os.path.join(output_dir, f"{split_name}_noisy.txt")
        
        with open(clean_path, 'w', encoding='utf-8') as clean_file, \
             open(noisy_path, 'w', encoding='utf-8') as noisy_file:
            
            for idx in tqdm(split_indices, desc=f"Creating {split_name} split"):
                clean_text = clean_texts[idx]
                noisy_text = generate_noisy_text(clean_text, p_word)
                
                clean_file.write(clean_text + '\n')
                noisy_file.write(noisy_text + '\n')
        
        print(f"Created {split_name} split with {len(split_indices)} examples")

def main():
    parser = argparse.ArgumentParser(description="Data preparation for spell correction")
    parser.add_argument("--clean_corpus", type=str, required=True, 
                        help="Path to clean corpus file")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save synthetic errors")
    parser.add_argument("--p_word", type=float, default=0.2,
                        help="Probability of applying noise to each word")
    
    args = parser.parse_args()
    
    # Load clean corpus
    clean_texts = load_clean_corpus(args.clean_corpus)
    
    # Create synthetic errors
    create_synthetic_errors(clean_texts, args.output_dir, args.p_word)
    
    print("Data preparation complete!")

if __name__ == "__main__":
    main()
