"""
Enhanced data preparation script for the fast spell correction system.
This script handles:
1. Clean text corpus preparation
2. Real error dataset processing
3. Advanced synthetic error generation with multiple error types
"""

import os
import random
import argparse
import torch
import numpy as np
from tqdm import tqdm
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
import re
import nltk
from nltk.metrics import edit_distance

# Try to import metaphone, but provide fallback if not available
try:
    from metaphone import doublemetaphone
except ImportError:
    print("Double Metaphone not available. Install with: pip install metaphone")
    doublemetaphone = None

# Try to ensure NLTK punkt tokenizer is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def load_clean_corpus(filepath: str) -> List[str]:
    """
    Load clean text from the given filepath
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(lines)} clean sentences from {filepath}")
    return lines

# CHARACTER-LEVEL NOISE FUNCTIONS

def char_deletion(word: str) -> str:
    """Delete a random character from the word."""
    if len(word) <= 1:
        return word
    pos = random.randint(0, len(word) - 1)
    return word[:pos] + word[pos+1:]

def char_insertion(word: str) -> str:
    """Insert a random character into the word."""
    pos = random.randint(0, len(word))
    char = random.choice('abcdefghijklmnopqrstuvwxyz')
    return word[:pos] + char + word[pos:]

def char_substitution(word: str) -> str:
    """Substitute a random character in the word."""
    if len(word) == 0:
        return word
    pos = random.randint(0, len(word) - 1)
    chars = 'abcdefghijklmnopqrstuvwxyz'.replace(word[pos].lower(), '')
    char = random.choice(chars)
    return word[:pos] + char + word[pos+1:]

def char_swap(word: str) -> str:
    """Swap two adjacent characters in the word."""
    if len(word) <= 1:
        return word
    pos = random.randint(0, len(word) - 2)
    return word[:pos] + word[pos+1] + word[pos] + word[pos+2:]

def char_case_change(word: str) -> str:
    """Change the case of a random character."""
    if len(word) == 0:
        return word
    pos = random.randint(0, len(word) - 1)
    char = word[pos]
    if char.islower():
        new_char = char.upper()
    else:
        new_char = char.lower()
    return word[:pos] + new_char + word[pos+1:]

def char_adjacent_key(word: str) -> str:
    """Replace a character with an adjacent one on the keyboard."""
    if len(word) == 0:
        return word
    
    keyboard_adjacency = {
        'a': 'qwsz', 'b': 'vghn', 'c': 'xdfv', 'd': 'serfcx', 'e': 'wrsdf',
        'f': 'drtgvc', 'g': 'ftyhbv', 'h': 'gyujnb', 'i': 'ujklo', 'j': 'huikmn',
        'k': 'jiolm', 'l': 'kop', 'm': 'njk', 'n': 'bhjm', 'o': 'iklp',
        'p': 'ol', 'q': 'wa', 'r': 'edft', 's': 'awedxz', 't': 'rfgy',
        'u': 'yhji', 'v': 'cfgb', 'w': 'qase', 'x': 'zsdc', 'y': 'tghu',
        'z': 'asx'
    }
    
    pos = random.randint(0, len(word) - 1)
    char = word[pos].lower()
    if char in keyboard_adjacency:
        new_char = random.choice(keyboard_adjacency[char])
        return word[:pos] + new_char + word[pos+1:]
    
    return word

def phonetic_error(word: str) -> str:
    """Generate a phonetically similar error using Double Metaphone."""
    if not doublemetaphone or len(word) <= 3:
        return word
    
    # Get double metaphone representation
    metaphones = doublemetaphone(word)
    
    # Simple phonetic error: remove a character from a position
    # where it might not affect pronunciation much
    vowels = "aeiouy"
    consonants = "bcdfghjklmnpqrstvwxz"
    
    # Simplistic approach: if word has a double letter, remove one
    for i in range(len(word) - 1):
        if word[i] == word[i+1]:
            return word[:i] + word[i+1:]
    
    # If word ends with common suffix, try making an error there
    if word.endswith('ing'):
        return word[:-3] + 'in'
    elif word.endswith('ed'):
        return word[:-2] + 'd'
    elif word.endswith('s') and len(word) > 3:
        return word[:-1]
    
    # Default: just return the word unchanged
    return word

def vowel_error(word: str) -> str:
    """Swap one vowel for another in the word."""
    if len(word) <= 2:
        return word
        
    vowels = "aeiou"
    vowel_positions = [i for i, char in enumerate(word.lower()) if char in vowels]
    
    if not vowel_positions:
        return word
        
    pos = random.choice(vowel_positions)
    other_vowels = vowels.replace(word[pos].lower(), '')
    new_vowel = random.choice(other_vowels)
    
    return word[:pos] + new_vowel + word[pos+1:]

# Main character noise function
def character_noise(word: str, p: float = 0.2) -> str:
    """
    Apply character-level noise to a word with probability p per word
    Operations: deletion, insertion, substitution, swap, case change, 
                adjacent key, phonetic error, vowel error
    """
    if random.random() > p or len(word) <= 1:
        return word
    
    # Weight different error types
    error_funcs = [
        (char_deletion, 0.20),
        (char_insertion, 0.20),
        (char_substitution, 0.15),
        (char_swap, 0.15),
        (char_case_change, 0.05),
        (char_adjacent_key, 0.15),
        (phonetic_error, 0.05),
        (vowel_error, 0.05)
    ]
    
    # Normalize weights
    total_weight = sum(weight for _, weight in error_funcs)
    normalized_weights = [weight/total_weight for _, weight in error_funcs]
    
    # Select a function based on weights
    selected_func = random.choices(
        [func for func, _ in error_funcs],
        weights=normalized_weights,
        k=1
    )[0]
    
    return selected_func(word)

# WORD-LEVEL NOISE FUNCTIONS

def word_deletion(words: List[str]) -> List[str]:
    """Delete a random word."""
    if len(words) <= 1:
        return words
    pos = random.randint(0, len(words) - 1)
    return words[:pos] + words[pos+1:]

def word_repetition(words: List[str]) -> List[str]:
    """Repeat a random word."""
    if not words:
        return words
    pos = random.randint(0, len(words) - 1)
    return words[:pos+1] + [words[pos]] + words[pos+1:]

def word_swap(words: List[str]) -> List[str]:
    """Swap two adjacent words."""
    if len(words) <= 1:
        return words
    pos = random.randint(0, len(words) - 2)
    return words[:pos] + [words[pos+1], words[pos]] + words[pos+2:]

def word_level_noise(words: List[str], p: float = 0.1) -> List[str]:
    """
    Apply word-level noise with probability p
    Operations: deletion, repetition, swap
    """
    if random.random() > p or not words:
        return words
    
    operation = random.choice(['delete', 'repeat', 'swap'])
    
    if operation == 'delete':
        return word_deletion(words)
    elif operation == 'repeat':
        return word_repetition(words)
    elif operation == 'swap':
        return word_swap(words)
    
    return words

def generate_noisy_text(text: str, p_word: float = 0.2, p_sentence: float = 0.1) -> str:
    """
    Generate noisy text by applying character-level and word-level noise
    """
    words = text.split()
    
    # Apply word-level noise
    words = word_level_noise(words, p_sentence)
    
    # Apply character-level noise
    noisy_words = [character_noise(word, p_word) for word in words]
    
    # Random chance to remove spaces between some words (common typing error)
    if random.random() < 0.05 and len(noisy_words) > 1:
        pos = random.randint(0, len(noisy_words) - 2)
        noisy_words[pos] = noisy_words[pos] + noisy_words[pos+1]
        noisy_words.pop(pos+1)
    
    return ' '.join(noisy_words)

def create_synthetic_errors(clean_texts: List[str], 
                           output_dir: str,
                           p_word: float = 0.2,
                           p_sentence: float = 0.1,
                           split_ratio: Dict[str, float] = {'train': 0.8, 'val': 0.1, 'test': 0.1},
                           seed: int = 42):
    """
    Create synthetic errors from clean texts and save to output_dir
    """
    # Set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    
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
                noisy_text = generate_noisy_text(clean_text, p_word, p_sentence)
                
                clean_file.write(clean_text + '\n')
                noisy_file.write(noisy_text + '\n')
        
        print(f"Created {split_name} split with {len(split_indices)} examples")

def prepare_real_errors(input_path: str, output_dir: str, split_ratio: Dict[str, float] = {'train': 0.8, 'val': 0.1, 'test': 0.1}):
    """
    Prepare real spelling error datasets
    """
    print(f"Preparing real errors from {input_path}")
    # Function to be implemented based on the format of real error datasets
    pass

def main():
    parser = argparse.ArgumentParser(description="Data preparation for spell correction")
    parser.add_argument("--clean_corpus", type=str, required=True, 
                        help="Path to clean corpus file")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save synthetic errors")
    parser.add_argument("--p_word", type=float, default=0.2,
                        help="Probability of applying noise to each word")
    parser.add_argument("--p_sentence", type=float, default=0.1,
                        help="Probability of applying word-level noise to each sentence")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Load clean corpus
    clean_texts = load_clean_corpus(args.clean_corpus)
    
    # Create synthetic errors
    create_synthetic_errors(
        clean_texts, 
        args.output_dir, 
        args.p_word,
        args.p_sentence,
        seed=args.seed
    )
    
    print("Data preparation complete!")

if __name__ == "__main__":
    main()
