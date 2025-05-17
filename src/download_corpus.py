"""
Script to download and preprocess large text corpora for spell correction training.
Includes Wikipedia, OpenWebText, and BookCorpus samples.
"""
import os
import re
import nltk
import random
import datasets
from tqdm import tqdm
from pathlib import Path

# Ensure necessary NLTK data is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Define paths
DATA_DIR = Path("../data")
CLEAN_CORPUS_DIR = DATA_DIR / "clean_corpus"
CLEAN_CORPUS_DIR.mkdir(exist_ok=True, parents=True)

def download_and_process_wiki():
    """Download a sample of Wikipedia articles and process them."""
    print("Downloading Wikipedia sample...")
    wiki_dataset = datasets.load_dataset("wikipedia", "20220301.en", split="train[:10000]")
    
    with open(CLEAN_CORPUS_DIR / "wikipedia_sample.txt", "w", encoding="utf-8") as f:
        for article in tqdm(wiki_dataset, desc="Processing Wikipedia"):
            # Extract text and perform basic cleaning
            text = article["text"]
            # Basic cleaning: remove multiple spaces, weird characters, etc.
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r'[^\w\s.,!?;:\'\"-]', '', text)
            
            # Split into sentences
            sentences = nltk.sent_tokenize(text)
            
            # Write clean sentences to file
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence.split()) > 5:  # Only keep sentences with at least 5 words
                    f.write(sentence + "\n")
    
    print(f"Wikipedia sample saved to {CLEAN_CORPUS_DIR / 'wikipedia_sample.txt'}")

def download_and_process_openwebtext():
    """Download a sample of OpenWebText and process it."""
    print("Downloading OpenWebText sample...")
    openwebtext_dataset = datasets.load_dataset("openwebtext", split="train[:5000]")
    
    with open(CLEAN_CORPUS_DIR / "openwebtext_sample.txt", "w", encoding="utf-8") as f:
        for item in tqdm(openwebtext_dataset, desc="Processing OpenWebText"):
            text = item["text"]
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r'[^\w\s.,!?;:\'\"-]', '', text)
            
            sentences = nltk.sent_tokenize(text)
            
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence.split()) > 5:
                    f.write(sentence + "\n")
    
    print(f"OpenWebText sample saved to {CLEAN_CORPUS_DIR / 'openwebtext_sample.txt'}")

def download_and_process_bookcorpus():
    """Download a sample of BookCorpus and process it."""
    print("Downloading BookCorpus sample...")
    bookcorpus_dataset = datasets.load_dataset("bookcorpus", split="train[:5000]")
    
    with open(CLEAN_CORPUS_DIR / "bookcorpus_sample.txt", "w", encoding="utf-8") as f:
        for item in tqdm(bookcorpus_dataset, desc="Processing BookCorpus"):
            text = item["text"]
            text = re.sub(r'\s+', ' ', text)
            text = re.sub(r'[^\w\s.,!?;:\'\"-]', '', text)
            
            sentences = nltk.sent_tokenize(text)
            
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence.split()) > 5:
                    f.write(sentence + "\n")
    
    print(f"BookCorpus sample saved to {CLEAN_CORPUS_DIR / 'bookcorpus_sample.txt'}")

def download_bea_error_dataset():
    """Download and prepare the BEA-60K spelling error dataset."""
    print("Downloading BEA-60K dataset...")
    # Since BEA-60K isn't directly available in HuggingFace datasets,
    # we'll use a placeholder approach and download programmatically
    
    BEA_DIR = DATA_DIR / "real_errors" / "bea60k"
    BEA_DIR.mkdir(exist_ok=True, parents=True)
    
    # This would normally contain code to download from the actual source
    # For now, create a placeholder info file
    with open(BEA_DIR / "download_info.txt", "w", encoding="utf-8") as f:
        f.write("The BEA-60K dataset should be downloaded from: https://www.cl.cam.ac.uk/research/nl/bea2019st/\n")
        f.write("After downloading, place the files in this directory.\n")
    
    print(f"BEA-60K download information saved to {BEA_DIR / 'download_info.txt'}")

def combine_corpora():
    """Combine all downloaded corpora into a single large corpus file."""
    all_sentences = []
    
    # Read all corpora
    for corpus_file in CLEAN_CORPUS_DIR.glob("*.txt"):
        with open(corpus_file, "r", encoding="utf-8") as f:
            sentences = [line.strip() for line in f if line.strip()]
            all_sentences.extend(sentences)
    
    # Shuffle and save
    random.shuffle(all_sentences)
    
    with open(DATA_DIR / "combined_clean_corpus.txt", "w", encoding="utf-8") as f:
        for sentence in all_sentences:
            f.write(sentence + "\n")
    
    print(f"Combined corpus saved to {DATA_DIR / 'combined_clean_corpus.txt'}")
    print(f"Total sentences: {len(all_sentences)}")

if __name__ == "__main__":
    print("Starting corpus download and processing...")
    
    # Download and process each corpus
    download_and_process_wiki()
    download_and_process_openwebtext()
    download_and_process_bookcorpus()
    download_bea_error_dataset()
    
    # Combine all corpora
    combine_corpora()
    
    print("Corpus download and processing complete!")
