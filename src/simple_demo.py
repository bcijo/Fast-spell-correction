import torch
from torch import nn
import os
from torch.utils.data import Dataset, DataLoader
import random
from tqdm import tqdm

# Simple custom model for demonstration purposes
class SimpleSpellCorrector(nn.Module):
    def __init__(self, vocab_size=50000, embedding_dim=128, hidden_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)
        
    def forward(self, x):
        # x has shape [batch_size, seq_len]
        embedded = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        lstm_out, _ = self.lstm(embedded)  # [batch_size, seq_len, hidden_dim*2]
        logits = self.fc(lstm_out)  # [batch_size, seq_len, vocab_size]
        return logits

# Create a simple tokenizer function
def simple_tokenize(text):
    return list(text.lower())

# Dictionary of characters to indices for our simple model
CHARS = "abcdefghijklmnopqrstuvwxyz0123456789 .,!?"
char_to_idx = {c: i for i, c in enumerate(CHARS)}
idx_to_char = {i: c for i, c in enumerate(CHARS)}
vocab_size = len(CHARS)

# Function to convert text to indices
def text_to_indices(text):
    return [char_to_idx.get(c, len(CHARS)-1) for c in simple_tokenize(text)]

# Function to add random noise to text
def add_noise(text, p=0.2):
    result = []
    for c in text:
        if c in char_to_idx and random.random() < p:
            operation = random.choice(['delete', 'insert', 'substitute', 'swap'])
            
            if operation == 'delete':
                continue  # Skip this character
            
            elif operation == 'insert':
                result.append(c)
                result.append(random.choice(CHARS[:-5]))  # Exclude special tokens
                
            elif operation == 'substitute':
                result.append(random.choice(CHARS[:-5]))  # Exclude special tokens
                
            elif operation == 'swap' and len(result) > 0:
                prev = result.pop()
                result.append(c)
                result.append(prev)
        else:
            result.append(c)
            
    return ''.join(result)

# Create a dataset class
class SpellCorrectionDataset(Dataset):
    def __init__(self, clean_texts, p_noise=0.2, max_len=50):
        self.clean_texts = clean_texts
        self.p_noise = p_noise
        self.max_len = max_len
        
    def __len__(self):
        return len(self.clean_texts)
    
    def __getitem__(self, idx):
        clean_text = self.clean_texts[idx]
        noisy_text = add_noise(clean_text, self.p_noise)
        
        clean_indices = text_to_indices(clean_text)
        noisy_indices = text_to_indices(noisy_text)
        
        # Pad or truncate sequences to max_len
        if len(clean_indices) > self.max_len:
            clean_indices = clean_indices[:self.max_len]
        else:
            clean_indices = clean_indices + [0] * (self.max_len - len(clean_indices))
            
        if len(noisy_indices) > self.max_len:
            noisy_indices = noisy_indices[:self.max_len]
        else:
            noisy_indices = noisy_indices + [0] * (self.max_len - len(noisy_indices))
        
        return {
            'clean_text': clean_text,
            'noisy_text': noisy_text,
            'clean_indices': torch.tensor(clean_indices, dtype=torch.long),
            'noisy_indices': torch.tensor(noisy_indices, dtype=torch.long)
        }

# Sample demo data
sample_texts = [
    "hello world",
    "this is a simple example",
    "spell correction is important",
    "natural language processing",
    "machine learning is fun"
]

# Create a dataset and dataloader
dataset = SpellCorrectionDataset(sample_texts)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Initialize model
model = SimpleSpellCorrector(vocab_size=vocab_size)
print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

# Display a sample
sample = next(iter(dataloader))
print(f"Clean text: {sample['clean_text'][0]}")
print(f"Noisy text: {sample['noisy_text'][0]}")

# Show the model can process input
with torch.no_grad():
    logits = model(sample['noisy_indices'])
    print(f"Output shape: {logits.shape}")

print("Demo completed successfully!")
