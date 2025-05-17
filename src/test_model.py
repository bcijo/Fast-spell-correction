import torch
import os
from transformers import AutoTokenizer, BertForMaskedLM

# Disable Tensorflow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize model and tokenizer
model_name = "prajjwal1/bert-tiny"  # Smaller model that should load faster
print(f"Loading model: {model_name}")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = BertForMaskedLM.from_pretrained(model_name)

print(f"Model loaded successfully. Model size: {sum(p.numel() for p in model.parameters())} parameters")

# Moving to device (cpu/gpu)
model.to(device)

# Sample test
text = "This is a smiple test with a spellnig error."
inputs = tokenizer(text, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model(**inputs)

print("Test inference successful!")

# Try a masked prediction
masked_text = "This is a [MASK] test."
inputs = tokenizer(masked_text, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model(**inputs)

mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
logits = outputs.logits
mask_token_logits = logits[0, mask_token_index, :]
top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()

for token in top_5_tokens:
    print(f"Predicted word: {tokenizer.decode([token])}")
