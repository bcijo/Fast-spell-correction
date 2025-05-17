import re

# Read the clean corpus
with open('D:/abhin/Comding/ML/My Research lets say/Spell-correction/data/clean_corpus.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Split into sentences
sentences = re.split(r'[.!?]', text)
sentences = [s.strip() for s in sentences if len(s.strip()) > 20]

# Save the first 1000 sentences
with open('D:/abhin/Comding/ML/My Research lets say/Spell-correction/data/clean_sentences.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(sentences[:1000]))

print(f"Processed {len(sentences[:1000])} sentences")
