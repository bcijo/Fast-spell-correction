"""
Unit tests for the spell correction functionality.
"""

import os
import sys
import unittest
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

# Add the src directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_prep import character_noise, generate_noisy_text
from src.infer import SpellCorrector

class TestDataPrep(unittest.TestCase):
    """Tests for data preparation functions"""
    
    def test_character_noise(self):
        """Test character noise generation"""
        # Set seed for reproducibility
        torch.manual_seed(42)
        
        word = "hello"
        
        # Test with p=0 (no noise)
        self.assertEqual(character_noise(word, p=0), "hello")
        
        # Test with p=1 (always add noise)
        noisy_word = character_noise(word, p=1)
        self.assertNotEqual(noisy_word, word)
        
        # Test that Levenshtein distance is small
        import Levenshtein
        self.assertLessEqual(Levenshtein.distance(word, noisy_word), 2)
    
    def test_generate_noisy_text(self):
        """Test noisy text generation"""
        # Set seed for reproducibility
        torch.manual_seed(42)
        
        text = "this is a test sentence for spell checking"
        
        # Test with p=0 (no noise)
        self.assertEqual(generate_noisy_text(text, p_word=0), text)
        
        # Test with p=1 (always add noise)
        noisy_text = generate_noisy_text(text, p_word=1)
        self.assertNotEqual(noisy_text, text)
        
        # Check that we have the same number of words
        self.assertEqual(len(noisy_text.split()), len(text.split()))

class MockModel:
    """Mock model for testing SpellCorrector"""
    
    def __init__(self):
        self.eval_called = False
    
    def eval(self):
        self.eval_called = True
    
    def to(self, device):
        self.device = device
        return self
    
    def __call__(self, **kwargs):
        # Return a mock output with logits
        class MockOutput:
            def __init__(self):
                # Create some dummy logits
                batch_size = kwargs.get('input_ids', torch.tensor([[0]])).shape[0]
                seq_len = kwargs.get('input_ids', torch.tensor([[0]])).shape[1]
                vocab_size = 100
                self.logits = torch.zeros((batch_size, seq_len, vocab_size))
                # Make the first token predict "the"
                self.logits[0, 0, 50] = 10.0
        
        return MockOutput()

class MockTokenizer:
    """Mock tokenizer for testing SpellCorrector"""
    
    def __init__(self):
        self.mask_token = "[MASK]"
        self.mask_token_id = 103
    
    def get_vocab(self):
        return {"hello": 0, "world": 1, "the": 50}
    
    def tokenize(self, text):
        return text.split()
    
    def convert_tokens_to_string(self, tokens):
        return " ".join(tokens)
    
    def convert_tokens_to_ids(self, tokens):
        vocab = self.get_vocab()
        return [vocab.get(t, 2) for t in tokens]  # 2 is a dummy OOV ID
    
    def convert_ids_to_tokens(self, ids):
        if isinstance(ids, int):
            ids = [ids]
        vocab_inv = {v: k for k, v in self.get_vocab().items()}
        return [vocab_inv.get(i, "unk") for i in ids]
    
    def __call__(self, text, **kwargs):
        if isinstance(text, str):
            tokens = self.tokenize(text)
            input_ids = torch.tensor([self.convert_tokens_to_ids(tokens)])
            return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}
        return text

class TestSpellCorrector(unittest.TestCase):
    """Tests for SpellCorrector class"""
    
    def setUp(self):
        """Set up test environment"""
        # Create a mock model and tokenizer
        self.mock_model = MockModel()
        self.mock_tokenizer = MockTokenizer()
        
        # Patch AutoModelForMaskedLM and AutoTokenizer
        self._orig_auto_model = AutoModelForMaskedLM.from_pretrained
        self._orig_auto_tokenizer = AutoTokenizer.from_pretrained
        
        AutoModelForMaskedLM.from_pretrained = lambda *args, **kwargs: self.mock_model
        AutoTokenizer.from_pretrained = lambda *args, **kwargs: self.mock_tokenizer
    
    def tearDown(self):
        """Clean up after tests"""
        # Restore original methods
        AutoModelForMaskedLM.from_pretrained = self._orig_auto_model
        AutoTokenizer.from_pretrained = self._orig_auto_tokenizer
    
    def test_init(self):
        """Test SpellCorrector initialization"""
        corrector = SpellCorrector("dummy_path")
        
        # Check that the model is set to eval mode
        self.assertTrue(self.mock_model.eval_called)
        
        # Check that vocabulary is built
        self.assertEqual(corrector.vocab, set(self.mock_tokenizer.get_vocab().keys()))
    
    def test_is_suspect(self):
        """Test suspect token detection"""
        corrector = SpellCorrector("dummy_path")
        
        # Known token should not be suspect
        self.assertFalse(corrector.is_suspect("hello"))
        
        # Unknown token should be suspect
        self.assertTrue(corrector.is_suspect("helloo"))
        
        # Very short token should not be suspect even if unknown
        self.assertFalse(corrector.is_suspect("a"))

# Run the tests
if __name__ == "__main__":
    unittest.main()
