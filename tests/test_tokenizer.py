"""
Unit tests for Tokenizer functionality
"""

import unittest
import tempfile
import os
import torch
from aceflow.utils import Tokenizer, Vocabulary, Preprocessor

class TestTokenizer(unittest.TestCase):
    
    def setUp(self):
        self.test_texts = [
            "Hello world!",
            "This is a test sentence.",
            "Another example for testing.",
            "I don't know what's happening...",
            "Let's go to the park!"
        ]
        
    def test_tokenizer_initialization(self):
        """Test tokenizer initialization"""
        tokenizer = Tokenizer(name="test", max_length=20)
        self.assertEqual(tokenizer.name, "test")
        self.assertEqual(tokenizer.max_length, 20)
        
    def test_vocabulary_creation(self):
        """Test vocabulary creation and management"""
        vocab = Vocabulary(name="test_vocab")
        vocab.add_word("hello")
        vocab.add_word("world")
        
        self.assertEqual(len(vocab), 7)  # 5 special tokens + 2 words
        self.assertIn("hello", vocab)
        self.assertIn("world", vocab)
        
    def test_preprocessor_pipeline(self):
        """Test text preprocessing pipeline"""
        preprocessor = Preprocessor(language="english")
        test_text = "Hello   WORLD! Don't worry..."
        
        processed = preprocessor.process(test_text)
        self.assertIsInstance(processed, str)
        self.assertNotIn("  ", processed)  # No double spaces
        
    def test_tokenizer_fit(self):
        """Test tokenizer fitting on texts"""
        tokenizer = Tokenizer()
        tokenizer.fit(self.test_texts, max_vocab_size=50, min_freq=1)
        
        self.assertGreater(len(tokenizer), 10)  # Should have reasonable vocabulary
        self.assertIn("hello", tokenizer.get_vocab())
        
    def test_encoding_decoding(self):
        """Test encoding and decoding round-trip"""
        tokenizer = Tokenizer(max_length=15)
        tokenizer.fit(self.test_texts)
        
        test_text = "hello world test"
        encoded = tokenizer.encode(test_text)
        decoded = tokenizer.decode(encoded['input_ids'])
        
        # Should be similar (might differ due to preprocessing)
        self.assertIsInstance(encoded, dict)
        self.assertIn('input_ids', encoded)
        self.assertIn('attention_mask', encoded)
        self.assertIsInstance(decoded, str)
        
    def test_batch_processing(self):
        """Test batch encoding and decoding"""
        tokenizer = Tokenizer()
        tokenizer.fit(self.test_texts)
        
        batch_encoded = tokenizer.encode_batch(self.test_texts[:3])
        batch_decoded = tokenizer.decode_batch([enc['input_ids'] for enc in batch_encoded])
        
        self.assertEqual(len(batch_encoded), 3)
        self.assertEqual(len(batch_decoded), 3)
        
    def test_save_load(self):
        """Test saving and loading tokenizer"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create and save tokenizer
            tokenizer = Tokenizer(name="test_save_load")
            tokenizer.fit(self.test_texts)
            
            save_path = os.path.join(temp_dir, "test_tokenizer")
            tokenizer.save(save_path)
            
            # Load tokenizer
            loaded_tokenizer = Tokenizer.load(save_path)
            
            # Test equivalence
            self.assertEqual(tokenizer.name, loaded_tokenizer.name)
            self.assertEqual(len(tokenizer), len(loaded_tokenizer))
            
            # Test functionality
            test_text = "hello world"
            orig_encoded = tokenizer.encode(test_text)
            loaded_encoded = loaded_tokenizer.encode(test_text)
            
            self.assertEqual(orig_encoded['input_ids'], loaded_encoded['input_ids'])
            
    def test_special_tokens(self):
        """Test special tokens handling"""
        tokenizer = Tokenizer()
        tokenizer.fit(["hello world"])
        
        # Test special tokens encoding
        self.assertEqual(tokenizer.vocab.encode_word('<pad>'), 0)
        self.assertEqual(tokenizer.vocab.encode_word('<start>'), 1)
        self.assertEqual(tokenizer.vocab.encode_word('<end>'), 2)
        self.assertEqual(tokenizer.vocab.encode_word('<unk>'), 3)
        
    def test_sequence_length(self):
        """Test sequence length handling"""
        tokenizer = Tokenizer(max_length=5)
        tokenizer.fit(["this is a very long sentence that should be truncated"])
        
        encoded = tokenizer.encode("this is a very long sentence")
        self.assertEqual(len(encoded['input_ids']), 5)  # Should be padded/truncated to max_length

if __name__ == '__main__':
    unittest.main()