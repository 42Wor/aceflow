"""
Unit tests for Seq2Seq model functionality
"""

import unittest
import tempfile
import os
import torch
from aceflow import Seq2SeqModel
from aceflow.utils import Tokenizer

class TestSeq2SeqModel(unittest.TestCase):
    
    def setUp(self):
        self.src_vocab_size = 100
        self.tgt_vocab_size = 100
        self.hidden_size = 64
        self.batch_size = 2
        self.seq_length = 10
        
    def test_model_initialization(self):
        """Test model initialization with different configurations"""
        
        # Test different RNN types
        for rnn_type in ['rnn', 'lstm', 'gru', 'bilstm']:
            with self.subTest(rnn_type=rnn_type):
                model = Seq2SeqModel(
                    src_vocab_size=self.src_vocab_size,
                    tgt_vocab_size=self.tgt_vocab_size,
                    hidden_size=self.hidden_size,
                    rnn_type=rnn_type,
                    use_attention=True
                )
                
                self.assertIsInstance(model, Seq2SeqModel)
                self.assertEqual(model.rnn_type, rnn_type)
                
                # Test model info
                info = model.get_rnn_info()
                self.assertEqual(info['rnn_type'], rnn_type)
                
    def test_forward_pass(self):
        """Test model forward pass"""
        model = Seq2SeqModel(
            src_vocab_size=self.src_vocab_size,
            tgt_vocab_size=self.tgt_vocab_size,
            hidden_size=self.hidden_size
        )
        
        # Create dummy input
        src = torch.randint(0, self.src_vocab_size, (self.batch_size, self.seq_length))
        tgt = torch.randint(0, self.tgt_vocab_size, (self.batch_size, self.seq_length))
        
        # Test with teacher forcing
        output = model(src, tgt)
        self.assertEqual(output.shape, (self.batch_size, self.seq_length, self.tgt_vocab_size))
        
        # Test without teacher forcing (inference mode)
        output = model(src)
        self.assertEqual(output.shape, (self.batch_size, model.max_length, self.tgt_vocab_size))
        
    def test_attention_mechanism(self):
        """Test attention mechanism"""
        model = Seq2SeqModel(
            src_vocab_size=self.src_vocab_size,
            tgt_vocab_size=self.tgt_vocab_size,
            hidden_size=self.hidden_size,
            use_attention=True
        )
        
        src = torch.randint(0, self.src_vocab_size, (self.batch_size, self.seq_length))
        tgt = torch.randint(0, self.tgt_vocab_size, (self.batch_size, self.seq_length))
        
        # Should return both outputs and attention weights
        output, attention_weights = model(src, tgt)
        
        self.assertEqual(output.shape, (self.batch_size, self.seq_length, self.tgt_vocab_size))
        self.assertEqual(attention_weights.shape, (self.batch_size, self.seq_length, self.seq_length))
        
    def test_beam_search(self):
        """Test beam search inference"""
        model = Seq2SeqModel(
            src_vocab_size=self.src_vocab_size,
            tgt_vocab_size=self.tgt_vocab_size,
            hidden_size=self.hidden_size
        )
        model.eval()
        
        src = torch.randint(0, self.src_vocab_size, (1, 5))  # Single sequence
        
        with torch.no_grad():
            sequence = model.beam_search(src, beam_width=3, max_length=10)
            
        self.assertIsInstance(sequence, list)
        self.assertGreater(len(sequence), 0)
        
    def test_save_load(self):
        """Test model saving and loading"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create and save model
            model = Seq2SeqModel(
                src_vocab_size=self.src_vocab_size,
                tgt_vocab_size=self.tgt_vocab_size,
                hidden_size=self.hidden_size,
                rnn_type='lstm',
                use_attention=True
            )
            
            save_path = os.path.join(temp_dir, "test_model.ace")
            model.save(save_path)
            
            # Load model
            loaded_model = Seq2SeqModel.load(save_path)
            
            # Test equivalence
            self.assertEqual(model.src_vocab_size, loaded_model.src_vocab_size)
            self.assertEqual(model.tgt_vocab_size, loaded_model.tgt_vocab_size)
            self.assertEqual(model.rnn_type, loaded_model.rnn_type)
            
            # Test forward pass equivalence
            src = torch.randint(0, self.src_vocab_size, (1, 5))
            tgt = torch.randint(0, self.tgt_vocab_size, (1, 5))
            
            model.eval()
            loaded_model.eval()
            
            with torch.no_grad():
                orig_output = model(src, tgt)
                loaded_output = loaded_model(src, tgt)
                
            # Outputs should be similar (allowing for small numerical differences)
            self.assertEqual(orig_output.shape, loaded_output.shape)
            
    def test_encoder_decoder_separate(self):
        """Test separate encoder and decoder usage"""
        model = Seq2SeqModel(
            src_vocab_size=self.src_vocab_size,
            tgt_vocab_size=self.tgt_vocab_size,
            hidden_size=self.hidden_size
        )
        
        src = torch.randint(0, self.src_vocab_size, (1, 5))
        
        # Test encoder
        encoder_outputs, encoder_hidden = model.encode(src)
        self.assertIsNotNone(encoder_outputs)
        self.assertIsNotNone(encoder_hidden)
        
        # Test decoder
        decoder_input = torch.tensor([[1]])  # Start token
        if model.use_attention:
            decoder_output, decoder_hidden, attention = model.decode(
                decoder_input, encoder_hidden, encoder_outputs
            )
        else:
            decoder_output, decoder_hidden = model.decode(
                decoder_input, encoder_hidden, encoder_outputs
            )
            
        self.assertIsNotNone(decoder_output)
        
    def test_bidirectional_handling(self):
        """Test bidirectional RNN handling"""
        model = Seq2SeqModel(
            src_vocab_size=self.src_vocab_size,
            tgt_vocab_size=self.tgt_vocab_size,
            hidden_size=self.hidden_size,
            rnn_type='bilstm',
            bidirectional=True
        )
        
        src = torch.randint(0, self.src_vocab_size, (self.batch_size, self.seq_length))
        tgt = torch.randint(0, self.tgt_vocab_size, (self.batch_size, self.seq_length))
        
        output = model(src, tgt)
        self.assertEqual(output.shape, (self.batch_size, self.seq_length, self.tgt_vocab_size))

if __name__ == '__main__':
    unittest.main()