"""
Unit tests for training functionality
"""

import unittest
import tempfile
import os
import torch
from aceflow import Seq2SeqModel
from aceflow.utils import Tokenizer, create_data_loader
from aceflow.trainers import Trainer

class TestTraining(unittest.TestCase):
    
    def setUp(self):
        # Create small dataset
        self.english_sentences = ["hello world", "how are you", "good morning"]
        self.french_sentences = ["bonjour le monde", "comment allez vous", "bonjour"]
        
        # Initialize tokenizers
        self.src_tokenizer = Tokenizer(name="test_src")
        self.tgt_tokenizer = Tokenizer(name="test_tgt")
        self.src_tokenizer.fit(self.english_sentences)
        self.tgt_tokenizer.fit(self.french_sentences)
        
        # Create data loader
        self.train_loader = create_data_loader(
            self.english_sentences, self.french_sentences,
            self.src_tokenizer, self.tgt_tokenizer,
            batch_size=2, max_length=10
        )
        
    def test_trainer_initialization(self):
        """Test trainer initialization"""
        model = Seq2SeqModel(
            src_vocab_size=len(self.src_tokenizer),
            tgt_vocab_size=len(self.tgt_tokenizer),
            hidden_size=64
        )
        
        trainer = Trainer(model, learning_rate=0.001)
        
        self.assertIsInstance(trainer.model, Seq2SeqModel)
        self.assertIsNotNone(trainer.optimizer)
        self.assertIsNotNone(trainer.criterion)
        
    def test_training_epoch(self):
        """Test single training epoch"""
        model = Seq2SeqModel(
            src_vocab_size=len(self.src_tokenizer),
            tgt_vocab_size=len(self.tgt_tokenizer),
            hidden_size=64
        )
        
        trainer = Trainer(model, learning_rate=0.001)
        
        # Test training epoch
        train_loss, train_acc = trainer.train_epoch(self.train_loader)
        
        self.assertIsInstance(train_loss, float)
        self.assertIsInstance(train_acc, float)
        self.assertGreaterEqual(train_acc, 0.0)
        self.assertLessEqual(train_acc, 1.0)
        
    def test_validation_epoch(self):
        """Test single validation epoch"""
        model = Seq2SeqModel(
            src_vocab_size=len(self.src_tokenizer),
            tgt_vocab_size=len(self.tgt_tokenizer),
            hidden_size=64
        )
        
        trainer = Trainer(model, learning_rate=0.001)
        
        # Test validation epoch
        val_loss, val_acc = trainer.validate_epoch(self.train_loader)  # Using train as val for test
        
        self.assertIsInstance(val_loss, float)
        self.assertIsInstance(val_acc, float)
        self.assertGreaterEqual(val_acc, 0.0)
        self.assertLessEqual(val_acc, 1.0)
        
    def test_full_training(self):
        """Test complete training process"""
        with tempfile.TemporaryDirectory() as temp_dir:
            model = Seq2SeqModel(
                src_vocab_size=len(self.src_tokenizer),
                tgt_vocab_size=len(self.tgt_tokenizer),
                hidden_size=64
            )
            
            trainer = Trainer(model, learning_rate=0.001)
            
            save_path = os.path.join(temp_dir, "test_model.ace")
            
            # Train for a few epochs
            history = trainer.train(
                self.train_loader, self.train_loader,
                epochs=3,
                save_path=save_path
            )
            
            # Check history
            self.assertIn('train_loss', history)
            self.assertIn('train_accuracy', history)
            self.assertEqual(len(history['train_loss']), 3)
            
            # Check model was saved
            self.assertTrue(os.path.exists(save_path))
            
    def test_training_history_saving(self):
        """Test training history saving"""
        with tempfile.TemporaryDirectory() as temp_dir:
            model = Seq2SeqModel(
                src_vocab_size=len(self.src_tokenizer),
                tgt_vocab_size=len(self.tgt_tokenizer),
                hidden_size=64
            )
            
            trainer = Trainer(model, learning_rate=0.001)
            
            # Train briefly
            trainer.train(self.train_loader, self.train_loader, epochs=2)
            
            # Save history
            history_path = os.path.join(temp_dir, "history.json")
            trainer.save_training_history(history_path)
            
            # Check file exists
            self.assertTrue(os.path.exists(history_path))
            
    def test_different_rnn_types_training(self):
        """Test training with different RNN types"""
        for rnn_type in ['rnn', 'lstm', 'gru']:
            with self.subTest(rnn_type=rnn_type):
                model = Seq2SeqModel(
                    src_vocab_size=len(self.src_tokenizer),
                    tgt_vocab_size=len(self.tgt_tokenizer),
                    hidden_size=64,
                    rnn_type=rnn_type
                )
                
                trainer = Trainer(model, learning_rate=0.001)
                train_loss, train_acc = trainer.train_epoch(self.train_loader)
                
                self.assertIsInstance(train_loss, float)
                self.assertIsInstance(train_acc, float)
                
    def test_teacher_forcing(self):
        """Test teacher forcing during training"""
        model = Seq2SeqModel(
            src_vocab_size=len(self.src_tokenizer),
            tgt_vocab_size=len(self.tgt_tokenizer),
            hidden_size=64,
            teacher_forcing_ratio=1.0  # Always use teacher forcing
        )
        
        trainer = Trainer(model, learning_rate=0.001)
        
        # Test with different teacher forcing ratios
        for tf_ratio in [0.0, 0.5, 1.0]:
            train_loss, train_acc = trainer.train_epoch(
                self.train_loader, teacher_forcing_ratio=tf_ratio
            )
            
            self.assertIsInstance(train_loss, float)
            self.assertIsInstance(train_acc, float)

if __name__ == '__main__':
    unittest.main()