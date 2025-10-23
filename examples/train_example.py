import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from tqdm import tqdm
from aceflow_seq2seq import Seq2Seq, Tokenizer, SequenceDataLoader, save_model, load_model
from aceflow_seq2seq.utils.metrics import calculate_bleu

def create_sample_data(num_samples=1000):
    """Create sample translation-like data"""
    source_texts = []
    target_texts = []
    
    # Simple patterns for demonstration
    patterns = [
        ("hello world", "hola mundo"),
        ("good morning", "buenos dias"),
        ("how are you", "como estas"),
        ("thank you", "gracias"),
        ("I love programming", "me encanta programar"),
        ("machine learning", "aprendizaje automatico"),
        ("neural networks", "redes neuronales"),
        ("deep learning", "aprendizaje profundo"),
    ]
    
    for i in range(num_samples):
        src, tgt = patterns[i % len(patterns)]
        source_texts.append(src)
        target_texts.append(tgt)
    
    return source_texts, target_texts

def main():
    print("AceFlow Seq2Seq Example Training")
    print("=" * 50)
    
    # Create sample data
    print("Creating sample data...")
    source_texts, target_texts = create_sample_data(1000)
    
    # Initialize tokenizers
    print("Building vocabularies...")
    src_tokenizer = Tokenizer(vocab_size=5000)
    tgt_tokenizer = Tokenizer(vocab_size=5000)
    
    src_tokenizer.build_vocab(source_texts)
    tgt_tokenizer.build_vocab(target_texts)
    
    # Encode sequences
    print("Encoding sequences...")
    src_sequences = [src_tokenizer.encode(text) for text in tqdm(source_texts)]
    tgt_sequences = [tgt_tokenizer.encode(text) for text in tqdm(target_texts)]
    
    # Initialize model
    print("Initializing model...")
    model = Seq2Seq(
        src_vocab_size=src_tokenizer.get_vocab_size(),
        tgt_vocab_size=tgt_tokenizer.get_vocab_size(),
        embedding_dim=128,
        hidden_dim=256,
        num_layers=2
    )
    
    # Create data loader
    dataloader = SequenceDataLoader(src_sequences, tgt_sequences, batch_size=32)
    
    # Training loop
    print("Starting training...")
    num_epochs = 10
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        num_batches = 0
        
        with tqdm(total=len(dataloader), desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for batch_src, batch_tgt in dataloader:
                batch_loss = 0
                
                for src_seq, tgt_seq in zip(batch_src, batch_tgt):
                    loss, decoded = model.train_step(src_seq, tgt_seq, optimizer=None)
                    batch_loss += loss
                
                avg_batch_loss = batch_loss / len(batch_src)
                epoch_loss += avg_batch_loss
                num_batches += 1
                
                pbar.set_postfix({'loss': f'{avg_batch_loss:.4f}'})
                pbar.update(1)
        
        avg_epoch_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1} completed. Average Loss: {avg_epoch_loss:.4f}")
        
        # Save model checkpoint
        if (epoch + 1) % 5 == 0:
            checkpoint_path = f"seq2seq_checkpoint_epoch_{epoch+1}.ace"
            save_model(model, checkpoint_path, metadata={
                'epoch': epoch + 1,
                'loss': avg_epoch_loss,
                'src_vocab_size': src_tokenizer.get_vocab_size(),
                'tgt_vocab_size': tgt_tokenizer.get_vocab_size()
            })
            print(f"Checkpoint saved: {checkpoint_path}")
    
    # Final model save
    print("Saving final model...")
    save_model(model, "final_seq2seq_model.ace", metadata={
        'training_completed': True,
        'total_epochs': num_epochs,
        'final_loss': avg_epoch_loss
    })
    
    # Test the model
    print("\nTesting model...")
    test_phrases = [
        "hello world",
        "good morning", 
        "machine learning"
    ]
    
    for phrase in test_phrases:
        encoded = src_tokenizer.encode(phrase)
        decoded_tokens, attention_weights = model.predict(encoded)
        decoded_text = tgt_tokenizer.decode(decoded_tokens)
        
        print(f"Source: '{phrase}' -> Target: '{decoded_text}'")
    
    # Demonstrate loading
    print("\nDemonstrating model loading...")
    loaded_model = load_model("final_seq2seq_model.ace")
    
    # Test loaded model
    test_phrase = "hello world"
    encoded = src_tokenizer.encode(test_phrase)
    decoded_tokens, _ = loaded_model.predict(encoded)
    decoded_text = tgt_tokenizer.decode(decoded_tokens)
    print(f"Loaded model - Source: '{test_phrase}' -> Target: '{decoded_text}'")

if __name__ == "__main__":
    main()