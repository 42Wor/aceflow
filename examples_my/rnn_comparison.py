import torch
from aceflow import Seq2SeqModel
from aceflow.utils import Tokenizer, create_data_loader
from aceflow.trainers import Trainer

def compare_rnn_types():
    """Compare different RNN types for the same task"""
    
    # Sample data
    english_sentences = ["hello world", "how are you", "good morning", "what is your name"]
    french_sentences = ["bonjour le monde", "comment allez vous", "bonjour", "quel est votre nom"]
    
    # Initialize tokenizers
    src_tokenizer = Tokenizer(name="english")
    tgt_tokenizer = Tokenizer(name="french")
    src_tokenizer.fit(english_sentences)
    tgt_tokenizer.fit(french_sentences)
    
    # RNN types to compare
    rnn_types = ['rnn', 'lstm', 'gru', 'bilstm', 'bigru']
    
    results = {}
    
    for rnn_type in rnn_types:
        print(f"\n=== Training with {rnn_type.upper()} ===")
        
        # Create model with specific RNN type
        model = Seq2SeqModel(
            src_vocab_size=len(src_tokenizer),
            tgt_vocab_size=len(tgt_tokenizer),
            hidden_size=128,
            num_layers=2,
            rnn_type=rnn_type,
            use_attention=True,
            bidirectional=rnn_type.startswith('bi')  # Auto-set bidirectional
        )
        
        # Get model info
        info = model.get_rnn_info()
        print(f"Model info: {info}")
        
        # Create data loader
        train_loader = create_data_loader(
            english_sentences, french_sentences, 
            src_tokenizer, tgt_tokenizer, 
            batch_size=2, max_length=10
        )
        
        # Train briefly
        trainer = Trainer(model, learning_rate=0.001)
        history = trainer.train(
            train_loader, train_loader, 
            epochs=5,  # Short training for comparison
            save_path=f"models/model_{rnn_type}.ace"
        )
        
        # Store results
        results[rnn_type] = {
            'final_loss': history['train_loss'][-1],
            'parameters': info['total_parameters'],
            'config': info
        }
    
    # Print comparison
    print("\n=== RNN Type Comparison ===")
    for rnn_type, result in results.items():
        print(f"{rnn_type.upper():8} | Loss: {result['final_loss']:.4f} | Params: {result['parameters']:>7,}")

if __name__ == "__main__":
    compare_rnn_types()