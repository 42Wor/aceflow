"""
Advanced translation example with multiple RNN types and attention mechanisms
"""

import torch
import json
import time
from aceflow import Seq2SeqModel
from aceflow.utils import Tokenizer, create_data_loader
from aceflow.trainers import Trainer

def load_dataset():
    """Load a larger dataset for training"""
    # Expanded dataset
    english_sentences = [
        "hello world", "how are you", "good morning", "what is your name",
        "i love programming", "the weather is nice", "see you later",
        "thank you", "have a nice day", "where is the station",
        "my name is john", "how old are you", "where are you from",
        "what time is it", "i am hungry", "let us go to the park",
        "this is amazing", "i do not understand", "can you help me",
        "what do you think", "i like to read books", "the cat is sleeping",
        "we are learning machine learning", "python is a great language",
        "deep learning is fascinating", "have a good weekend"
    ]

    french_sentences = [
        "bonjour le monde", "comment allez vous", "bonjour", "quel est votre nom",
        "j aime la programmation", "le temps est agreable", "a plus tard",
        "merci", "passez une bonne journee", "ou est la gare",
        "je m appelle jean", "quel age avez vous", "d ou venez vous",
        "quelle heure est il", "j ai faim", "allons au parc",
        "c est incroyable", "je ne comprends pas", "pouvez vous m aider",
        "qu en pensez vous", "j aime lire des livres", "le chat dort",
        "nous apprenons l apprentissage automatique", "python est un excellent langage",
        "l apprentissage profond est fascinant", "bon week end"
    ]
    
    return english_sentences, french_sentences

def create_advanced_model(config):
    """Create model with advanced configuration"""
    return Seq2SeqModel(
        src_vocab_size=config['src_vocab_size'],
        tgt_vocab_size=config['tgt_vocab_size'],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        rnn_type=config['rnn_type'],
        use_attention=config['use_attention'],
        bidirectional=config['bidirectional'],
        attention_method=config['attention_method'],
        dropout=config['dropout'],
        teacher_forcing_ratio=config['teacher_forcing_ratio']
    )

def main():
    print("🚀 AceFlow Advanced Translation Example")
    print("=" * 50)
    
    # Load dataset
    english_sentences, french_sentences = load_dataset()
    
    print(f"📊 Dataset Size: {len(english_sentences)} sentence pairs")
    
    # Split data (80% train, 20% validation)
    split_idx = int(0.8 * len(english_sentences))
    train_eng, val_eng = english_sentences[:split_idx], english_sentences[split_idx:]
    train_fr, val_fr = french_sentences[:split_idx], french_sentences[split_idx:]
    
    print(f"📈 Training samples: {len(train_eng)}")
    print(f"📊 Validation samples: {len(val_eng)}")

    # Initialize advanced tokenizers
    print("\n🔤 Initializing Advanced Tokenizers...")
    src_tokenizer = Tokenizer(
        name="english_advanced",
        language="english",
        max_length=20,
        padding="post",
        truncation="post"
    )
    
    tgt_tokenizer = Tokenizer(
        name="french_advanced", 
        language="french",
        max_length=20,
        padding="post",
        truncation="post"
    )

    # Fit tokenizers on full dataset
    src_tokenizer.fit(english_sentences, max_vocab_size=2000, min_freq=1)
    tgt_tokenizer.fit(french_sentences, max_vocab_size=2000, min_freq=1)
    
    print(f"✅ Source vocabulary size: {len(src_tokenizer)}")
    print(f"✅ Target vocabulary size: {len(tgt_tokenizer)}")

    # Create data loaders
    train_loader = create_data_loader(
        train_eng, train_fr, 
        src_tokenizer, tgt_tokenizer, 
        batch_size=4, max_length=20
    )
    
    val_loader = create_data_loader(
        val_eng, val_fr,
        src_tokenizer, tgt_tokenizer,
        batch_size=4, max_length=20
    )

    # Model configurations to try
    model_configs = [
        {
            'name': 'bilstm_concat_attention',
            'rnn_type': 'bilstm',
            'hidden_size': 256,
            'num_layers': 2,
            'use_attention': True,
            'bidirectional': True,
            'attention_method': 'concat',
            'dropout': 0.2,
            'teacher_forcing_ratio': 0.5
        },
        {
            'name': 'gru_general_attention', 
            'rnn_type': 'gru',
            'hidden_size': 256,
            'num_layers': 2,
            'use_attention': True,
            'bidirectional': False,
            'attention_method': 'general',
            'dropout': 0.2,
            'teacher_forcing_ratio': 0.5
        }
    ]

    results = {}
    
    for config in model_configs:
        print(f"\n🧪 Training Model: {config['name']}")
        print("-" * 40)
        
        # Update config with vocab sizes
        config.update({
            'src_vocab_size': len(src_tokenizer),
            'tgt_vocab_size': len(tgt_tokenizer)
        })
        
        # Create model
        model = create_advanced_model(config)
        model_info = model.get_rnn_info()
        
        print(f"📋 Model Configuration:")
        print(f"  RNN Type: {model_info['rnn_type'].upper()}")
        print(f"  Hidden Size: {model_info['hidden_size']}")
        print(f"  Layers: {model_info['num_layers']}")
        print(f"  Bidirectional: {model_info['bidirectional']}")
        print(f"  Attention: {model_info['has_attention']}")
        print(f"  Parameters: {model_info['total_parameters']:,}")
        
        # Train model
        trainer = Trainer(model, learning_rate=0.001)
        
        start_time = time.time()
        history = trainer.train(
            train_loader, val_loader,
            epochs=30,
            save_path=f"models/{config['name']}.ace",
            teacher_forcing_ratio=config['teacher_forcing_ratio'],
            eval_every=3
        )
        training_time = time.time() - start_time
        
        # Store results
        final_train_loss = history['train_loss'][-1]
        final_val_loss = history['val_loss'][-1] if history['val_loss'] else final_train_loss
        
        results[config['name']] = {
            'final_train_loss': final_train_loss,
            'final_val_loss': final_val_loss,
            'training_time': training_time,
            'parameters': model_info['total_parameters'],
            'config': config
        }
        
        print(f"✅ Training completed in {training_time:.2f} seconds")
        print(f"📊 Final Train Loss: {final_train_loss:.4f}")
        print(f"📊 Final Val Loss: {final_val_loss:.4f}")

    # Compare results
    print("\n📊 Model Comparison Results:")
    print("=" * 50)
    for model_name, result in results.items():
        print(f"\n🔍 {model_name}:")
        print(f"  Final Train Loss: {result['final_train_loss']:.4f}")
        print(f"  Final Val Loss: {result['final_val_loss']:.4f}")
        print(f"  Training Time: {result['training_time']:.2f}s")
        print(f"  Parameters: {result['parameters']:,}")

    # Save tokenizers
    src_tokenizer.save("tokenizers/english_advanced")
    tgt_tokenizer.save("tokenizers/french_advanced")
    
    # Save comparison results
    with open("logs/model_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to: logs/model_comparison.json")
    print("🎉 Advanced Translation Example Completed!")

if __name__ == "__main__":
    main()