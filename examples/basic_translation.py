"""
Basic translation example with AceFlow
Demonstrates simple English to French translation
"""

import torch
import os
from aceflow import Seq2SeqModel
from aceflow.utils import Tokenizer, create_data_loader
from aceflow.trainers import Trainer

def setup_directories():
    """Create necessary directories"""
    os.makedirs("models", exist_ok=True)
    os.makedirs("tokenizers", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

def main():
    print("🚀 AceFlow Basic Translation Example")
    print("=" * 50)
    
    # Setup directories
    setup_directories()
    
    # Sample training data (English to French)
    english_sentences = [
        "hello world", "how are you", "good morning", "what is your name",
        "i love programming", "the weather is nice", "see you later",
        "thank you", "have a nice day", "where is the station"
    ]

    french_sentences = [
        "bonjour le monde", "comment allez vous", "bonjour", "quel est votre nom",
        "j aime la programmation", "le temps est agreable", "a plus tard",
        "merci", "passez une bonne journee", "ou est la gare"
    ]

    print("📝 Sample Data:")
    for i, (eng, fr) in enumerate(zip(english_sentences[:3], french_sentences[:3])):
        print(f"  {eng} -> {fr}")
    
    # Initialize tokenizers
    print("\n🔤 Initializing Tokenizers...")
    src_tokenizer = Tokenizer(
        name="english_basic",
        language="english",
        max_length=15,
        padding="post"
    )
    
    tgt_tokenizer = Tokenizer(
        name="french_basic", 
        language="french",
        max_length=15,
        padding="post"
    )

    # Fit tokenizers
    print("📊 Fitting tokenizers...")
    src_tokenizer.fit(english_sentences, max_vocab_size=1000, min_freq=1)
    tgt_tokenizer.fit(french_sentences, max_vocab_size=1000, min_freq=1)
    
    print(f"✅ Source vocabulary size: {len(src_tokenizer)}")
    print(f"✅ Target vocabulary size: {len(tgt_tokenizer)}")

    # Create data loaders
    print("\n📦 Creating Data Loaders...")
    train_loader = create_data_loader(
        english_sentences, french_sentences, 
        src_tokenizer, tgt_tokenizer, 
        batch_size=2, max_length=15
    )

    # Initialize model
    print("\n🧠 Initializing Model...")
    model = Seq2SeqModel(
        src_vocab_size=len(src_tokenizer),
        tgt_vocab_size=len(tgt_tokenizer),
        hidden_size=128,
        num_layers=2,
        rnn_type='lstm',
        use_attention=True,
        teacher_forcing_ratio=0.5
    )

    # Display model info
    model_info = model.get_rnn_info()
    print(f"✅ Model Type: {model_info['rnn_type'].upper()}")
    print(f"✅ Hidden Size: {model_info['hidden_size']}")
    print(f"✅ Attention: {model_info['has_attention']}")
    print(f"✅ Parameters: {model_info['total_parameters']:,}")

    # Initialize trainer
    print("\n🎯 Initializing Trainer...")
    trainer = Trainer(model, learning_rate=0.001)

    # Train model
    print("\n🏋️ Training Model...")
    history = trainer.train(
        train_loader, train_loader,  # Using same data for train/val for simplicity
        epochs=20,
        save_path="models/basic_translation.ace",
        teacher_forcing_ratio=0.5,
        eval_every=2
    )

    # Save tokenizers
    print("\n💾 Saving Tokenizers...")
    src_tokenizer.save("tokenizers/english_basic")
    tgt_tokenizer.save("tokenizers/french_basic")

    # Load model for inference
    print("\n🔍 Loading Model for Inference...")
    loaded_model = Seq2SeqModel.load("models/basic_translation.ace")
    loaded_src_tokenizer = Tokenizer.load("tokenizers/english_basic")
    loaded_tgt_tokenizer = Tokenizer.load("tokenizers/french_basic")

    # Test translations
    print("\n🌍 Translation Examples:")
    test_sentences = [
        "hello world",
        "how are you", 
        "good morning",
        "thank you"
    ]

    def translate(text, model, src_tokenizer, tgt_tokenizer):
        encoded = src_tokenizer.encode(text, return_tensors='list')
        input_tensor = torch.tensor([encoded], dtype=torch.long)
        
        with torch.no_grad():
            output_sequence = model.beam_search(input_tensor, beam_width=3)
            translation = tgt_tokenizer.decode(output_sequence)
        
        return translation

    for text in test_sentences:
        translation = translate(text, loaded_model, loaded_src_tokenizer, loaded_tgt_tokenizer)
        print(f"  '{text}' -> '{translation}'")

    # Save training history
    trainer.save_training_history("logs/basic_training_history.json")
    print(f"\n✅ Training history saved to: logs/basic_training_history.json")
    print(f"✅ Model saved to: models/basic_translation.ace")
    print(f"✅ Tokenizers saved to: tokenizers/")

    print("\n🎉 Basic Translation Example Completed Successfully!")

if __name__ == "__main__":
    main()