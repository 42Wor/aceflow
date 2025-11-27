import torch
from aceflow import Seq2SeqModel
from aceflow import Tokenizer, create_data_loader
from aceflow import Seq2SeqTrainer

def clean_training_example():
    """Demonstrate clean training with the complete model"""
    
    print("🚀 Starting Clean Training Example")
    print("=" * 50)
    
    # Sample data
    english_sentences = [
        "hello world", "how are you", "good morning", 
        "what is your name", "thank you", "see you later"
    ]
    
    french_sentences = [
        "bonjour le monde", "comment allez vous", "bonjour", 
        "quel est votre nom", "merci", "a plus tard"
    ]

    print("📊 Preparing data...")
    # Initialize tokenizers
    src_tokenizer = Tokenizer(name="english", max_length=15)
    tgt_tokenizer = Tokenizer(name="french", max_length=15)
    src_tokenizer.fit(english_sentences)
    tgt_tokenizer.fit(french_sentences)

    print(f"📝 Vocabulary sizes - Source: {len(src_tokenizer)}, Target: {len(tgt_tokenizer)}")

    # Create data loaders
    train_loader = create_data_loader(
        english_sentences, french_sentences, 
        src_tokenizer, tgt_tokenizer, 
        batch_size=2, max_length=10
    )

    val_loader = create_data_loader(
        english_sentences[:2], french_sentences[:2],
        src_tokenizer, tgt_tokenizer,
        batch_size=2, max_length=10
    )

    print("🧠 Initializing model...")
    # Initialize model using factory function (recommended)
    model = Seq2SeqModel(
        src_vocab_size=len(src_tokenizer),
        tgt_vocab_size=len(tgt_tokenizer),
        hidden_size=64,
        num_layers=1,
        use_attention=True,
        rnn_type='lstm',
        dropout=0.1
    )

    print("⚡ Model Information:")
    model_info = model.get_rnn_info()
    for key, value in model_info.items():
        print(f"  {key}: {value}")

    print("⚡ Initializing trainer...")
    # Initialize trainer
    trainer = Seq2SeqTrainer(
        model=model,
        learning_rate=0.001,
        device='auto',
        early_stopping_patience=3,
        teacher_forcing_ratio=0.5
    )

    print("🎯 Starting training...")
    # Train model
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=5,  # Short for demo
        save_path="models/clean_model.ace"
    )

    print("\n✅ Training completed successfully!")
    
    # Save training history
    trainer.save_training_history("clean_training_history.json")
    
    # Save tokenizers
    src_tokenizer.save("tokenizers/english_clean")
    tgt_tokenizer.save("tokenizers/french_clean")
    
    # Test beam search
    print("🔍 Testing beam search...")
    test_sentence = "hello world"
    encoded = src_tokenizer.encode(test_sentence, return_tensors='list')
    input_tensor = torch.tensor([encoded], dtype=torch.long)
    
    with torch.no_grad():
        result = model.beam_search(input_tensor, beam_width=3, max_length=10)
        decoded = tgt_tokenizer.decode(result)
        print(f"  '{test_sentence}' → '{decoded}'")
    
    print("📁 Files saved:")
    print("  - models/clean_model.ace")
    print("  - models/clean_model_best.ace") 
    print("  - clean_training_history.json")
    print("  - tokenizers/english_clean/")
    print("  - tokenizers/french_clean/")
    
    return trainer, history

if __name__ == "__main__":
    trainer, history = clean_training_example()