import torch
from aceflow import Seq2SeqModel
from aceflow.utils import Tokenizer, create_data_loader
from aceflow.trainers import Seq2SeqTrainer

def clean_training_example():
    # Sample data
    english_sentences = ["hello world", "how are you", "good morning", "what is your name"]
    french_sentences = ["bonjour le monde", "comment allez vous", "bonjour", "quel est votre nom"]
    
    # Initialize tokenizers
    src_tokenizer = Tokenizer(name="english")
    tgt_tokenizer = Tokenizer(name="french")
    src_tokenizer.fit(english_sentences)
    tgt_tokenizer.fit(french_sentences)
    
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
    
    # Initialize model
    model = Seq2SeqModel(
        src_vocab_size=len(src_tokenizer),
        tgt_vocab_size=len(tgt_tokenizer),
        hidden_size=128,
        num_layers=2,
        use_attention=True
    )
    
    # Create trainer - auto-detect device and disable AMP if no GPU
    trainer = Seq2SeqTrainer(
        model=model,
        learning_rate=0.001,
        early_stopping_patience=5,
        teacher_forcing_ratio=0.5,
        use_amp=False,  # Disable AMP for CPU training
        gradient_clip=1.0
    )
    
    # Train model
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=80,
        save_path="models/clean_model.ace"
    )
    
    # Save training report
    trainer.save_training_history("training_history.json")
    
    print("\nTraining completed successfully!")
    return trainer, history

if __name__ == "__main__":
    trainer, history = clean_training_example()