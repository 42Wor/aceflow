import torch
from aceflow import Seq2SeqModel
from aceflow.utils import Tokenizer, create_data_loader
from aceflow.trainers import Trainer, ExampleCallbacks

def demo_enhanced_training():
    """Demonstrate enhanced training features"""
    
    # Sample data
    english_sentences = [
        "hello world", "how are you", "good morning", "what is your name",
        "i love programming", "the weather is nice", "see you later",
        "thank you", "have a nice day", "where is the station"
    ] * 10  # Repeat for more data
    
    french_sentences = [
        "bonjour le monde", "comment allez vous", "bonjour", "quel est votre nom",
        "j aime la programmation", "le temps est agreable", "a plus tard",
        "merci", "passez une bonne journee", "ou est la gare"
    ] * 10

    # Initialize tokenizers
    src_tokenizer = Tokenizer(name="english")
    tgt_tokenizer = Tokenizer(name="french")
    src_tokenizer.fit(english_sentences)
    tgt_tokenizer.fit(french_sentences)

    # Create data loaders
    train_loader = create_data_loader(
        english_sentences[:80], french_sentences[:80], 
        src_tokenizer, tgt_tokenizer, 
        batch_size=8, max_length=15
    )

    val_loader = create_data_loader(
        english_sentences[80:], french_sentences[80:],
        src_tokenizer, tgt_tokenizer,
        batch_size=8, max_length=15
    )

    # Initialize model
    model = Seq2SeqModel(
        src_vocab_size=len(src_tokenizer),
        tgt_vocab_size=len(tgt_tokenizer),
        hidden_size=128,
        num_layers=2,
        use_attention=True
    )

    # Initialize enhanced trainer
    trainer = Trainer(
        model=model,
        learning_rate=0.001,
        optimizer='adam',
        device='auto',
        max_grad_norm=1.0,
        scheduler='plateau',  # Use learning rate scheduler
        scheduler_patience=3,
        scheduler_factor=0.5
    )

    # Add callbacks
    trainer.add_callback(ExampleCallbacks())

    # Train with enhanced features
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=60,
        save_path="enhanced_model.ace",
        teacher_forcing_ratio=0.5,
        eval_every=1,
        early_stopping_patience=5,  # Enable early stopping
        resume=False
    )

    # Get training summary
    summary = trainer.get_training_summary()
    print("\nTraining Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    # Plot training history
    trainer.plot_training_history("training_plot.png")

    # Evaluate final model
    results = trainer.evaluate(val_loader)
    print(f"\nFinal Evaluation: {results}")

    # Save tokenizers
    src_tokenizer.save("english_tokenizer")
    tgt_tokenizer.save("french_tokenizer")

if __name__ == "__main__":
    demo_enhanced_training()