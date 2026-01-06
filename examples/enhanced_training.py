import torch
import numpy as np
from aceflow import Seq2SeqModel
from aceflow.utils import Tokenizer, create_data_loader
from aceflow.trainers import Seq2SeqTrainer, ModelCheckpoint, EarlyStopping, ProgressLogger
from aceflow.trainers.training_utils import plot_training_history, save_training_report, count_parameters

def demo_enhanced_training():
    """Demonstrate enhanced training features with the new trainer system"""
    
    print("🚀 Starting Enhanced Training Demo")
    print("=" * 50)
    
    # Sample data
    english_sentences = [
        "hello world", "how are you", "good morning", "what is your name",
        "i love programming", "the weather is nice", "see you later",
        "thank you", "have a nice day", "where is the station"
    ] * 100  # Repeat for more data
    
    french_sentences = [
        "bonjour le monde", "comment allez vous", "bonjour", "quel est votre nom",
        "j aime la programmation", "le temps est agreable", "a plus tard",
        "merci", "passez une bonne journee", "ou est la gare"
    ] * 100

    print("📊 Preparing data...")
    # Initialize tokenizers
    src_tokenizer = Tokenizer(name="english", max_length=15)
    tgt_tokenizer = Tokenizer(name="french", max_length=15)
    src_tokenizer.fit(english_sentences)
    tgt_tokenizer.fit(french_sentences)

    print(f"📝 Vocabulary sizes - Source: {len(src_tokenizer)}, Target: {len(tgt_tokenizer)}")
   # src_tokenizer.save("english_tokenizer")
    #tgt_tokenizer.save("french_tokenizer")
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

    print("🧠 Initializing model...")
    # Initialize model
    model = Seq2SeqModel(
        src_vocab_size=len(src_tokenizer),
        tgt_vocab_size=len(tgt_tokenizer),
        hidden_size=128,
        num_layers=2,
        use_attention=True,
        dropout=0.1,
        rnn_type='lstm'
    )

    print("⚡ Initializing enhanced trainer...")
    # Initialize enhanced trainer with proper parameters
    trainer = Seq2SeqTrainer(
        model=model,
        learning_rate=0.001,
        device='auto',  # Auto-detect device (CPU/GPU/MPS)
        early_stopping_patience=7,
        gradient_clip=1.0,
        use_amp=False,  # Set to True if you have CUDA and want mixed precision
        teacher_forcing_ratio=0.5
    )

    print("🛠️ Setting up training callbacks...")
    # Setup advanced callbacks
    callbacks = [
        ModelCheckpoint(
            "models/enhanced_model_best.ace", 
            monitor='val_loss',
            save_best_only=True,
            mode='min'
        ),
        EarlyStopping(
            patience=7,
            monitor='val_loss',
            min_delta=0.001,
            mode='min'
        ),
        ProgressLogger()
    ]

    print("🎯 Starting training...")
    # Train with enhanced features
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=30,  # Reduced for demo purposes
        save_path="models/enhanced_model.ace",
        eval_every=1,
        callbacks=callbacks
    )

    print("\n✅ Training completed!")
    print("=" * 50)

    # Get training summary with proper type conversion
    best_epoch = trainer.get_best_epoch()
    best_val_loss = float(min(history['val_loss'])) if history['val_loss'] else float('inf')
    final_train_loss = float(history['train_loss'][-1]) if history['train_loss'] else 0.0
    final_val_loss = float(history['val_loss'][-1]) if history['val_loss'] else 0.0
    final_train_acc = float(history['train_accuracy'][-1]) if history['train_accuracy'] else 0.0
    final_val_acc = float(history['val_accuracy'][-1]) if history['val_accuracy'] else 0.0

    print("\n📈 Training Summary:")
    print(f"  • Best epoch: {int(best_epoch) + 1}")
    print(f"  • Best validation loss: {best_val_loss:.4f}")
    print(f"  • Final training loss: {final_train_loss:.4f}")
    print(f"  • Final validation loss: {final_val_loss:.4f}")
    print(f"  • Final training accuracy: {final_train_acc:.4f}")
    print(f"  • Final validation accuracy: {final_val_acc:.4f}")
    print(f"  • Total parameters: {count_parameters(model):,}")
    print(f"  • Device used: {trainer.device}")

    # Save training history
    trainer.save_training_history("training_history.json")
    
    # Create training report with proper serialization
    config = {
        'model': {
            'src_vocab_size': int(len(src_tokenizer)),
            'tgt_vocab_size': int(len(tgt_tokenizer)),
            'hidden_size': 128,
            'num_layers': 2,
            'use_attention': True,
            'rnn_type': 'lstm',
            'total_parameters': int(count_parameters(model))
        },
        'training': {
            'epochs': 30,
            'batch_size': 8,
            'learning_rate': 0.001,
            'early_stopping_patience': 7,
            'teacher_forcing_ratio': 0.5,
            'device': str(trainer.device)
        },
        'data': {
            'train_samples': 80,
            'val_samples': 20,
            'src_vocab_size': int(len(src_tokenizer)),
            'tgt_vocab_size': int(len(tgt_tokenizer)),
            'max_sequence_length': 15
        }
    }
    
    try:
        save_training_report(history, config, "training_report.json")
        print("✓ Training report saved to: training_report.json")
    except Exception as e:
        print(f"⚠️ Could not save training report: {e}")

    # Plot training history
    try:
        plot_training_history(history, "training_plots.png", show=False)
    except Exception as e:
        print(f"⚠️ Could not generate plots: {e}")

    # Demonstrate translation with trained model
    print("\n🔍 Testing translation...")
    test_sentences = [
        "hello world",
        "how are you", 
        "what is your name",
        "thank you",
        "see you later"
    ]

    # Load the best model for inference
    try:
        best_model = Seq2SeqModel.load("models/enhanced_model_best.ace")
        best_model.eval()
        
        print("  Translation Examples:")
        print("  " + "-" * 40)
        for test_sentence in test_sentences:
            # Encode input
            encoded = src_tokenizer.encode(test_sentence, return_tensors='list')
            input_tensor = torch.tensor([encoded], dtype=torch.long)
            
            # Generate translation
            with torch.no_grad():
                if hasattr(best_model, 'beam_search'):
                    output_sequence = best_model.beam_search(input_tensor, beam_width=3)
                else:
                    outputs = best_model(input_tensor, teacher_forcing_ratio=0)
                    output_sequence = outputs.argmax(dim=-1)[0].cpu().numpy()
                
                translation = tgt_tokenizer.decode(output_sequence)
                print(f"  • '{test_sentence}' → '{translation}'")
                
    except Exception as e:
        print(f"  • Could not load model for translation test: {e}")

    # Save tokenizers
    try:
        src_tokenizer.save("tokenizers/english_tokenizer")
        tgt_tokenizer.save("tokenizers/french_tokenizer")
        print("✓ Tokenizers saved to: tokenizers/")
    except Exception as e:
        print(f"⚠️ Could not save tokenizers: {e}")

    print("\n🎉 Enhanced training demo completed successfully!")
    print("📁 Generated files:")
    print("  - models/enhanced_model.ace (Final model)")
    print("  - models/enhanced_model_best.ace (Best model)") 
    print("  - training_history.json (Training metrics)")
    print("  - training_report.json (Detailed report)")
    print("  - training_plots.png (Training curves)")
    print("  - tokenizers/ (Saved tokenizers)")

    return trainer, history

if __name__ == "__main__":
    trainer, history = demo_enhanced_training()