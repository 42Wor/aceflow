import torch
from aceflow import Seq2SeqModel
from aceflow.utils import Tokenizer, create_data_loader
from aceflow.trainers import Trainer

# Very simple test data
english_sentences = ["hello world", "how are you", "what is your name"]
french_sentences = ["bonjour le monde", "comment allez-vous", "comment vous appelez-vous"]

# Initialize tokenizers
src_tokenizer = Tokenizer(max_length=10)
tgt_tokenizer = Tokenizer(max_length=10)

src_tokenizer.fit(english_sentences)
tgt_tokenizer.fit(french_sentences)

# Create data loaders
train_loader = create_data_loader(english_sentences, french_sentences, src_tokenizer, tgt_tokenizer, batch_size=2)
val_loader = create_data_loader(english_sentences[:2], french_sentences[:2], src_tokenizer, tgt_tokenizer, batch_size=2)

print("🚀 DEMONSTRATING EARLY STOPPING IN ACTION")
print("="*50)

# Test with very low patience to see immediate effect
model = Seq2SeqModel(
    src_vocab_size=len(src_tokenizer),
    tgt_vocab_size=len(tgt_tokenizer), 
    hidden_size=64,
    num_layers=1,
    use_attention=True
)

trainer = Trainer(
    model=model,
    learning_rate=0.001,
    early_stopping_patience=2,  # Stop after just 2 epochs without improvement
    early_stopping_min_delta=0.001
)

print("Training with early stopping (patience=2)...")
print("This will stop when validation loss doesn't improve for 2 consecutive epochs")
print("-" * 50)

history = trainer.train(
    train_loader,
    val_loader, 
    epochs=100,  # Set very high - early stopping will stop much sooner
    save_path="demo_model.ace",
    teacher_forcing_ratio=0.5,
    eval_every=1
)

print("\n" + "="*50)
print("RESULTS:")
print(f"📈 Total epochs possible: 100")
print(f"🛑 Epochs actually trained: {len(history['train_loss'])}")
print(f"✅ Early stopping triggered: {trainer.early_stop}")
print(f"🏆 Best validation loss: {trainer.best_val_loss:.4f}")

if trainer.early_stop:
    epochs_saved = 100 - len(history['train_loss'])
    print(f"💡 Saved {epochs_saved} epochs ({epochs_saved}% computation time saved!)")
    print("🎯 This demonstrates how early stopping prevents wasted training")
else:
    print("💡 Model kept improving throughout training")
    print("🎯 Early stopping would trigger if validation loss plateaued")

# Show the validation loss trend
print(f"\n📊 Validation loss trend (last 5 epochs):")
val_losses = history['val_loss'][-5:] if len(history['val_loss']) >= 5 else history['val_loss']
for i, loss in enumerate(val_losses, start=len(history['val_loss'])-len(val_losses)+1):
    trend = "➡️" if i > 1 and loss >= history['val_loss'][i-2] else "⬇️"
    print(f"  Epoch {i}: {loss:.4f} {trend}")