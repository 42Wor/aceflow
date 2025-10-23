from aceflow import Seq2Seq, Tokenizer, DataLoader, Trainer, ACEFormat
import numpy as np

# Sample data - use simpler sentences
src_texts = ["hello world", "how are you", "good morning", "nice day"]
tgt_texts = ["hola mundo", "como estas", "buenos dias", "buen dia"]

# Build tokenizers
src_tokenizer = Tokenizer()
tgt_tokenizer = Tokenizer()
src_tokenizer.build_vocab(src_texts)
tgt_tokenizer.build_vocab(tgt_texts)

print(f"Source vocab size: {src_tokenizer.vocab_size}")
print(f"Target vocab size: {tgt_tokenizer.vocab_size}")

# Create data loader with smaller batch size
data_loader = DataLoader(src_texts, tgt_texts, src_tokenizer, tgt_tokenizer, 
                        batch_size=2, max_length=10)

# Create model with smaller dimensions for testing
model = Seq2Seq(
    src_vocab_size=src_tokenizer.vocab_size,
    tgt_vocab_size=tgt_tokenizer.vocab_size,
    embedding_dim=64,    # Smaller for testing
    hidden_dim=128,      # Smaller for testing
    num_layers=1,        # Simpler
    use_attention=True
)

# Train model with lower learning rate
trainer = Trainer(model, learning_rate=0.0001)

print("Starting training...")
for epoch in range(5):  # Fewer epochs for testing
    loss = trainer.train_epoch(data_loader)
    print(f"Epoch {epoch+1}, Loss: {loss:.4f}")

# Save model
ACEFormat.save_model(model, "seq2seq_model.ace")
print("Model saved as 'seq2seq_model.ace'")

# Test prediction
print("\nTesting prediction...")
test_text = "hello"
test_encoded = src_tokenizer.encode(test_text).reshape(1, -1)
prediction = model.predict(test_encoded)
decoded = tgt_tokenizer.decode(prediction[0])
print(f"Input: '{test_text}' -> Output: '{decoded}'")