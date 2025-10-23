import numpy as np
from aceflow import Seq2Seq, Tokenizer, DataLoader, Trainer, ACEFormat

# Sample data
src_texts = ["hello world", "how are you", "good morning"]
tgt_texts = ["hola mundo", "como estas", "buenos dias"]

# Build tokenizers
src_tokenizer = Tokenizer()
tgt_tokenizer = Tokenizer()
src_tokenizer.build_vocab(src_texts)
tgt_tokenizer.build_vocab(tgt_texts)

# Create data loader
data_loader = DataLoader(src_texts, tgt_texts, src_tokenizer, tgt_tokenizer, batch_size=2)

# Create model
model = Seq2Seq(
    src_vocab_size=src_tokenizer.vocab_size,
    tgt_vocab_size=tgt_tokenizer.vocab_size,
    embedding_dim=128,
    hidden_dim=256,
    num_layers=2,
    use_attention=True
)

# Train model
trainer = Trainer(model, learning_rate=0.001)
for epoch in range(10):
    loss = trainer.train_epoch(data_loader)
    print(f"Epoch {epoch+1}, Loss: {loss:.4f}")

# Save model
ACEFormat.save_model(model, "seq2seq_model.ace")

# Load model
loaded_model = ACEFormat.load_model("seq2seq_model.ace")

# Make prediction
test_src = src_tokenizer.encode("hello friend")
prediction = loaded_model.predict(test_src.reshape(1, -1))
print("Prediction:", tgt_tokenizer.decode(prediction[0]))