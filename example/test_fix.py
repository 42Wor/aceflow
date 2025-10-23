from aceflow import Seq2Seq, Tokenizer, DataLoader

# Test that imports work
print("Testing imports...")

# Test tokenizer
tokenizer = Tokenizer()
tokenizer.build_vocab(["hello world", "test sentence"])
print(f"Tokenizer vocab size: {tokenizer.vocab_size}")

# Test model creation
model = Seq2Seq(
    src_vocab_size=100,
    tgt_vocab_size=100,
    embedding_dim=128,
    hidden_dim=256
)
print("Model created successfully")

# Test data loader
src_texts = ["hello world", "how are you"]
tgt_texts = ["hola mundo", "como estas"]
data_loader = DataLoader(src_texts, tgt_texts, tokenizer, tokenizer, batch_size=2)
print("DataLoader created successfully")

print("All tests passed!")