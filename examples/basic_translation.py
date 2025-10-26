import torch
from aceflow import Seq2SeqModel, Tokenizer
from aceflow.trainers import Trainer
from aceflow.utils.data_loader import create_data_loader

# Sample data
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

# Initialize tokenizers
src_tokenizer = Tokenizer()
tgt_tokenizer = Tokenizer()

# Build vocabularies
src_tokenizer.fit(english_sentences)
tgt_tokenizer.fit(french_sentences)

print(f"Source vocabulary size: {src_tokenizer.vocab_size}")
print(f"Target vocabulary size: {tgt_tokenizer.vocab_size}")

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
    src_vocab_size=src_tokenizer.vocab_size,
    tgt_vocab_size=tgt_tokenizer.vocab_size,
    hidden_size=128,
    num_layers=2,
    use_attention=True
)

# Initialize trainer
trainer = Trainer(model, learning_rate=0.001)

# Train model
history = trainer.train(
    train_loader, val_loader, 
    epochs=10, 
    save_path="translation_model.ace"
)

# Save tokenizers
src_tokenizer.save("src_tokenizer.pkl")
tgt_tokenizer.save("tgt_tokenizer.pkl")

# Load model for inference
loaded_model = Seq2SeqModel.load("translation_model.ace")

# Example inference
test_sentence = "hello world"
test_encoded = src_tokenizer.encode(test_sentence)
test_tensor = torch.tensor([test_encoded], dtype=torch.long)

with torch.no_grad():
    output_sequence = loaded_model.beam_search(test_tensor, beam_width=3)
    translated = tgt_tokenizer.decode(output_sequence)
    print(f"Input: {test_sentence}")
    print(f"Translation: {translated}")