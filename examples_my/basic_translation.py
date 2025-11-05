import torch
from aceflow import Seq2SeqModel, Tokenizer
from aceflow.trainers import Trainer
from aceflow.utils.data_loader import create_data_loader

# Sample data
english_sentences = [
    "hello world", "how are you", "good morning", "what is your name",
    "i love programming", "the weather is nice", "see you later",
    "thank you", "have a nice day", "where is the station"
]*10 

french_sentences = [
    "bonjour le monde", "comment allez vous", "bonjour", "quel est votre nom",
    "j aime la programmation", "le temps est agreable", "a plus tard",
    "merci", "passez une bonne journee", "ou est la gare"
]*10

# Initialize tokenizers
src_tokenizer = Tokenizer()
tgt_tokenizer = Tokenizer()

# Build vocabularies
src_tokenizer.fit(english_sentences)
tgt_tokenizer.fit(french_sentences)

print(f"Source vocabulary size: {src_tokenizer.get_vocab_size}")
print(f"Target vocabulary size: {tgt_tokenizer.get_vocab_size}")

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
    src_get_vocab_size=src_tokenizer.vocab_size,
    tgt_get_vocab_size=tgt_tokenizer.vocab_size,
    hidden_size=128,
    num_layers=2,
    use_attention=True
)

# Initialize trainer
trainer = Trainer(model, learning_rate=0.001)

# Train model with proper save_path
# Train model with proper save_path
history = trainer.train(
    train_loader, val_loader, 
    epochs=10, 
    save_path="models/translation_model.ace",  # This is the base path
    teacher_forcing_ratio=0.5,
    eval_every=1
)

# Save training history
trainer.save_training_history("training_history.json")

# Save tokenizers
src_tokenizer.save("models/src_tokenizer.pkl")
tgt_tokenizer.save("models/tgt_tokenizer.pkl")

# Load model for inference - USE THE CORRECT PATH
# Option 1: Load the best model
loaded_model = Seq2SeqModel.load("models/translation_model_best.ace")

# Option 2: Load the final model  
# loaded_model = Seq2SeqModel.load("models/translation_model_final.ace")

# Option 3: Load specific epoch
# loaded_model = Seq2SeqModel.load("models/translation_model_epoch_10.ace")

# Example inference
test_sentence = "hello world"
test_encoded = src_tokenizer.encode(test_sentence)
test_tensor = torch.tensor([test_encoded], dtype=torch.long)

with torch.no_grad():
    output_sequence = loaded_model.beam_search(test_tensor, beam_width=3)
    translated = tgt_tokenizer.decode(output_sequence)
    print(f"Input: {test_sentence}")
    print(f"Translation: {translated}")