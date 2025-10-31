import torch
from aceflow import Seq2SeqModel
from aceflow.utils import Tokenizer, create_data_loader
from aceflow.trainers import Trainer

# Sample data with more variety
english_sentences = [
    "Hello world! How are you today?",
    "I don't know what's happening...",
    "The quick brown fox jumps over the lazy dog.",
    "What's your name? My name is John.",
    "I love machine learning and AI!",
    "The weather is nice today, isn't it?",
    "See you later, alligator!",
    "Thank you very much for your help.",
    "Have a nice day and enjoy your meal!",
    "Where is the train station located?"
]

french_sentences = [
    "Bonjour le monde! Comment allez-vous aujourd'hui?",
    "Je ne sais pas ce qui se passe...",
    "Le rapide renard brun saute par-dessus le chien paresseux.",
    "Comment vous appelez-vous? Je m'appelle Jean.",
    "J'adore l'apprentissage automatique et l'IA!",
    "Le temps est agréable aujourd'hui, n'est-ce pas?",
    "À plus tard, alligator!",
    "Merci beaucoup pour votre aide.",
    "Passez une bonne journée et bon appétit!",
    "Où se trouve la gare?"
]

# Initialize enhanced tokenizers
src_tokenizer = Tokenizer(
    name="english_tokenizer",
    language="english",
    max_length=15,
    padding="post",
    truncation="post"
)

tgt_tokenizer = Tokenizer(
    name="french_tokenizer", 
    language="french",
    max_length=15,
    padding="post",
    truncation="post"
)

# Fit tokenizers
print("Fitting source tokenizer...")
src_tokenizer.fit(english_sentences, max_vocab_size=5000, min_freq=1)

print("Fitting target tokenizer...")
tgt_tokenizer.fit(french_sentences, max_vocab_size=5000, min_freq=1)

# Display tokenizer info
print("\nSource Tokenizer Info:")
print(src_tokenizer.info())

print("\nTarget Tokenizer Info:")
print(tgt_tokenizer.info())

# Create data loaders
train_loader = create_data_loader(
    english_sentences, french_sentences, 
    src_tokenizer, tgt_tokenizer, 
    batch_size=2, max_length=15
)

val_loader = create_data_loader(
    english_sentences[:2], french_sentences[:2],
    src_tokenizer, tgt_tokenizer,
    batch_size=2, max_length=15
)

# Initialize model
model = Seq2SeqModel(
    src_vocab_size=len(src_tokenizer),
    tgt_vocab_size=len(tgt_tokenizer),
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
    save_path="models/translation_model.ace",
    teacher_forcing_ratio=0.5,
    eval_every=1
)

# Save tokenizers to organized folders
src_tokenizer.save("tokenizers/english_tokenizer")
tgt_tokenizer.save("tokenizers/french_tokenizer")

# Save training history
trainer.save_training_history("training_history.json")

# Load tokenizers for inference
print("\nLoading tokenizers...")
loaded_src_tokenizer = Tokenizer.load("tokenizers/english_tokenizer")
loaded_tgt_tokenizer = Tokenizer.load("tokenizers/french_tokenizer")

# Load model for inference
loaded_model = Seq2SeqModel.load("models/translation_model_best.ace")

# Example inference with enhanced tokenizer
test_sentences = [
    "hello world",
    "how are you",
    "what is your name"
]

print("\n=== Translation Examples ===")
for test_sentence in test_sentences:
    # Encode with enhanced tokenizer
    encoded = loaded_src_tokenizer.encode(test_sentence, return_tensors='list')
    test_tensor = torch.tensor([encoded], dtype=torch.long)
    
    with torch.no_grad():
        output_sequence = loaded_model.beam_search(test_tensor, beam_width=3)
        translated = loaded_tgt_tokenizer.decode(output_sequence)
        
        print(f"Input: {test_sentence}")
        print(f"Encoded: {encoded}")
        print(f"Translation: {translated}")
        print("-" * 50)

# Test tokenizer functionality
print("\n=== Tokenizer Testing ===")
test_text = "Hello world! How are you today?"
print(f"Original: {test_text}")
print(f"Preprocessed: {src_tokenizer.preprocessor(test_text)}")
print(f"Tokenized: {src_tokenizer.tokenize(test_text)}")
print(f"Encoded: {src_tokenizer.encode(test_text)}")