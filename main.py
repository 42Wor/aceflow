import numpy as np
import aceflow

# --- 1. Data Preparation ---

# A simple vocabulary for our character-level model
# PAD = padding, EOS = end of sequence, SOS = start of sequence
VOCAB = ['<PAD>', 'h', 'e', 'l', 'o', 'w', 'r', 'd']
SEQ_LENGTH = 5

class Vocabulary:
    def __init__(self, vocab_list):
        self.vocab = vocab_list
        self.char_to_int = {char: i for i, char in enumerate(self.vocab)}
        self.int_to_char = {i: char for i, char in enumerate(self.vocab)}

    def to_indices(self, chars):
        return [self.char_to_int[c] for c in chars]

    def to_chars(self, indices):
        return "".join([self.int_to_char[i] for i in indices])

    def size(self):
        return len(self.vocab)

# --- 2. Generate Synthetic Data ---
# Our task is to reverse a sequence. e.g., "hello" -> "olleh"
def generate_data(num_samples, vocab, seq_length):
    X = []
    y = []
    for _ in range(num_samples):
        # Generate a random sequence of characters
        indices = np.random.randint(1, len(vocab), size=seq_length) # Skip PAD
        X.append(list(indices))
        y.append(list(reversed(indices)))
    
    # Convert to one-hot for the target
    y_one_hot = np.zeros((num_samples, seq_length, len(vocab)), dtype=np.float32)
    for i, seq in enumerate(y):
        for t, char_idx in enumerate(seq):
            y_one_hot[i, t, char_idx] = 1.0
            
    return np.array(X, dtype=np.int32), y_one_hot

# --- 3. Main Script ---
if __name__ == "__main__":
    print("--- Building AceFlow Char-Reversal Model ---")
    
    # Setup
    vocab = Vocabulary(VOCAB)
    HIDDEN_SIZE = 64
    EMBED_SIZE = 16
    
    # Create the model from our library
    model = aceflow.Seq2Seq(
        vocab_size=vocab.size(),
        hidden_size=HIDDEN_SIZE,
        embed_size=EMBED_SIZE
    )
    
    print("Model initialized. Generating training data...")
    X_train, y_train = generate_data(5000, vocab.vocab, SEQ_LENGTH)
    
    print("Data generated. Starting training (offloaded to Rust core)...")
    # Train the model
    model.fit(X_train, y_train, epochs=100, learning_rate=0.01, batch_size=64)
    
    print("\n--- Training Complete. Testing Predictions ---")
    
    # Test on a few examples
    test_words = ["hello", "world"]
    for word in test_words:
        # Convert word to integer sequence
        input_indices = vocab.to_indices(word)
        
        # Get prediction from the model
        predicted_indices = model.predict(input_indices)
        
        # Convert prediction back to characters
        predicted_word = vocab.to_chars(predicted_indices)
        
        print(f"Input:    '{word}'")
        print(f"Expected: '{word[::-1]}'")
        print(f"Predicted: '{predicted_word}'\n")