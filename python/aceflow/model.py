import numpy as np
from . import aceflow_core

def he_initialization(n_in, n_out):
    """ He initialization for weights. """
    return np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in)

class Seq2Seq:
    """
    A character-level Seq2Seq model for sequence reversal.
    The heavy computation for training is offloaded to a Rust core.
    """
    def __init__(self, vocab_size, hidden_size, embed_size):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size

        # --- Initialize parameters as NumPy arrays ---
        # Python will own and manage the memory for these arrays.
        
        # Embedding
        self.params = {}
        self.params['W_embed'] = np.random.randn(vocab_size, embed_size) * 0.01

        # Encoder GRU
        concat_size_enc = embed_size + hidden_size
        self.params['W_uz_enc'] = he_initialization(concat_size_enc, hidden_size)
        self.params['W_ur_enc'] = he_initialization(concat_size_enc, hidden_size)
        self.params['W_uh_enc'] = he_initialization(concat_size_enc, hidden_size)

        # Decoder GRU
        concat_size_dec = embed_size + hidden_size
        self.params['W_uz_dec'] = he_initialization(concat_size_dec, hidden_size)
        self.params['W_ur_dec'] = he_initialization(concat_size_dec, hidden_size)
        self.params['W_uh_dec'] = he_initialization(concat_size_dec, hidden_size)
        
        # Output Layer
        self.params['W_out'] = he_initialization(hidden_size, vocab_size)
        self.params['b_out'] = np.zeros((1, vocab_size))

    def fit(self, X, y, epochs=100, batch_size=32, learning_rate=0.001):
        """
        Trains the model.
        X: Input sequences (integer encoded)
        y: Target sequences (one-hot encoded)
        """
        num_samples = X.shape[0]
        
        for epoch in range(epochs):
            epoch_loss = 0
            
            # Simple batching
            for i in range(0, num_samples, batch_size):
                X_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size]
                
                # Call the high-performance Rust function for a training step
                loss, gradients = aceflow_core.run_training_step(
                    self.params['W_embed'],
                    self.params['W_uz_enc'], self.params['W_ur_enc'], self.params['W_uh_enc'],
                    self.params['W_uz_dec'], self.params['W_ur_dec'], self.params['W_uh_dec'],
                    self.params['W_out'], self.params['b_out'],
                    X_batch, y_batch,
                    learning_rate
                )
                
                epoch_loss += loss
                
                # --- Update weights in Python using gradients from Rust ---
                # NOTE: Because our Rust BPTT is simplified, we only update the output layer.
                # A full implementation would update all parameters.
                if 'dw_out' in gradients:
                    self.params['W_out'] -= learning_rate * gradients['dw_out']
                    self.params['b_out'] -= learning_rate * gradients['db_out']
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss / (num_samples / batch_size):.4f}")

    def predict(self, input_seq):
        """ Predicts the reversed sequence for a single input. """
        # This forward pass is implemented in Python for simplicity.
        # It could also be moved to Rust for max performance.
        
        # --- Encoder ---
        h = np.zeros((1, self.hidden_size))
        for char_idx in input_seq:
            embed = self.params['W_embed'][char_idx, :].reshape(1, -1)
            combined = np.hstack([embed, h])
            z = 1 / (1 + np.exp(-combined @ self.params['W_uz_enc']))
            r = 1 / (1 + np.exp(-combined @ self.params['W_ur_enc']))
            h_tilde_combined = np.hstack([embed, r * h])
            h_tilde = np.tanh(h_tilde_combined @ self.params['W_uh_enc'])
            h = (1 - z) * h + z * h_tilde
            
        # --- Decoder ---
        decoder_h = h
        # Start with the reversed version of the first character for decoding
        decoder_input_idx = input_seq[-1]
        output_seq = []

        for _ in range(len(input_seq)):
            embed = self.params['W_embed'][decoder_input_idx, :].reshape(1, -1)
            combined = np.hstack([embed, decoder_h])
            z = 1 / (1 + np.exp(-combined @ self.params['W_uz_dec']))
            r = 1 / (1 + np.exp(-combined @ self.params['W_ur_dec']))
            h_tilde_combined = np.hstack([embed, r * decoder_h])
            h_tilde = np.tanh(h_tilde_combined @ self.params['W_uh_dec'])
            decoder_h = (1 - z) * decoder_h + z * h_tilde
            
            logits = decoder_h @ self.params['W_out'] + self.params['b_out']
            
            # Softmax
            probs = np.exp(logits - np.max(logits)) / np.sum(np.exp(logits - np.max(logits)))
            
            # Get the character with the highest probability
            next_char_idx = np.argmax(probs)
            output_seq.append(next_char_idx)
            decoder_input_idx = next_char_idx
            
        return output_seq