import numpy as np
from typing import Optional, Tuple

class Layer:
    def __init__(self):
        self.params = {}
        self.grads = {}
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    def backward(self, dout: np.ndarray) -> np.ndarray:
        raise NotImplementedError

class Dense(Layer):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Xavier initialization
        scale = np.sqrt(2.0 / (input_dim + output_dim))
        self.params['W'] = np.random.randn(input_dim, output_dim) * scale
        self.params['b'] = np.zeros((1, output_dim))
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        return np.dot(x, self.params['W']) + self.params['b']
    
    def backward(self, dout: np.ndarray) -> np.ndarray:
        self.grads['W'] = np.dot(self.x.T, dout)
        self.grads['b'] = np.sum(dout, axis=0, keepdims=True)
        return np.dot(dout, self.params['W'].T)

class LSTMCell(Layer):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Combined weights for input, forget, output, and candidate gates
        total_dim = input_dim + hidden_dim
        self.params['W'] = np.random.randn(total_dim, 4 * hidden_dim) * 0.01
        self.params['b'] = np.zeros((1, 4 * hidden_dim))
    
    def forward(self, x: np.ndarray, h_prev: np.ndarray, c_prev: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        self.x = x
        self.h_prev = h_prev
        self.c_prev = c_prev
        
        # Combine input and previous hidden state
        combined = np.concatenate([x, h_prev], axis=1)
        
        # Compute all gates
        gates = np.dot(combined, self.params['W']) + self.params['b']
        
        # Split gates
        i, f, o, g = np.split(gates, 4, axis=1)
        
        # Apply activations
        i = self.sigmoid(i)  # Input gate
        f = self.sigmoid(f)  # Forget gate
        o = self.sigmoid(o)  # Output gate
        g = np.tanh(g)       # Candidate memory
        
        # Update cell state and hidden state
        c_next = f * c_prev + i * g
        h_next = o * np.tanh(c_next)
        
        self.cache = (i, f, o, g, combined, c_next)
        return h_next, c_next
    
    def backward(self, dh_next: np.ndarray, dc_next: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        i, f, o, g, combined, c_next = self.cache
        
        # Gradients through output gate and cell state
        do = dh_next * np.tanh(c_next)
        dc = dc_next + dh_next * o * (1 - np.tanh(c_next) ** 2)
        
        # Gradients through forget, input, and candidate gates
        df = dc * self.c_prev
        di = dc * g
        dg = dc * i
        
        # Gate derivatives
        di_input = di * i * (1 - i)
        df_input = df * f * (1 - f)
        do_input = do * o * (1 - o)
        dg_input = dg * (1 - g ** 2)
        
        # Combine gate gradients
        dgates = np.concatenate([di_input, df_input, do_input, dg_input], axis=1)
        
        # Gradients for weights and bias
        self.grads['W'] = np.dot(combined.T, dgates)
        self.grads['b'] = np.sum(dgates, axis=0, keepdims=True)
        
        # Gradient for combined input
        dcombined = np.dot(dgates, self.params['W'].T)
        
        # Split gradients
        dx = dcombined[:, :self.x.shape[1]]
        dh_prev = dcombined[:, self.x.shape[1]:]
        
        # Gradient for previous cell state
        dc_prev = dc * f
        
        return dx, dh_prev, dc_prev
    
    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.clip(x, -50, 50)))

class Embedding(Layer):
    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.params['W'] = np.random.randn(vocab_size, embedding_dim) * 0.01
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        return self.params['W'][x]
    
    def backward(self, dout: np.ndarray) -> np.ndarray:
        # For embedding layer, we only update the gradients for used embeddings
        np.add.at(self.grads['W'], self.x, dout)
        return None