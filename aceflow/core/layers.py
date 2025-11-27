import torch
import torch.nn as nn
import numpy as np

try:
    from aceflow._rnn_ops import rnn_forward as rnn_forward_c
    HAS_C_EXTENSION = True
except ImportError:
    HAS_C_EXTENSION = False
    print("Warning: C extension not available, using PyTorch implementation")

class RNNLayer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0, 
                 rnn_type='lstm', bidirectional=False):
        super(RNNLayer, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type.lower()
        self.bidirectional = bidirectional
        
        # Store for C extension
        self._c_weights = None
        self._c_biases = None
        
        # Create PyTorch RNN layer (fallback)
        rnn_class = self._get_rnn_class()
        self.rnn = rnn_class(
            input_size, hidden_size, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Extract weights and biases for C extension
        if HAS_C_EXTENSION:
            self._extract_parameters_for_c()
    
    def _get_rnn_class(self):
        if self.rnn_type == 'rnn':
            return nn.RNN
        elif self.rnn_type == 'lstm':
            return nn.LSTM
        elif self.rnn_type == 'gru':
            return nn.GRU
        else:
            raise ValueError(f"Unsupported RNN type: {self.rnn_type}")
    
    def _extract_parameters_for_c(self):
        """Extract weights and biases for C extension"""
        all_weights = []
        all_biases = []
        
        for layer in self.rnn.all_weights:
            weights = []
            biases = []
            for param in layer:
                if param.dim() == 2:  # Weight matrix
                    weights.append(param.detach().numpy())
                else:  # Bias vector
                    biases.append(param.detach().numpy())
            
            # Flatten and concatenate
            if weights:
                all_weights.append(np.concatenate([w.flatten() for w in weights]))
            if biases:
                all_biases.append(np.concatenate([b.flatten() for b in biases]))
        
        if all_weights:
            self._c_weights = np.concatenate(all_weights).astype(np.float32)
        if all_biases:
            self._c_biases = np.concatenate(all_biases).astype(np.float32)
    
    def forward(self, x, hidden=None):
        # Always use PyTorch implementation for now until C extension is stable
        return self.rnn(x, hidden)
    
    def _forward_c(self, x, hidden=None):
        """C-optimized forward pass - TEMPORARILY DISABLED"""
        # For now, always use PyTorch implementation
        return self.rnn(x, hidden)
        
        # The C extension call below has argument count issues
        # We'll fix this after verifying the C extension works properly
        """
        batch_size, seq_len, input_size = x.size()
        
        # Initialize hidden states if not provided
        if hidden is None:
            if self.rnn_type == 'lstm':
                h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
                c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
                hidden = (h0, c0)
            else:
                hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        
        # Convert to numpy for C extension
        x_np = x.detach().numpy().astype(np.float32)
        
        if self.rnn_type == 'lstm':
            h_np = hidden[0].detach().numpy().astype(np.float32)
            c_np = hidden[1].detach().numpy().astype(np.float32)
        else:
            h_np = hidden.detach().numpy().astype(np.float32)
            c_np = np.zeros((self.num_layers, batch_size, self.hidden_size), dtype=np.float32)
        
        # Map RNN type to C enum
        rnn_type_map = {'rnn': 0, 'lstm': 1, 'gru': 2}
        c_rnn_type = rnn_type_map.get(self.rnn_type, 1)  # Default to LSTM
        
        # Call C extension - FIXED ARGUMENT COUNT
        # The C function expects 11 arguments, but we were passing 12
        try:
            output_np, final_hidden_np, final_cell_np = rnn_forward_c(
                x_np, h_np, c_np, self._c_weights, self._c_biases,
                batch_size, seq_len, input_size, self.hidden_size,
                self.num_layers, c_rnn_type  # Removed bidirectional parameter
            )
            
            # Convert back to torch tensors
            output = torch.from_numpy(output_np)
            
            if self.rnn_type == 'lstm':
                final_hidden = torch.from_numpy(final_hidden_np)
                final_cell = torch.from_numpy(final_cell_np)
                hidden_out = (final_hidden, final_cell)
            else:
                hidden_out = torch.from_numpy(final_hidden_np)
            
            return output, hidden_out
            
        except Exception as e:
            print(f"⚠️ C extension failed, falling back to PyTorch: {e}")
            return self.rnn(x, hidden)
        """
    
    def get_output_size(self):
        return self.hidden_size * (2 if self.bidirectional else 1)
class Encoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers=2, dropout=0.1, 
                 rnn_type='lstm', bidirectional=False, embedding_dim=None):
        super(Encoder, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        
        # Use hidden_size for embedding if not specified
        embedding_dim = embedding_dim or hidden_size
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        
        # RNN layer
        self.rnn = RNNLayer(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            rnn_type=rnn_type,
            bidirectional=bidirectional
        )
        
    def forward(self, x, hidden=None):
        embedded = self.dropout(self.embedding(x))
        output, hidden = self.rnn(embedded, hidden)
        return output, hidden
    
    def get_output_size(self):
        return self.rnn.get_output_size()

class Decoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers=2, dropout=0.1, 
                 rnn_type='lstm', encoder_bidirectional=False):
        super(Decoder, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        
        # Adjust hidden size if encoder is bidirectional
        encoder_factor = 2 if encoder_bidirectional else 1
        decoder_input_size = hidden_size * encoder_factor
        
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        # RNN layer (decoder is usually unidirectional)
        self.rnn = RNNLayer(
            input_size=hidden_size,  # Embedding size
            hidden_size=decoder_input_size,  # To match encoder output
            num_layers=num_layers,
            dropout=dropout,
            rnn_type=rnn_type,
            bidirectional=False
        )
        
        # Output projection
        self.out = nn.Linear(decoder_input_size, vocab_size)
        
    def forward(self, x, hidden):
        embedded = self.dropout(self.embedding(x))
        output, hidden = self.rnn(embedded, hidden)
        output = self.out(output)
        return output, hidden

class Attention(nn.Module):
    def __init__(self, hidden_size, method='dot'):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.method = method
        
        if method == 'general':
            self.attn = nn.Linear(hidden_size, hidden_size)
        elif method == 'concat':
            self.attn = nn.Linear(hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))
        
    def forward(self, hidden, encoder_outputs):
        if self.method == 'dot':
            attention_energies = self.dot_score(hidden, encoder_outputs)
        elif self.method == 'general':
            attention_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attention_energies = self.concat_score(hidden, encoder_outputs)
        
        return F.softmax(attention_energies, dim=1)
    
    def dot_score(self, hidden, encoder_outputs):
        return torch.sum(hidden * encoder_outputs, dim=2)
    
    def general_score(self, hidden, encoder_outputs):
        energy = self.attn(encoder_outputs)
        return torch.sum(hidden * energy, dim=2)
    
    def concat_score(self, hidden, encoder_outputs):
        hidden = hidden.unsqueeze(1).repeat(1, encoder_outputs.size(1), 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), 2)))
        return torch.sum(self.v * energy, dim=2)