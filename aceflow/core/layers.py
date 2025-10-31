import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RNNLayer(nn.Module):
    """Unified RNN layer supporting multiple RNN types"""
    
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0, 
                 rnn_type='lstm', bidirectional=False):
        super(RNNLayer, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type.lower()
        self.bidirectional = bidirectional
        
        # Validate RNN type
        valid_rnn_types = ['rnn', 'lstm', 'gru', 'birnn', 'bilstm', 'bigru']
        if self.rnn_type not in valid_rnn_types:
            raise ValueError(f"Invalid RNN type: {rnn_type}. Choose from {valid_rnn_types}")
        
        # Handle bidirectional types
        if self.rnn_type.startswith('bi'):
            self.bidirectional = True
            self.rnn_type = self.rnn_type[2:]  # Remove 'bi' prefix
        
        # Create RNN layer
        if self.rnn_type == 'rnn':
            self.rnn = nn.RNN(
                input_size, hidden_size, num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional
            )
        elif self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(
                input_size, hidden_size, num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional
            )
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(
                input_size, hidden_size, num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=bidirectional
            )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, hidden=None):
        output, hidden = self.rnn(x, hidden)
        output = self.dropout(output)
        return output, hidden
    
    def get_output_size(self):
        """Get output size considering bidirectional"""
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