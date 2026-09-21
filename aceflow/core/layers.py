import torch
import torch.nn as nn

from mallow_core.nn import AdditiveAttention as CoreAdditiveAttention
from mallow_core.nn import Linear as CoreLinear
from mallow_core.nn import as_tensor

from .native import native_rnn


def _needs_autograd(value):
    if value is None:
        return False
    if isinstance(value, tuple):
        return any(_needs_autograd(item) for item in value)
    return bool(getattr(value, "requires_grad", False))


class RNNLayer(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1,
        dropout=0.0,
        rnn_type="lstm",
        bidirectional=False,
    ):
        super(RNNLayer, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        kind = rnn_type.lower()
        if kind.startswith("bi"):
            kind = kind[2:]
        self.rnn_type = kind
        self.bidirectional = bidirectional
        rnn_class = self._get_rnn_class()
        self.rnn = rnn_class(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

    def _get_rnn_class(self):
        if self.rnn_type == "rnn":
            return nn.RNN
        if self.rnn_type == "lstm":
            return nn.LSTM
        if self.rnn_type == "gru":
            return nn.GRU
        raise ValueError(f"Unsupported RNN type: {self.rnn_type}")

    def forward(self, x, hidden=None):
        if self.training or x.requires_grad or _needs_autograd(hidden):
            return self.rnn(x, hidden)
        return native_rnn(self.rnn, x, hidden, self.rnn_type)

    def get_output_size(self):
        return self.hidden_size * (2 if self.bidirectional else 1)


class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        hidden_size,
        num_layers=2,
        dropout=0.1,
        rnn_type="lstm",
        bidirectional=False,
        embedding_dim=None,
    ):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        embedding_dim = embedding_dim or hidden_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.rnn = RNNLayer(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            rnn_type=rnn_type,
            bidirectional=bidirectional,
        )

    def forward(self, x, hidden=None):
        embedded = self.dropout(self.embedding(x))
        output, hidden = self.rnn(embedded, hidden)
        return output, hidden

    def get_output_size(self):
        return self.rnn.get_output_size()


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        hidden_size,
        num_layers=2,
        dropout=0.1,
        rnn_type="lstm",
        encoder_bidirectional=False,
    ):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        encoder_factor = 2 if encoder_bidirectional else 1
        decoder_input_size = hidden_size * encoder_factor
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.rnn = RNNLayer(
            input_size=hidden_size,
            hidden_size=decoder_input_size,
            num_layers=num_layers,
            dropout=dropout,
            rnn_type=rnn_type,
            bidirectional=False,
        )
        self.out = nn.Linear(decoder_input_size, vocab_size)

    def forward(self, x, hidden):
        embedded = self.dropout(self.embedding(x))
        output, hidden = self.rnn(embedded, hidden)
        output = self.out(output)
        return output, hidden


class Attention(nn.Module):
    """Luong/Bahdanau scores for torch training; native AdditiveAttention for inference tensors."""

    def __init__(self, hidden_size, method="dot"):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.method = method
        if method == "general":
            self.attn = nn.Linear(hidden_size, hidden_size)
        elif method == "concat":
            self.attn = nn.Linear(hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))

    def forward(self, hidden, encoder_outputs):
        if self.method == "dot":
            attention_energies = torch.sum(hidden * encoder_outputs, dim=2)
        elif self.method == "general":
            energy = self.attn(encoder_outputs)
            attention_energies = torch.sum(hidden * energy, dim=2)
        else:
            hidden = hidden.unsqueeze(1).repeat(1, encoder_outputs.size(1), 1)
            energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), 2)))
            attention_energies = torch.sum(self.v * energy, dim=2)
        return torch.softmax(attention_energies, dim=1)

    @staticmethod
    def native_context(query, keys, values, w_keys, w_query, v):
        attn = CoreAdditiveAttention(CoreLinear(w_keys), CoreLinear(w_query), CoreLinear(v))
        return attn(as_tensor(query), as_tensor(keys), as_tensor(values))
