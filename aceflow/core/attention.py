import torch
import torch.nn as nn
import torch.nn.functional as F

from mallow_core.nn import Linear as CoreLinear
from mallow_core.nn import MultiHeadAttention as CoreMultiHeadAttention
from mallow_core.nn import as_tensor, scaled_dot_product_attention

from .native import native_bahdanau
from .layers import RNNLayer


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size, method="concat"):
        super(BahdanauAttention, self).__init__()
        self.hidden_size = hidden_size
        self.method = method
        if method == "concat":
            self.W1 = nn.Linear(hidden_size, hidden_size, bias=False)
            self.W2 = nn.Linear(hidden_size, hidden_size, bias=False)
            self.V = nn.Linear(hidden_size, 1, bias=False)
        elif method == "general":
            self.attn = nn.Linear(hidden_size, hidden_size)

    def forward(self, decoder_hidden, encoder_outputs):
        if (
            not self.training
            and not decoder_hidden.requires_grad
            and not encoder_outputs.requires_grad
        ):
            w1 = w2 = v = None
            if self.method == "concat":
                w1 = self.W1.weight
                w2 = self.W2.weight
                v = self.V.weight.reshape(-1)
            elif self.method == "general":
                w1 = self.attn.weight
            return native_bahdanau(
                decoder_hidden, encoder_outputs, self.method, w1, w2, v
            )
        return self._torch_forward(decoder_hidden, encoder_outputs)

    def _torch_forward(self, decoder_hidden, encoder_outputs):
        seq_len = encoder_outputs.size(1)
        if self.method == "concat":
            decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, seq_len, 1)
            energy = torch.tanh(self.W1(encoder_outputs) + self.W2(decoder_hidden))
            attention_scores = self.V(energy).squeeze(-1)
        elif self.method == "general":
            decoder_hidden = decoder_hidden.unsqueeze(1)
            energy = self.attn(encoder_outputs)
            attention_scores = torch.bmm(decoder_hidden, energy.transpose(1, 2)).squeeze(1)
        else:
            decoder_hidden = decoder_hidden.unsqueeze(1)
            attention_scores = torch.bmm(
                decoder_hidden, encoder_outputs.transpose(1, 2)
            ).squeeze(1)
        attention_weights = F.softmax(attention_scores, dim=1)
        context_vector = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)
        context_vector = context_vector.squeeze(1)
        return context_vector, attention_weights


class MultiHeadAttention(nn.Module):
    """Trainable torch MHA; inference can drop to mallow-core scaled-dot product."""

    def __init__(self, hidden_size, num_heads=8, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        q = self.query(query)
        k = self.key(key)
        v = self.value(value)
        if query.requires_grad or key.requires_grad or value.requires_grad:
            batch, q_len, _ = q.shape
            kv_len = k.size(1)
            head_dim = self.hidden_size // self.num_heads
            qh = q.view(batch, q_len, self.num_heads, head_dim).transpose(1, 2)
            kh = k.view(batch, kv_len, self.num_heads, head_dim).transpose(1, 2)
            vh = v.view(batch, kv_len, self.num_heads, head_dim).transpose(1, 2)
            scores = torch.matmul(qh, kh.transpose(-2, -1)) / (head_dim**0.5)
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)
            weights = self.dropout(F.softmax(scores, dim=-1))
            context = torch.matmul(weights, vh)
            context = context.transpose(1, 2).contiguous().view(batch, q_len, self.hidden_size)
            return self.output(context), weights
        outputs = []
        for b in range(q.size(0)):
            outputs.append(
                scaled_dot_product_attention(
                    as_tensor(q[b].detach().cpu().numpy()),
                    as_tensor(k[b].detach().cpu().numpy()),
                    as_tensor(v[b].detach().cpu().numpy()),
                    self.num_heads,
                    False,
                )
            )
        stacked = torch.stack(
            [torch.from_numpy(out.numpy()) for out in outputs], dim=0
        ).to(query.device)
        return self.output(stacked), None

    def to_core(self) -> CoreMultiHeadAttention:
        return CoreMultiHeadAttention(
            CoreLinear(as_tensor(self.query.weight.detach().cpu().numpy())),
            CoreLinear(as_tensor(self.key.weight.detach().cpu().numpy())),
            CoreLinear(as_tensor(self.value.weight.detach().cpu().numpy())),
            CoreLinear(as_tensor(self.output.weight.detach().cpu().numpy())),
            self.num_heads,
        )


class AttentionalDecoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        hidden_size,
        num_layers=2,
        dropout=0.1,
        rnn_type="lstm",
        attention_method="concat",
        encoder_bidirectional=False,
    ):
        super(AttentionalDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        encoder_factor = 2 if encoder_bidirectional else 1
        self.encoder_output_size = hidden_size * encoder_factor
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.attention = BahdanauAttention(self.encoder_output_size, method=attention_method)
        self.rnn = RNNLayer(
            input_size=hidden_size + self.encoder_output_size,
            hidden_size=self.encoder_output_size,
            num_layers=num_layers,
            dropout=dropout,
            rnn_type=rnn_type,
            bidirectional=False,
        )
        self.out = nn.Linear(self.encoder_output_size * 2, vocab_size)

    def forward(self, x, hidden, encoder_outputs):
        embedded = self.dropout(self.embedding(x))
        if isinstance(hidden, tuple):
            decoder_hidden = hidden[0][-1]
        else:
            decoder_hidden = hidden[-1]
        context, attention_weights = self.attention(decoder_hidden, encoder_outputs)
        rnn_input = torch.cat([embedded, context.unsqueeze(1)], dim=2)
        output, hidden = self.rnn(rnn_input, hidden)
        output = torch.cat([output, context.unsqueeze(1)], dim=2)
        output = self.out(output)
        return output, hidden, attention_weights
