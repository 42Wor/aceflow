"""Parity checks between AceFlow torch modules and the mallow-core native kernels."""

import unittest

import torch
from aceflow.core.attention import BahdanauAttention
from aceflow.core.layers import RNNLayer


class TestNativeBackend(unittest.TestCase):
    def test_lstm_eval_matches_torch(self):
        torch.manual_seed(0)
        layer = RNNLayer(input_size=6, hidden_size=8, num_layers=1, rnn_type="lstm")
        x = torch.randn(2, 5, 6)
        layer.eval()
        with torch.no_grad():
            native_out, native_hidden = layer(x)
            torch_out, torch_hidden = layer.rnn(x)
        self.assertTrue(torch.allclose(native_out, torch_out, atol=1e-4, rtol=1e-4))
        self.assertTrue(torch.allclose(native_hidden[0], torch_hidden[0], atol=1e-4, rtol=1e-4))
        self.assertTrue(torch.allclose(native_hidden[1], torch_hidden[1], atol=1e-4, rtol=1e-4))

    def test_concat_attention_eval_matches_torch(self):
        torch.manual_seed(1)
        attn = BahdanauAttention(hidden_size=8, method="concat")
        decoder = torch.randn(3, 8)
        encoder = torch.randn(3, 7, 8)
        attn.eval()
        with torch.no_grad():
            native_ctx, native_w = attn(decoder, encoder)
            torch_ctx, torch_w = attn._torch_forward(decoder, encoder)
        self.assertTrue(torch.allclose(native_w, torch_w, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(native_ctx, torch_ctx, atol=1e-5, rtol=1e-5))


if __name__ == "__main__":
    unittest.main()
