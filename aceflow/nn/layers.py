from __future__ import annotations

import math
from typing import Optional
import numpy as np

from .module import Module, Parameter
from ..core.tensor import Tensor


class Linear(Module):
    """Fully connected layer: y = xA^T + b."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        k = 1.0 / math.sqrt(in_features)
        w = np.random.uniform(-k, k, (out_features, in_features)).astype(np.float32)
        self.weight = Parameter(Tensor(w))

        if bias:
            b = np.random.uniform(-k, k, (out_features,)).astype(np.float32)
            self.bias = Parameter(Tensor(b))
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        out = x.matmul(Tensor(self.weight.numpy().T, x.device))
        if self.bias is not None:
            out = out + self.bias
        return out


class RMSNorm(Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = Parameter(Tensor(np.ones(d_model, dtype=np.float32)))

    def forward(self, x: Tensor) -> Tensor:
        arr = x.numpy()
        rms = np.sqrt(np.mean(np.square(arr), axis=-1, keepdims=True) + self.eps)
        normed = (arr / rms) * self.weight.numpy()
        return Tensor(normed, x.device)


class LayerNorm(Module):
    """Standard Layer Normalization."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = Parameter(Tensor(np.ones(d_model, dtype=np.float32)))
        self.bias = Parameter(Tensor(np.zeros(d_model, dtype=np.float32)))

    def forward(self, x: Tensor) -> Tensor:
        arr = x.numpy()
        mean = np.mean(arr, axis=-1, keepdims=True)
        var = np.var(arr, axis=-1, keepdims=True)
        normed = ((arr - mean) / np.sqrt(var + self.eps)) * self.weight.numpy() + self.bias.numpy()
        return Tensor(normed, x.device)


class SwiGLU(Module):
    """SwiGLU feed-forward network layer."""

    def __init__(self, d_model: int, hidden_dim: Optional[int] = None, multiple_of: int = 64):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = int(2 * (4 * d_model) / 3)
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w_gate = Linear(d_model, hidden_dim, bias=False)
        self.w_up = Linear(d_model, hidden_dim, bias=False)
        self.w_down = Linear(hidden_dim, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        gate = self.w_gate(x).numpy()
        up = self.w_up(x).numpy()
        silu = gate / (1.0 + np.exp(-gate))
        inter = Tensor(silu * up, x.device)
        return self.w_down(inter)
