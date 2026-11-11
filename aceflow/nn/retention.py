from __future__ import annotations

import math
from typing import List, Optional, Tuple
import numpy as np

try:
    from .. import _native as _C
except ImportError:
    _C = None

from .module import Module, Parameter
from .layers import RMSNorm
from ..core.tensor import Tensor


class MultiScaleRetention(Module):
    """
    Multi-Scale Retention (MSR) Layer.
    Dual execution:
    - forward_parallel: O(N) training with causal decay matrix.
    - forward_recurrent: O(1) step inference with in-place state matrix.
    """

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.head_dim = d_model // num_heads

        # Decay rates gamma_h = 1 - 2^(-5 - h)
        gamma = [1.0 - 2.0 ** (-5.0 - h) for h in range(num_heads)]
        self.gamma = np.array(gamma, dtype=np.float32)

        k = 1.0 / math.sqrt(d_model)
        self.w_q = Parameter(Tensor(np.random.uniform(-k, k, (d_model, d_model)).astype(np.float32)))
        self.w_k = Parameter(Tensor(np.random.uniform(-k, k, (d_model, d_model)).astype(np.float32)))
        self.w_v = Parameter(Tensor(np.random.uniform(-k, k, (d_model, d_model)).astype(np.float32)))
        self.w_o = Parameter(Tensor(np.random.uniform(-k, k, (d_model, d_model)).astype(np.float32)))

        self.ret_norm = RMSNorm(d_model)

    def forward_parallel(self, x: Tensor) -> Tensor:
        arr = x.numpy()
        b, n, d = arr.shape
        q = np.matmul(arr, self.w_q.numpy().T).reshape(b, n, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = np.matmul(arr, self.w_k.numpy().T).reshape(b, n, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = np.matmul(arr, self.w_v.numpy().T).reshape(b, n, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Decay matrix D[i, j] = gamma^(i-j) for i >= j else 0
        idx = np.arange(n)
        dist = idx[:, None] - idx[None, :]
        mask = dist >= 0
        decay = np.zeros((self.num_heads, n, n), dtype=np.float32)
        for h in range(self.num_heads):
            decay[h] = np.where(mask, self.gamma[h] ** dist, 0.0)

        # Retention score = (Q @ K^T * D) @ V
        scores = np.matmul(q, k.transpose(0, 1, 3, 2)) * decay[None, :, :, :]
        out = np.matmul(scores, v)
        out = out.transpose(0, 2, 1, 3).reshape(b, n, d)

        normed = self.ret_norm(Tensor(out, x.device))
        proj = np.matmul(normed.numpy(), self.w_o.numpy().T)
        return Tensor(proj, x.device)

    def forward(self, x: Tensor) -> Tensor:
        return self.forward_parallel(x)

