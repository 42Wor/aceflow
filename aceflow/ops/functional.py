from __future__ import annotations

from typing import Optional, Union
import numpy as np

try:
    from .. import _native as _C
except ImportError:
    _C = None

from ..core.tensor import Tensor


def matmul(a: Tensor, b: Tensor) -> Tensor:
    return a.matmul(b)


def rms_norm(x: Tensor, weight: Tensor, eps: float = 1e-6) -> Tensor:
    if x._native is not None and weight._native is not None:
        pass
    arr = x.numpy()
    w = weight.numpy()
    variance = np.mean(np.square(arr), axis=-1, keepdims=True)
    normed = arr / np.sqrt(variance + eps) * w
    return Tensor(normed, x.device)


def swiglu(x: Tensor, w_gate: Tensor, w_up: Tensor, w_down: Tensor) -> Tensor:
    arr = x.numpy()
    gate = np.matmul(arr, w_gate.numpy().T)
    up = np.matmul(arr, w_up.numpy().T)
    silu = gate / (1.0 + np.exp(-gate))
    inter = silu * up
    out = np.matmul(inter, w_down.numpy().T)
    return Tensor(out, x.device)


def gelu(x: Tensor) -> Tensor:
    arr = x.numpy()
    cdf = 0.5 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (arr + 0.044715 * np.power(arr, 3))))
    return Tensor(arr * cdf, x.device)


def silu(x: Tensor) -> Tensor:
    arr = x.numpy()
    return Tensor(arr / (1.0 + np.exp(-arr)), x.device)


def softmax(x: Tensor, axis: int = -1) -> Tensor:
    arr = x.numpy()
    e_x = np.exp(arr - np.max(arr, axis=axis, keepdims=True))
    return Tensor(e_x / np.sum(e_x, axis=axis, keepdims=True), x.device)

