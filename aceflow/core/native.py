"""mallow-core backends that replace AceFlow's retired C extensions."""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
import torch
try:
    from .._native import bahdanau_attention_forward, rnn_forward
except ImportError:
    try:
        from mallow_core import bahdanau_attention_forward, rnn_forward
    except ImportError:
        bahdanau_attention_forward = None
        rnn_forward = None

HiddenState = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]


def pack_torch_rnn(module: torch.nn.Module) -> Tuple[np.ndarray, np.ndarray]:
    """Flatten PyTorch RNN/LSTM/GRU weights into the native packed layout."""
    weights = []
    biases = []
    num_dir = 2 if module.bidirectional else 1
    for layer in range(module.num_layers):
        for direction in range(num_dir):
            suffix = f"_l{layer}" + ("_reverse" if direction == 1 else "")
            w_ih = getattr(module, f"weight_ih{suffix}")
            w_hh = getattr(module, f"weight_hh{suffix}")
            weights.append(w_ih.detach().cpu().contiguous().numpy().reshape(-1))
            weights.append(w_hh.detach().cpu().contiguous().numpy().reshape(-1))
            if module.bias:
                biases.append(
                    getattr(module, f"bias_ih{suffix}")
                    .detach()
                    .cpu()
                    .contiguous()
                    .numpy()
                    .reshape(-1)
                )
                biases.append(
                    getattr(module, f"bias_hh{suffix}")
                    .detach()
                    .cpu()
                    .contiguous()
                    .numpy()
                    .reshape(-1)
                )
            else:
                zeros = np.zeros(w_ih.shape[0], dtype=np.float32)
                biases.extend([zeros, zeros.copy()])
    return (
        np.concatenate(weights).astype(np.float32, copy=False),
        np.concatenate(biases).astype(np.float32, copy=False),
    )


def native_rnn(
    module: torch.nn.Module,
    x: torch.Tensor,
    hidden: Optional[HiddenState],
    kind: str,
) -> Tuple[torch.Tensor, HiddenState]:
    x_np = np.ascontiguousarray(x.detach().cpu().numpy(), dtype=np.float32)
    batch = x_np.shape[0]
    dirs = 2 if module.bidirectional else 1
    stacked = module.num_layers * dirs
    hidden_size = module.hidden_size
    cell_np = None
    if hidden is None:
        h_np = np.zeros((stacked, batch, hidden_size), dtype=np.float32)
        if kind == "lstm":
            cell_np = np.zeros_like(h_np)
    elif isinstance(hidden, tuple):
        h_np = np.ascontiguousarray(hidden[0].detach().cpu().numpy(), dtype=np.float32)
        cell_np = np.ascontiguousarray(hidden[1].detach().cpu().numpy(), dtype=np.float32)
    else:
        h_np = np.ascontiguousarray(hidden.detach().cpu().numpy(), dtype=np.float32)
    weights, biases = pack_torch_rnn(module)
    output, h_n, c_n = rnn_forward(
        x_np, h_np, weights, biases, kind, bool(module.bidirectional), cell_np
    )
    device = x.device
    out_t = torch.from_numpy(np.ascontiguousarray(output)).to(device)
    h_t = torch.from_numpy(np.ascontiguousarray(h_n)).to(device)
    if kind == "lstm":
        c_t = torch.from_numpy(np.ascontiguousarray(c_n)).to(device)
        return out_t, (h_t, c_t)
    return out_t, h_t


def native_bahdanau(
    decoder_hidden: torch.Tensor,
    encoder_outputs: torch.Tensor,
    method: str,
    w1: Optional[torch.Tensor] = None,
    w2: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    query = np.ascontiguousarray(decoder_hidden.detach().cpu().numpy(), dtype=np.float32)
    keys = np.ascontiguousarray(encoder_outputs.detach().cpu().numpy(), dtype=np.float32)
    w1_np = None if w1 is None else np.ascontiguousarray(w1.detach().cpu().numpy(), dtype=np.float32)
    w2_np = None if w2 is None else np.ascontiguousarray(w2.detach().cpu().numpy(), dtype=np.float32)
    v_np = None if v is None else np.ascontiguousarray(v.detach().cpu().numpy(), dtype=np.float32)
    weights, context = bahdanau_attention_forward(
        query, keys, method, w1_np, w2_np, v_np
    )
    device = decoder_hidden.device
    return (
        torch.from_numpy(np.ascontiguousarray(context)).to(device),
        torch.from_numpy(np.ascontiguousarray(weights)).to(device),
    )


# Alias for backwards compatibility
native_bahdanau_attention = native_bahdanau
