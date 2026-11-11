from __future__ import annotations

from typing import Optional
import numpy as np

try:
    from .. import _native as _C
except ImportError:
    _C = None

from ..core.tensor import Tensor


def mel_filters(n_mels: int = 80, n_fft: int = 400, sample_rate: int = 16000) -> np.ndarray:
    if _C is not None:
        return _C.mel_filters(n_mels, n_fft, sample_rate)
    raise NotImplementedError("Native DSP extension not available.")


def log_mel_spectrogram(
    audio: np.ndarray,
    n_mels: int = 80,
    n_fft: int = 400,
    hop_length: int = 160,
    padding: int = 0,
) -> np.ndarray:
    if _C is not None:
        return _C.log_mel_spectrogram(audio, n_mels, n_fft, hop_length, padding)
    raise NotImplementedError("Native DSP extension not available.")


def pad_or_trim(array: np.ndarray, length: int) -> np.ndarray:
    if _C is not None:
        return _C.pad_or_trim(array, length)
    if len(array) > length:
        return array[:length]
    return np.pad(array, (0, max(0, length - len(array))))

