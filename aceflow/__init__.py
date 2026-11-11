from __future__ import annotations

from .core.device import Device
from .core.tensor import Tensor
from . import audio
from . import io
from . import nn
from . import ops
from . import tokenization
from .io import load_model, save_model, load_mtf, save_mtf
from .trainers import BaseTrainer, ModelCheckpoint, EarlyStopping, ProgressLogger

try:
    from importlib.metadata import version as _version
    __version__ = _version("aceflow")
except Exception:
    __version__ = "0.2.0"

__all__ = [
    "Device",
    "Tensor",
    "ops",
    "nn",
    "audio",
    "tokenization",
    "io",
    "load_model",
    "save_model",
    "load_mtf",
    "save_mtf",
    "BaseTrainer",
    "ModelCheckpoint",
    "EarlyStopping",
    "ProgressLogger",
]
