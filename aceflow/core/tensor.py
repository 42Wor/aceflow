from __future__ import annotations

from typing import Any, List, Optional, Tuple, Union
import numpy as np

try:
    from .. import _native as _C
except ImportError:
    _C = None

from .device import Device


class Tensor:
    """High-performance Tensor abstraction backed by native AceFlow engine."""

    def __init__(self, data: Union[np.ndarray, List[Any], Tensor], device: Optional[Union[Device, str]] = None):
        if isinstance(data, Tensor):
            self._native = data._native
            self._device = data._device
            return

        target_dev = Device(device)
        self._device = target_dev

        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=np.float32)
        elif data.dtype != np.float32:
            data = data.astype(np.float32)

        if not data.flags["C_CONTIGUOUS"]:
            data = np.ascontiguousarray(data)

        if _C is not None:
            if target_dev._handle is not None:
                self._native = _C.Tensor(data, shape=list(data.shape), device=target_dev._handle)
            else:
                self._native = _C.Tensor(data, shape=list(data.shape))
        else:
            self._native = None
            self._data = data

    @property
    def shape(self) -> List[int]:
        if self._native is not None:
            return list(self._native.shape)
        return list(self._data.shape)

    @property
    def device(self) -> Device:
        return self._device

    def to(self, device: Union[Device, str]) -> Tensor:
        target_dev = Device(device)
        if self._native is not None and target_dev._handle is not None:
            new_native = self._native.to(target_dev._handle)
            res = Tensor.__new__(Tensor)
            res._native = new_native
            res._device = target_dev
            return res
        return Tensor(self.numpy(), target_dev)

    def numpy(self) -> np.ndarray:
        if self._native is not None:
            return self._native.to_numpy()
        return self._data

    @classmethod
    def zeros(cls, shape: List[int], device: Optional[Union[Device, str]] = None) -> Tensor:
        target_dev = Device(device)
        if _C is not None and target_dev._handle is not None:
            native_t = _C.Tensor.zeros(shape, target_dev._handle)
            res = cls.__new__(cls)
            res._native = native_t
            res._device = target_dev
            return res
        return cls(np.zeros(shape, dtype=np.float32), target_dev)

    @classmethod
    def ones(cls, shape: List[int], device: Optional[Union[Device, str]] = None) -> Tensor:
        target_dev = Device(device)
        if _C is not None and target_dev._handle is not None:
            native_t = _C.Tensor.ones(shape, target_dev._handle)
            res = cls.__new__(cls)
            res._native = native_t
            res._device = target_dev
            return res
        return cls(np.ones(shape, dtype=np.float32), target_dev)

    def matmul(self, other: Tensor) -> Tensor:
        if self._native is not None and other._native is not None:
            native_out = self._native.matmul(other._native)
            res = Tensor.__new__(Tensor)
            res._native = native_out
            res._device = self._device
            return res
        return Tensor(np.matmul(self.numpy(), other.numpy()), self._device)

    def __matmul__(self, other: Tensor) -> Tensor:
        return self.matmul(other)

    def __add__(self, other: Union[Tensor, float]) -> Tensor:
        if isinstance(other, Tensor) and self._native is not None and other._native is not None:
            if self.shape == other.shape:
                res = Tensor.__new__(Tensor)
                res._native = self._native.add(other._native)
                res._device = self._device
                return res
            elif hasattr(self._native, "add_bias") and len(other.shape) == 1 and self.shape[-1] == other.shape[0]:
                res = Tensor.__new__(Tensor)
                res._native = self._native.add_bias(other._native)
                res._device = self._device
                return res
        arr = other.numpy() if isinstance(other, Tensor) else other
        return Tensor(self.numpy() + arr, self._device)

    def __repr__(self) -> str:
        return f"af.Tensor(shape={self.shape}, device='{self.device}')"
