from __future__ import annotations

from typing import Optional, Union

try:
    from .. import _native as _C
except ImportError:
    _C = None


class Device:
    """Device abstraction for AceFlow computations (CPU, WebGPU, CUDA)."""

    def __init__(self, spec: Optional[Union[str, Device]] = None):
        if isinstance(spec, Device):
            self._handle = spec._handle
            self.type = spec.type
            self.index = spec.index
            return

        if spec is None:
            if _C is not None:
                self._handle = _C.default_device()
                self.type = self._handle.device_type() if hasattr(self._handle, "device_type") else "cpu"
                self.index = self._handle.device_index() if hasattr(self._handle, "device_index") else 0
            else:
                self._handle = None
                self.type = "cpu"
                self.index = 0
            return

        if isinstance(spec, str):
            if ":" in spec:
                parts = spec.split(":", 1)
                self.type = parts[0].strip().lower()
                self.index = int(parts[1].strip())
            else:
                self.type = spec.strip().lower()
                self.index = 0

            if _C is not None:
                if self.type in ("gpu", "wgpu", "webgpu"):
                    self._handle = _C.Device.wgpu(self.index)
                else:
                    self._handle = _C.Device.cpu()
            else:
                self._handle = None
        else:
            self._handle = spec
            self.type = "cpu"
            self.index = 0

    @classmethod
    def cpu(cls) -> Device:
        return cls("cpu")

    @classmethod
    def wgpu(cls, index: int = 0) -> Device:
        return cls(f"wgpu:{index}")

    def __repr__(self) -> str:
        return f"Device(type='{self.type}', index={self.index})"

    def __str__(self) -> str:
        return f"{self.type}:{self.index}"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Device):
            return self.type == other.type and self.index == other.index
        if isinstance(other, str):
            return str(self) == other or self.type == other
        return False
