from __future__ import annotations

import json
import mmap
import os
import struct
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from ..core.tensor import Tensor

# MTF format specification constants (matching mtf-common)
MAGIC_BYTES = b"MZTENSOR"
MAGIC_FOOTER = b"MZTFOOTR"
FORMAT_VERSION = 2
ALIGNMENT_BOUNDARY = 256

# Dtype mappings
DTYPE_F32 = 0x01
DTYPE_F16 = 0x02
DTYPE_BF16 = 0x03
DTYPE_I32 = 0x04
DTYPE_I8 = 0x05

NUMPY_TO_DTYPE = {
    np.dtype(np.float32): DTYPE_F32,
    np.dtype(np.float16): DTYPE_F16,
    np.dtype(np.int32): DTYPE_I32,
    np.dtype(np.int8): DTYPE_I8,
}

DTYPE_TO_NUMPY = {
    DTYPE_F32: np.float32,
    DTYPE_F16: np.float16,
    DTYPE_I32: np.int32,
    DTYPE_I8: np.int8,
}


class MtfReader:
    """Zero-copy memory mapped MTF (Maaz Tensor Format) weight reader."""

    def __init__(self, filepath: str):
        self.filepath = filepath
        self._f = None
        self._mmap: Optional[mmap.mmap] = None
        self.metadata: Dict[str, Any] = {}
        self.tensors: Dict[str, Dict[str, Any]] = {}

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"MTF file not found: {filepath}")

        self._load()

    def _load(self):
        file_size = os.path.getsize(self.filepath)
        if file_size < 72:
            raise ValueError(f"MTF file too small ({file_size} bytes)")

        self._f = open(self.filepath, "rb")
        self._mmap = mmap.mmap(self._f.fileno(), 0, access=mmap.ACCESS_READ)

        # 1. Header (64 bytes)
        magic = self._mmap[0:8]
        if magic != MAGIC_BYTES:
            raise ValueError(f"Invalid MTF magic header: {magic}")

        version, header_size, flags, alignment = struct.unpack_from("<IIII", self._mmap, 8)
        meta_off, meta_len = struct.unpack_from("<QQ", self._mmap, 24)
        idx_off, idx_len = struct.unpack_from("<QQ", self._mmap, 40)
        tensor_count = struct.unpack_from("<I", self._mmap, 56)[0]

        # 2. Check footer
        footer = self._mmap[-8:]
        if footer != MAGIC_FOOTER:
            raise ValueError(f"Invalid MTF footer: {footer}")

        # 3. Read metadata
        if meta_len > 0:
            meta_bytes = self._mmap[meta_off:meta_off + meta_len]
            try:
                self.metadata = json.loads(meta_bytes.decode("utf-8"))
            except Exception:
                self.metadata = {}

        # 4. Read tensor index
        if idx_len > 0:
            idx_bytes = self._mmap[idx_off:idx_off + idx_len]
            index_list = json.loads(idx_bytes.decode("utf-8"))
            for entry in index_list:
                self.tensors[entry["name"]] = entry

    def __getitem__(self, name: str) -> Tensor:
        tensor = self.get_tensor(name)
        if tensor is None:
            raise KeyError(f"Tensor '{name}' not found in MTF file")
        return tensor

    def __contains__(self, name: str) -> bool:
        return name in self.tensors

    def keys(self):
        return self.tensors.keys()

    def get_tensor(self, name: str) -> Optional[Tensor]:
        if name not in self.tensors:
            return None

        meta = self.tensors[name]
        offset = meta["offset"]
        size_bytes = meta["size_bytes"]
        shape = tuple(meta["shape"])
        dtype_id = meta.get("dtype", DTYPE_F32)

        np_dtype = DTYPE_TO_NUMPY.get(dtype_id, np.float32)
        # Zero-copy memory buffer view from mmap
        buffer_slice = memoryview(self._mmap)[offset:offset + size_bytes]
        arr = np.frombuffer(buffer_slice, dtype=np_dtype).reshape(shape)

        return Tensor(arr)

    def to_dict(self) -> Dict[str, Tensor]:
        """Load all tensors into a state dictionary."""
        return {k: self[k] for k in self.tensors}

    def close(self):
        if self._mmap is not None:
            self._mmap.close()
            self._mmap = None
        if self._f is not None:
            self._f.close()
            self._f = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def save_mtf(
    filepath: str,
    tensors: Dict[str, Union[Tensor, np.ndarray, Any]],
    metadata: Optional[Dict[str, Any]] = None,
) -> int:
    """Save a dictionary of tensors to an MTF (Maaz Tensor Format) file with 256-byte alignment.
    
    Args:
        filepath: Target .mtf file path
        tensors: Dict of name -> Tensor or numpy ndarray or torch.Tensor
        metadata: Optional dictionary of metadata to store in the header
    
    Returns:
        Total bytes written.
    """
    if metadata is None:
        metadata = {}

    index = []
    # Header placeholder (64 bytes)
    header_size = 64
    
    # Metadata json
    meta_json = json.dumps(metadata).encode("utf-8")
    meta_len = len(meta_json)
    
    # Pre-process tensors and calculate sizes
    tensors_data = {}
    total_data_bytes = 0
    
    for name, t in tensors.items():
        if hasattr(t, "numpy"):
            arr = t.numpy()
        elif hasattr(t, "data") and isinstance(t.data, np.ndarray):
            arr = t.data
        elif isinstance(t, np.ndarray):
            arr = t
        else:
            arr = np.array(t, dtype=np.float32)

        # Ensure contiguous
        if not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr)

        dtype_id = NUMPY_TO_DTYPE.get(arr.dtype, DTYPE_F32)
        if arr.dtype not in NUMPY_TO_DTYPE:
            arr = arr.astype(np.float32)
            dtype_id = DTYPE_F32

        raw_bytes = arr.tobytes()
        tensors_data[name] = (arr.shape, dtype_id, raw_bytes)

    # Layout offsets
    # 0..64 : Header
    # 64 .. 64 + meta_len : Metadata
    cur_offset = header_size + meta_len

    # Align to 256 boundary for first tensor
    remainder = cur_offset % ALIGNMENT_BOUNDARY
    if remainder != 0:
        cur_offset += (ALIGNMENT_BOUNDARY - remainder)
    
    tensor_offsets = {}
    for name, (shape, dtype_id, raw_bytes) in tensors_data.items():
        # Align each tensor to 256 bytes
        rem = cur_offset % ALIGNMENT_BOUNDARY
        if rem != 0:
            cur_offset += (ALIGNMENT_BOUNDARY - rem)
        
        tensor_offsets[name] = cur_offset
        size_bytes = len(raw_bytes)
        cur_offset += size_bytes
        
        index.append({
            "name": name,
            "dtype": dtype_id,
            "shape": list(shape),
            "offset": tensor_offsets[name],
            "size_bytes": size_bytes,
        })

    # Tensor index at cur_offset
    idx_off = cur_offset
    idx_json = json.dumps(index).encode("utf-8")
    idx_len = len(idx_json)
    cur_offset += idx_len

    meta_off = header_size

    # Pack 64-byte Header
    # magic: [u8; 8]
    # format_version: u32, header_size: u32, flags: u32, alignment: u32
    # meta_offset: u64, meta_length: u64
    # idx_offset: u64, idx_length: u64
    # tensor_count: u32, reserved: [u8; 12]
    header = struct.pack(
        "<8sIIIIQQQQI12s",
        MAGIC_BYTES,
        FORMAT_VERSION,
        header_size,
        0,  # flags
        ALIGNMENT_BOUNDARY,
        meta_off,
        meta_len,
        idx_off,
        idx_len,
        len(tensors),
        b"\x00" * 12
    )

    with open(filepath, "wb") as f:
        # Write header
        f.write(header)
        # Write metadata
        f.write(meta_json)
        
        # Write tensors with alignment padding
        for name, (_, _, raw_bytes) in tensors_data.items():
            target_pos = tensor_offsets[name]
            cur_pos = f.tell()
            if target_pos > cur_pos:
                f.write(b"\x00" * (target_pos - cur_pos))
            f.write(raw_bytes)

        # Write index
        cur_pos = f.tell()
        if idx_off > cur_pos:
            f.write(b"\x00" * (idx_off - cur_pos))
        f.write(idx_json)

        # Write 8-byte footer
        f.write(MAGIC_FOOTER)

    return os.path.getsize(filepath)


def load_mtf(filepath: str) -> MtfReader:
    """Open and parse an MTF model file."""
    return MtfReader(filepath)


def load_model(filepath: str) -> MtfReader:
    """High-level model loader for MTF format."""
    return load_mtf(filepath)


def save_model(
    filepath: str,
    model_or_weights: Any,
    metadata: Optional[Dict[str, Any]] = None,
) -> int:
    """High-level model saver for MTF format."""
    if hasattr(model_or_weights, "state_dict"):
        weights = model_or_weights.state_dict()
    elif isinstance(model_or_weights, dict):
        weights = model_or_weights
    else:
        raise ValueError("model_or_weights must be a model with state_dict() or a dictionary of tensors.")
    return save_mtf(filepath, weights, metadata=metadata)


def convert_to_mtf(source_path: str, output_path: str) -> int:
    """Convert PyTorch .pt/.bin/.safetensors weights to .mtf with 256-byte alignment."""
    if source_path.endswith(".pt") or source_path.endswith(".bin"):
        try:
            import torch
            state_dict = torch.load(source_path, map_location="cpu", weights_only=False)
            if "model" in state_dict and isinstance(state_dict["model"], dict):
                state_dict = state_dict["model"]
            return save_mtf(output_path, state_dict)
        except Exception as e:
            raise RuntimeError(f"Failed to convert PyTorch checkpoint: {e}")
    else:
        raise NotImplementedError(f"Unsupported conversion format for {source_path}")
