from __future__ import annotations

from typing import List, Optional, Union

try:
    from .. import _native as _C
except ImportError:
    _C = None


class AutoTokenizer:
    """Unified fast BPE tokenizer backed by native SIMD routines."""

    def __init__(self, native_tokenizer: Any = None):
        self._native = native_tokenizer

    @classmethod
    def from_file(cls, tokenizer_json_path: str) -> AutoTokenizer:
        if _C is not None:
            native_tok = _C.AutoTokenizer.from_file(tokenizer_json_path)
            return cls(native_tok)
        raise RuntimeError("Native tokenizer module unavailable.")

    def encode(self, text: str) -> List[int]:
        if self._native is not None:
            return self._native.encode(text)
        return [ord(c) for c in text]

    def decode(self, token_ids: List[int]) -> str:
        if self._native is not None:
            return self._native.decode(token_ids)
        return "".join(chr(i) for i in token_ids if i < 256)

