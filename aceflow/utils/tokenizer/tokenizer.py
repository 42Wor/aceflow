import os
import json
from typing import List, Dict, Union, Optional

from mallow_core import AutoTokenizer

from .vocabulary import Vocabulary
from .preprocessor import Preprocessor


class Tokenizer:
    def __init__(
        self,
        name: str = "tokenizer",
        language: str = "english",
        max_length: int = 100,
        padding: str = "post",
        truncation: str = "post",
    ):
        self.name = name
        self.vocab = Vocabulary(name=f"{name}_vocab")
        self.preprocessor = Preprocessor(language)
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        self._auto: Optional[AutoTokenizer] = None

        self.config = {
            "name": name,
            "language": language,
            "max_length": max_length,
            "padding": padding,
            "truncation": truncation,
            "vocab_size": len(self.vocab),
            "preprocessor_steps": self.preprocessor.config["steps"],
        }
        self._refresh_auto()

    def _refresh_auto(self) -> None:
        self._auto = AutoTokenizer.from_word_level(
            self.vocab.word2idx,
            special_tokens=dict(self.vocab.special_tokens),
            max_length=self.max_length,
            padding=self.padding,
            truncation=self.truncation,
        )

    def fit(
        self,
        texts: List[str],
        max_vocab_size: int = 50000,
        min_freq: int = 2,
        preprocess: bool = True,
    ) -> "Tokenizer":
        if preprocess:
            texts = self.preprocessor.process_batch(texts)
        self.vocab.build_from_texts(texts, max_vocab_size, min_freq)
        self.config.update(
            {
                "vocab_size": len(self.vocab),
                "max_vocab_size": max_vocab_size,
                "min_freq": min_freq,
            }
        )
        self._refresh_auto()
        return self

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        preprocess: bool = True,
        return_tensors: Optional[str] = None,
    ) -> Union[List[int], Dict]:
        if preprocess:
            text = self.preprocessor(text)
        indices = self._auto.encode(text, add_special_tokens=add_special_tokens)
        if return_tensors == "list":
            return indices
        return {
            "input_ids": indices,
            "attention_mask": self._auto.attention_mask(indices),
            "token_count": sum(1 for idx in indices if idx != self.vocab.special_tokens["<pad>"]),
        }

    def encode_batch(
        self,
        texts: List[str],
        add_special_tokens: bool = True,
        preprocess: bool = True,
    ) -> List[Dict]:
        if preprocess:
            texts = self.preprocessor.process_batch(texts)
        batch_ids = self._auto.encode_batch(texts, add_special_tokens=add_special_tokens)
        return [
            {
                "input_ids": ids,
                "attention_mask": self._auto.attention_mask(ids),
                "token_count": sum(1 for idx in ids if idx != self.vocab.special_tokens["<pad>"]),
            }
            for ids in batch_ids
        ]

    def decode(
        self,
        indices: List[int],
        remove_special_tokens: bool = True,
        skip_padding: bool = True,
    ) -> str:
        tokens = []
        for idx in indices:
            if skip_padding and idx == self.vocab.special_tokens["<pad>"]:
                continue
            if remove_special_tokens and idx in [
                self.vocab.special_tokens["<start>"],
                self.vocab.special_tokens["<end>"],
            ]:
                continue
            token = self.vocab.decode_idx(idx)
            if token not in ["<pad>", "<start>", "<end>"] or not remove_special_tokens:
                tokens.append(token)
        return " ".join(tokens)

    def decode_batch(
        self,
        batch_indices: List[List[int]],
        remove_special_tokens: bool = True,
        skip_padding: bool = True,
    ) -> List[str]:
        return [
            self.decode(indices, remove_special_tokens, skip_padding)
            for indices in batch_indices
        ]

    def tokenize(self, text: str, preprocess: bool = True) -> List[str]:
        if preprocess:
            text = self.preprocessor(text)
        return text.split()

    def get_vocab(self) -> Dict[str, int]:
        return self.vocab.word2idx.copy()

    def get_vocab_size(self) -> int:
        return len(self.vocab)

    def add_special_tokens(self, tokens: Dict[str, int]):
        for token, idx in tokens.items():
            if token not in self.vocab.special_tokens:
                self.vocab.special_tokens[token] = idx
                self.vocab.word2idx[token] = idx
                self.vocab.idx2word[idx] = token
        self._refresh_auto()

    def save(self, folder_path: str):
        os.makedirs(folder_path, exist_ok=True)
        with open(os.path.join(folder_path, "tokenizer_config.json"), "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
        with open(os.path.join(folder_path, "vocab.json"), "w", encoding="utf-8") as f:
            json.dump(self.vocab.word2idx, f, indent=2, ensure_ascii=False)
        vocab_folder = os.path.join(folder_path, "vocabulary")
        self.vocab.save(vocab_folder)
        with open(
            os.path.join(folder_path, "preprocessor_config.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(self.preprocessor.config, f, indent=2, ensure_ascii=False)
        with open(os.path.join(folder_path, "tokenizer_info.json"), "w", encoding="utf-8") as f:
            json.dump(self.info(), f, indent=2, ensure_ascii=False)
        print(f"Tokenizer saved to {folder_path}")

    @classmethod
    def load(cls, folder_path: str) -> "Tokenizer":
        with open(os.path.join(folder_path, "tokenizer_config.json"), "r", encoding="utf-8") as f:
            config = json.load(f)
        tokenizer = cls(
            name=config["name"],
            language=config["language"],
            max_length=config["max_length"],
            padding=config["padding"],
            truncation=config["truncation"],
        )
        vocab_folder = os.path.join(folder_path, "vocabulary")
        tokenizer.vocab = Vocabulary.load(vocab_folder)
        with open(
            os.path.join(folder_path, "preprocessor_config.json"), "r", encoding="utf-8"
        ) as f:
            tokenizer.preprocessor.config = json.load(f)
        tokenizer.config = config
        tokenizer._refresh_auto()
        return tokenizer

    def info(self) -> Dict:
        return {
            "name": self.name,
            "vocabulary": self.vocab.info(),
            "preprocessor": self.preprocessor.info(),
            "config": self.config,
            "backend": "mallow_core.AutoTokenizer",
            "description": f"Tokenizer with {len(self.vocab)} tokens, {self.max_length} max length",
        }

    def __call__(self, text: str, **kwargs) -> Union[List[int], Dict]:
        return self.encode(text, **kwargs)

    def __len__(self) -> int:
        return len(self.vocab)

    def __str__(self) -> str:
        info = self.info()
        return (
            f"Tokenizer(name='{info['name']}', vocab_size={info['vocabulary']['total_words']}, "
            f"language='{info['preprocessor']['language']}')"
        )

    def __repr__(self) -> str:
        return self.__str__()
