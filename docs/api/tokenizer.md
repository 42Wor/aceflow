
# Tokenizer API Reference

Complete API documentation for the AceFlow Tokenizer class.

## Overview

The `Tokenizer` class provides advanced text tokenization with flexible preprocessing pipelines, vocabulary management, and sequence encoding/decoding capabilities.

## Class Definition

```python
Tokenizer(
    name: str = "tokenizer",
    language: str = 'english', 
    max_length: int = 100,
    padding: str = 'post',
    truncation: str = 'post'
)
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | `"tokenizer"` | Identifier for the tokenizer |
| `language` | `str` | `'english'` | Language for preprocessing rules |
| `max_length` | `int` | `100` | Maximum sequence length |
| `padding` | `str` | `'post'` | Padding direction: `'pre'` or `'post'` |
| `truncation` | `str` | `'post'` | Truncation direction: `'pre'` or `'post'` |

## Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Tokenizer name |
| `vocab` | `Vocabulary` | Vocabulary instance |
| `preprocessor` | `Preprocessor` | Preprocessor instance |
| `max_length` | `int` | Maximum sequence length |
| `padding` | `str` | Padding direction |
| `truncation` | `str` | Truncation direction |
| `config` | `dict` | Configuration dictionary |

## Methods

### `fit`

Build vocabulary from texts.

```python
fit(
    texts: List[str],
    max_vocab_size: int = 50000,
    min_freq: int = 2,
    preprocess: bool = True
) -> Tokenizer
```

**Parameters:**
- `texts` (`List[str]`): List of text samples
- `max_vocab_size` (`int`): Maximum vocabulary size
- `min_freq` (`int`): Minimum frequency for words to include
- `preprocess` (`bool`): Whether to preprocess texts before building vocabulary

**Returns:** `Tokenizer` - Returns self for method chaining

**Example:**
```python
tokenizer = Tokenizer()
texts = ["hello world", "good morning", "how are you"]
tokenizer.fit(texts, max_vocab_size=10000, min_freq=1)
```

### `encode`

Encode text to indices.

```python
encode(
    text: str,
    add_special_tokens: bool = True,
    preprocess: bool = True,
    return_tensors: Optional[str] = None
) -> Union[List[int], Dict[str, Any]]
```

**Parameters:**
- `text` (`str`): Input text to encode
- `add_special_tokens` (`bool`): Whether to add start/end tokens
- `preprocess` (`bool`): Whether to preprocess text before encoding
- `return_tensors` (`Optional[str]`): If `'list'`, return list instead of dict

**Returns:**
- If `return_tensors='list'`: `List[int]` - List of token indices
- Otherwise: `Dict` with keys:
  - `input_ids` (`List[int]`): Token indices
  - `attention_mask` (`List[int]`): Mask (1 for real tokens, 0 for padding)
  - `token_count` (`int`): Number of non-padding tokens

**Example:**
```python
# Basic encoding
encoded = tokenizer.encode("hello world")
print(encoded)
# {'input_ids': [1, 34, 56, 2, 0, 0],
#  'attention_mask': [1, 1, 1, 1, 0, 0],
#  'token_count': 4}

# List only
indices = tokenizer.encode("hello world", return_tensors='list')
# [1, 34, 56, 2, 0, 0]

# No special tokens
encoded = tokenizer.encode("hello world", add_special_tokens=False)
# {'input_ids': [34, 56, 0, 0, 0, 0], ...}

# No preprocessing  
encoded = tokenizer.encode("Hello World!", preprocess=False)
# Keeps original casing
```

### `encode_batch`

Encode a batch of texts.

```python
encode_batch(
    texts: List[str],
    add_special_tokens: bool = True,
    preprocess: bool = True
) -> List[Dict[str, Any]]
```

**Parameters:**
- `texts` (`List[str]`): List of texts to encode
- `add_special_tokens` (`bool`): Whether to add start/end tokens
- `preprocess` (`bool`): Whether to preprocess texts

**Returns:** `List[Dict]` - List of encoded results

**Example:**
```python
texts = ["hello world", "good morning", "how are you"]
batch_encoded = tokenizer.encode_batch(texts)

for text, encoded in zip(texts, batch_encoded):
    print(f"{text} -> {encoded['input_ids']}")
```

### `decode`

Decode indices back to text.

```python
decode(
    indices: List[int],
    remove_special_tokens: bool = True,
    skip_padding: bool = True
) -> str
```

**Parameters:**
- `indices` (`List[int]`): List of token indices to decode
- `remove_special_tokens` (`bool`): Whether to remove start/end tokens
- `skip_padding` (`bool`): Whether to skip padding tokens

**Returns:** `str` - Decoded text

**Example:**
```python
indices = [1, 34, 56, 2, 0, 0]  # [<start>, hello, world, <end>, <pad>, <pad>]

# Default: remove special tokens and padding
text = tokenizer.decode(indices)
print(text)  # "hello world"

# Keep special tokens
text = tokenizer.decode(indices, remove_special_tokens=False)
print(text)  # "<start> hello world <end>"

# Keep padding
text = tokenizer.decode(indices, skip_padding=False)
print(text)  # "hello world <pad> <pad>"
```

### `decode_batch`

Decode a batch of indices.

```python
decode_batch(
    batch_indices: List[List[int]],
    remove_special_tokens: bool = True,
    skip_padding: bool = True
) -> List[str]
```

**Parameters:**
- `batch_indices` (`List[List[int]]`): Batch of index sequences
- `remove_special_tokens` (`bool`): Whether to remove special tokens
- `skip_padding` (`bool`): Whether to skip padding tokens

**Returns:** `List[str]` - List of decoded texts

**Example:**
```python
batch_indices = [
    [1, 34, 56, 2, 0, 0],
    [1, 23, 45, 2, 0, 0],
    [1, 67, 89, 12, 2, 0]
]

texts = tokenizer.decode_batch(batch_indices)
print(texts)  # ["hello world", "good morning", "how are you"]
```

### `tokenize`

Tokenize text without converting to indices.

```python
tokenize(text: str, preprocess: bool = True) -> List[str]
```

**Parameters:**
- `text` (`str`): Text to tokenize
- `preprocess` (`bool`): Whether to preprocess text first

**Returns:** `List[str]` - List of tokens

**Example:**
```python
tokens = tokenizer.tokenize("Hello, world! How are you?")
print(tokens)  # ["hello", "world", "how", "are", "you"]
```

### `get_vocab`

Get vocabulary mapping.

```python
get_vocab() -> Dict[str, int]
```

**Returns:** `Dict[str, int]` - Copy of word-to-index mapping

**Example:**
```python
vocab = tokenizer.get_vocab()
print(f"Vocabulary size: {len(vocab)}")
print(f"'hello' index: {vocab.get('hello', 'OOV')}")
```

### `get_vocab_size`

Get vocabulary size.

```python
get_vocab_size() -> int
```

**Returns:** `int` - Number of tokens in vocabulary

**Example:**
```python
size = tokenizer.get_vocab_size()
print(f"Vocabulary size: {size}")
```

### `add_special_tokens`

Add special tokens to vocabulary.

```python
add_special_tokens(tokens: Dict[str, int]) -> None
```

**Parameters:**
- `tokens` (`Dict[str, int]`): Dictionary of token -> index mappings

**Example:**
```python
# Add custom special tokens
tokenizer.add_special_tokens({
    '<custom1>': 1000,
    '<custom2>': 1001
})
```

### `save`

Save tokenizer to folder.

```python
save(folder_path: str) -> None
```

**Parameters:**
- `folder_path` (`str`): Path to save tokenizer

**Creates:**
```
folder_path/
├── tokenizer_config.json
├── tokenizer_info.json
├── preprocessor_config.json
└── vocabulary/
    ├── vocabulary.pkl
    └── vocabulary_info.json
```

**Example:**
```python
tokenizer.save("my_tokenizer")
```

### `load`

Load tokenizer from folder.

```python
@classmethod
load(cls, folder_path: str) -> Tokenizer
```

**Parameters:**
- `folder_path` (`str`): Path to saved tokenizer

**Returns:** `Tokenizer` - Loaded tokenizer instance

**Example:**
```python
from aceflow.utils import Tokenizer
tokenizer = Tokenizer.load("my_tokenizer")
```

### `info`

Get tokenizer information.

```python
info() -> Dict[str, Any]
```

**Returns:** `Dict` - Tokenizer information including vocabulary and preprocessor details

**Example:**
```python
info = tokenizer.info()
print(info)
# {
#   'name': 'my_tokenizer',
#   'vocabulary': {...},
#   'preprocessor': {...},
#   'config': {...}
# }
```

## Special Methods

### `__call__`

Make tokenizer callable for encoding.

```python
__call__(text: str, **kwargs) -> Union[List[int], Dict[str, Any]]
```

**Example:**
```python
# Equivalent to tokenizer.encode(text)
encoded = tokenizer("hello world")
```

### `__len__`

Get vocabulary size.

```python
__len__() -> int
```

**Example:**
```python
size = len(tokenizer)
print(f"Vocabulary size: {size}")
```

### `__contains__`

Check if word is in vocabulary.

```python
__contains__(word: str) -> bool
```

**Example:**
```python
if "hello" in tokenizer:
    print("'hello' is in vocabulary")
```

### `__str__` and `__repr__`

String representation.

```python
__str__() -> str
__repr__() -> str
```

**Example:**
```python
print(tokenizer)
# "Tokenizer(name='my_tokenizer', vocab_size=1500, language='english')"
```

## Properties

### Vocabulary Access

```python
# Access vocabulary directly
tokenizer.vocab.word2idx      # word -> index mapping
tokenizer.vocab.idx2word      # index -> word mapping  
tokenizer.vocab.word_freq     # word frequency counter
tokenizer.vocab.special_tokens # special tokens mapping
```

### Preprocessor Access

```python
# Access and modify preprocessor
tokenizer.preprocessor.pipeline          # list of processing steps
tokenizer.preprocessor.config           # preprocessor configuration
tokenizer.preprocessor.contractions_map # contractions mapping
```

## Examples

### Basic Usage

```python
from aceflow.utils import Tokenizer

# Initialize
tokenizer = Tokenizer(name="english", max_length=50)

# Fit on data
texts = ["hello world", "good morning", "how are you"]
tokenizer.fit(texts)

# Encode/decode
encoded = tokenizer.encode("hello world")
decoded = tokenizer.decode(encoded['input_ids'])

print(f"Encoded: {encoded}")
print(f"Decoded: {decoded}")
```

### Advanced Configuration

```python
from aceflow.utils import Tokenizer, Preprocessor

# Custom preprocessor
preprocessor = Preprocessor(language="english")
preprocessor.remove_step('lowercase')  # Keep original casing

# Custom tokenizer
tokenizer = Tokenizer(
    name="custom",
    language="english", 
    max_length=100,
    padding="pre",      # Pad at beginning
    truncation="pre"    # Truncate from beginning
)
tokenizer.preprocessor = preprocessor

# Fit with custom settings
tokenizer.fit(
    texts, 
    max_vocab_size=20000,
    min_freq=1,
    preprocess=False  # Use raw texts
)
```

### Batch Processing

```python
# Process multiple texts efficiently
texts = [
    "Hello world!",
    "How are you today?",
    "This is a longer example text for demonstration."
]

# Encode batch
batch_encoded = tokenizer.encode_batch(texts)

# Decode batch  
original_texts = tokenizer.decode_batch(
    [enc['input_ids'] for enc in batch_encoded]
)

for original, decoded in zip(texts, original_texts):
    print(f"Original: {original}")
    print(f"Decoded:  {decoded}")
    print()
```

## Error Handling

### Common Exceptions

```python
try:
    # Attempt to encode very long text
    encoded = tokenizer.encode("x " * 200)  # 200 tokens
except Exception as e:
    print(f"Error: {e}")
    # Handle sequence too long error

try:
    # Attempt to use unfitted tokenizer
    encoded = tokenizer.encode("hello")
except RuntimeError as e:
    print(f"Error: {e}")
    # Tokenizer not fitted yet
```

## Performance Tips

### 1. Batch Processing

```python
# ✅ Good: Process in batches
batch_encoded = tokenizer.encode_batch(texts)

# ❌ Avoid: Process individually
for text in texts:
    encoded = tokenizer.encode(text)  # Slower
```

### 2. Preprocessing Control

```python
# ✅ Good: Skip preprocessing if already done
preprocessed_texts = [preprocess(t) for t in texts]
tokenizer.fit(preprocessed_texts, preprocess=False)

# ❌ Avoid: Double preprocessing
tokenizer.fit(texts, preprocess=True)  # Texts already preprocessed
```

### 3. Memory Management

```python
# For large datasets, process in chunks
def fit_large_dataset(tokenizer, texts, chunk_size=10000):
    for i in range(0, len(texts), chunk_size):
        chunk = texts[i:i + chunk_size]
        tokenizer.fit(chunk, preprocess=True)
```

## See Also

- [Preprocessor API](../api/preprocessor.md) - Text preprocessing utilities
- [Vocabulary API](../api/vocabulary.md) - Vocabulary management
- [Tokenizers Guide](../guides/tokenizers.md) - Usage guide and examples
- [Training Guide](../guides/training.md) - Using tokenizers in model training
```