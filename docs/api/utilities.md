
# Utilities API Reference

AceFlow provides comprehensive utility classes and functions for data loading, serialization, text processing, and vocabulary management.

## Table of Contents

- [Data Loader](#data-loader)
- [Serialization](#serialization)
- [Translation Dataset](#translation-dataset)
- [Preprocessor](#preprocessor)
- [Vocabulary](#vocabulary)
- [Tokenizers](#tokenizers)

## Data Loader

### `create_data_loader`

Creates a PyTorch DataLoader for sequence-to-sequence tasks with automatic batching and padding.

```python
from aceflow.utils import create_data_loader

train_loader = create_data_loader(
    src_texts,                    # List of source texts
    tgt_texts,                    # List of target texts
    src_tokenizer,                # Source tokenizer instance
    tgt_tokenizer,                # Target tokenizer instance
    batch_size=32,                # Batch size (default: 32)
    max_length=50,                # Maximum sequence length (default: 50)
    shuffle=True                  # Shuffle data (default: True)
)
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `src_texts` | `List[str]` | List of source sequence texts |
| `tgt_texts` | `List[str]` | List of target sequence texts |
| `src_tokenizer` | `Tokenizer` | Source language tokenizer |
| `tgt_tokenizer` | `Tokenizer` | Target language tokenizer |
| `batch_size` | `int` | Number of samples per batch |
| `max_length` | `int` | Maximum sequence length for padding/truncation |
| `shuffle` | `bool` | Whether to shuffle the data |

**Returns:** `torch.utils.data.DataLoader`

**Example:**

```python
from aceflow.utils import create_data_loader, Tokenizer

# Initialize tokenizers
src_tokenizer = Tokenizer(name="english", max_length=40)
tgt_tokenizer = Tokenizer(name="french", max_length=40)

# Fit tokenizers
src_tokenizer.fit(english_texts)
tgt_tokenizer.fit(french_texts)

# Create data loader
train_loader = create_data_loader(
    english_texts,
    french_texts,
    src_tokenizer,
    tgt_tokenizer,
    batch_size=64,
    max_length=40,
    shuffle=True
)

# Use in training loop
for batch in train_loader:
    src_sequences = batch['src']      # Shape: [batch_size, max_length]
    tgt_sequences = batch['tgt']      # Shape: [batch_size, max_length]
    src_texts = batch.get('src_text', [])     # Original source texts (if available)
    tgt_texts = batch.get('tgt_text', [])     # Original target texts (if available)
```

## Serialization

### `AceModelSerializer`

Handles saving and loading models in AceFlow's custom `.ace` format (zip-based with metadata).

```python
from aceflow.utils import AceModelSerializer

serializer = AceModelSerializer()
```

#### Methods

##### `save_model(model, filepath)`

Saves a Seq2SeqModel to `.ace` format with complete metadata.

```python
serializer.save_model(
    model,          # Seq2SeqModel instance to save
    filepath        # Path where to save the model (.ace extension)
)
```

**Saved Contents:**
- `model_weights.pt` - Model state dictionary
- `metadata.json` - Model configuration and hyperparameters
- Complete model architecture information

**Example:**

```python
from aceflow.utils import AceModelSerializer
from aceflow import Seq2SeqModel

# Create and train model
model = Seq2SeqModel(
    src_vocab_size=1000, 
    tgt_vocab_size=1000,
    hidden_size=256,
    use_attention=True
)

# Save using serializer directly
serializer = AceModelSerializer()
serializer.save_model(model, "my_model.ace")

# Or use model's save method (recommended)
model.save("my_model.ace")  # This uses AceModelSerializer internally
```

##### `load_model(filepath)`

Loads a Seq2SeqModel from `.ace` format with automatic configuration.

```python
model = serializer.load_model(filepath)  # Path to .ace file
```

**Returns:** `Seq2SeqModel` instance with restored architecture and weights

**Example:**

```python
from aceflow.utils import AceModelSerializer

# Load using serializer directly
serializer = AceModelSerializer()
model = serializer.load_model("my_model.ace")

# Or use model's load method (recommended)
model = Seq2SeqModel.load("my_model.ace")  # This uses AceModelSerializer internally

# Model is ready for inference or continued training
model.eval()
```

## Translation Dataset

### `TranslationDataset`

PyTorch Dataset class for sequence-to-sequence translation tasks with automatic tokenization and padding.

```python
from aceflow.utils import TranslationDataset

dataset = TranslationDataset(
    src_texts,          # List of source texts
    tgt_texts,          # List of target texts  
    src_tokenizer,      # Source tokenizer
    tgt_tokenizer,      # Target tokenizer
    max_length=50       # Maximum sequence length
)
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `src_texts` | `List[str]` | Source sequence texts |
| `tgt_texts` | `List[str]` | Target sequence texts |
| `src_tokenizer` | `Tokenizer` | Source language tokenizer |
| `tgt_tokenizer` | `Tokenizer` | Target language tokenizer |
| `max_length` | `int` | Maximum sequence length |

#### Methods

##### `__getitem__(idx)`

Returns a single data sample with encoded sequences.

```python
sample = dataset[0]  # Get first sample
```

**Returns:** Dictionary with keys:
- `'src'`: `torch.Tensor` - Encoded source sequence with special tokens
- `'tgt'`: `torch.Tensor` - Encoded target sequence with special tokens  
- `'src_text'`: `str` - Original source text (if provided)
- `'tgt_text'`: `str` - Original target text (if provided)

##### `__len__()`

Returns the number of samples in the dataset.

```python
num_samples = len(dataset)
```

#### Example

```python
from aceflow.utils import TranslationDataset
from torch.utils.data import DataLoader

# Create dataset
dataset = TranslationDataset(
    english_texts,
    french_texts, 
    src_tokenizer,
    tgt_tokenizer,
    max_length=40
)

# Create data loader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Iterate through batches
for batch_idx, batch in enumerate(dataloader):
    src_sequences = batch['src']      # Tensor: [32, 40]
    tgt_sequences = batch['tgt']      # Tensor: [32, 40]
    
    # Optional: Access original texts if needed
    if 'src_text' in batch:
        src_texts = batch['src_text']
        tgt_texts = batch['tgt_text']
    
    # Use in training loop...
    # outputs = model(src_sequences, tgt_sequences)
```

## Tokenizers

AceFlow provides a comprehensive tokenization system with preprocessing, vocabulary management, and serialization.

### `Tokenizer`

Main tokenizer class that combines preprocessing, vocabulary, and encoding/decoding functionality.

```python
from aceflow.utils import Tokenizer

tokenizer = Tokenizer(
    name="english_tokenizer",    # Identifier for the tokenizer
    language="english",          # Language for preprocessing
    max_length=100,              # Maximum sequence length
    padding="post",              # Padding direction: "post" or "pre"
    truncation="post"            # Truncation direction: "post" or "pre"
)
```

#### Key Features

- **Configurable preprocessing pipeline**
- **Automatic vocabulary building**
- **Batch encoding/decoding support**
- **Special tokens handling** (`<pad>`, `<start>`, `<end>`, `<unk>`)
- **Serialization to organized folder structure**

#### Methods

##### `fit(texts, max_vocab_size=50000, min_freq=2, preprocess=True)`

Build vocabulary from a list of texts.

```python
tokenizer.fit(
    texts,              # List of texts to build vocabulary from
    max_vocab_size=50000,  # Maximum vocabulary size
    min_freq=2,         # Minimum word frequency
    preprocess=True     # Apply preprocessing before building vocab
)
```

##### `encode(text, add_special_tokens=True, preprocess=True, return_tensors=None)`

Encode text to token indices.

```python
encoded = tokenizer.encode(
    "Hello world!",           # Text to encode
    add_special_tokens=True,  # Add <start> and <end> tokens
    preprocess=True,          # Apply preprocessing
    return_tensors='list'     # Return type: 'list' or None for dict
)
```

**Return formats:**
- `return_tensors='list'`: Returns list of indices
- Default: Returns dictionary with `input_ids`, `attention_mask`, `token_count`

##### `encode_batch(texts, add_special_tokens=True, preprocess=True)`

Encode a batch of texts.

```python
encoded_batch = tokenizer.encode_batch([
    "Hello world!",
    "How are you?",
    "Good morning!"
])
```

##### `decode(indices, remove_special_tokens=True, skip_padding=True)`

Decode token indices back to text.

```python
text = tokenizer.decode(
    [1, 42, 56, 2, 0, 0],     # Token indices
    remove_special_tokens=True, # Remove <start>, <end> tokens
    skip_padding=True          # Skip padding tokens
)
```

##### `tokenize(text, preprocess=True)`

Tokenize text without converting to indices.

```python
tokens = tokenizer.tokenize("Hello world! How are you?")
# Returns: ['hello', 'world', '!', 'how', 'are', 'you', '?']
```

##### `save(folder_path)`

Save tokenizer to organized folder structure.

```python
tokenizer.save("tokenizers/english_tokenizer")
```

**Creates:**
```
tokenizers/english_tokenizer/
├── tokenizer_config.json
├── tokenizer_info.json
├── preprocessor_config.json
└── vocabulary/
    ├── vocabulary.pkl
    └── vocabulary_info.json
```

##### `load(folder_path)`

Load tokenizer from folder.

```python
tokenizer = Tokenizer.load("tokenizers/english_tokenizer")
```

##### `info()`

Get comprehensive tokenizer information.

```python
info = tokenizer.info()
print(info)
```

#### Complete Example

```python
from aceflow.utils import Tokenizer

# Initialize with custom configuration
tokenizer = Tokenizer(
    name="my_tokenizer",
    language="english", 
    max_length=50,
    padding="post",
    truncation="post"
)

# Build vocabulary
texts = [
    "Hello world!",
    "This is a sample text.",
    "Another example for vocabulary building.",
    "Preprocessing will be applied automatically."
]
tokenizer.fit(texts, max_vocab_size=10000, min_freq=1)

# Encode text
encoded = tokenizer.encode("Hello world! How are you?")
print(encoded)
# {
#   'input_ids': [1, 42, 56, 23, 87, 12, 2, 0, ...],
#   'attention_mask': [1, 1, 1, 1, 1, 1, 1, 0, ...],
#   'token_count': 6
# }

# Decode back
decoded = tokenizer.decode(encoded['input_ids'])
print(decoded)  # "hello world ! how are you ?"

# Batch processing
batch_encoded = tokenizer.encode_batch([
    "First text",
    "Second example", 
    "Third sample"
])

# Save for later use
tokenizer.save("tokenizers/my_tokenizer")

# Load when needed
loaded_tokenizer = Tokenizer.load("tokenizers/my_tokenizer")
```

## Preprocessor

### `Preprocessor`

Configurable text preprocessing pipeline with language-specific processing.

```python
from aceflow.utils import Preprocessor

preprocessor = Preprocessor(
    language="english"  # Language for preprocessing (default: "english")
)
```

#### Built-in Processing Steps

1. **`decode_html`**: Decodes HTML entities (`&amp;` → `&`)
2. **`normalize_unicode`**: Normalizes unicode characters
3. **`expand_contractions`**: Expands contractions (`don't` → `do not`)
4. **`clean_special_chars`**: Removes/replaces special characters
5. **`normalize_whitespace`**: Normalizes whitespace
6. **`lowercase`**: Converts to lowercase

#### Methods

##### `process(text)`

Process a single text through the pipeline.

```python
cleaned_text = preprocessor.process("Hello    WORLD!   Don't worry.")
# Returns: "hello world do not worry"
```

##### `process_batch(texts)`

Process a batch of texts.

```python
texts = ["Hello WORLD!", "How ARE you?", "I'M fine!"]
cleaned_texts = preprocessor.process_batch(texts)
# Returns: ["hello world", "how are you", "i am fine"]
```

##### `add_step(step, position=None)`

Add a custom preprocessing step.

```python
def remove_numbers(text):
    import re
    return re.sub(r'\d+', '', text)

def custom_cleaner(text):
    return text.replace('@', 'at')

preprocessor.add_step(remove_numbers)
preprocessor.add_step(custom_cleaner, position=0)  # Add to beginning
```

##### `remove_step(step_name)`

Remove a processing step.

```python
preprocessor.remove_step('lowercase')  # Remove lowercase conversion
```

##### `set_language(language)`

Set language for language-specific processing.

```python
preprocessor.set_language("french")
```

#### Example

```python
from aceflow.utils import Preprocessor

# Create custom preprocessor pipeline
preprocessor = Preprocessor(language="english")

# Modify default pipeline
preprocessor.remove_step('expand_contractions')  # Keep contractions
preprocessor.remove_step('lowercase')            # Keep case

# Add custom steps
def remove_urls(text):
    import re
    return re.sub(r'http\S+', '', text)

def remove_extra_punctuation(text):
    import re
    return re.sub(r'[!?]{2,}', '!', text)

preprocessor.add_step(remove_urls)
preprocessor.add_step(remove_extra_punctuation)

# Use preprocessor
text = "Check this out: http://example.com!!   Don't MISS IT!!!"
cleaned = preprocessor.process(text)
print(cleaned)  # "Check this out:  Don't MISS IT!"

# Get pipeline info
info = preprocessor.info()
print(info['pipeline_steps'])
```

## Vocabulary

### `Vocabulary`

Manages word-to-index mapping with frequency tracking and special tokens.

```python
from aceflow.utils import Vocabulary

vocab = Vocabulary(name="my_vocabulary")
```

#### Special Tokens

| Token | Index | Description |
|-------|-------|-------------|
| `<pad>` | 0 | Padding token |
| `<start>` | 1 | Start of sequence token |
| `<end>` | 2 | End of sequence token |
| `<unk>` | 3 | Unknown word token |
| `<mask>` | 4 | Mask token (for future use) |

#### Methods

##### `build_from_texts(texts, max_size=50000, min_freq=2)`

Build vocabulary from text corpus.

```python
vocab.build_from_texts(
    texts,          # List of texts
    max_size=10000, # Maximum vocabulary size
    min_freq=2      # Minimum word frequency
)
```

##### `encode_word(word)`

Encode a word to its index.

```python
index = vocab.encode_word("hello")  # Returns integer
```

##### `decode_idx(idx)`

Decode an index to its word.

```python
word = vocab.decode_idx(42)  # Returns string
```

##### `most_common(n=10)`

Get most frequent words.

```python
common_words = vocab.most_common(20)
```

##### `save(folder_path)` / `load(folder_path)`

Save and load vocabulary.

```python
vocab.save("vocabularies/my_vocab")
loaded_vocab = Vocabulary.load("vocabularies/my_vocab")
```

#### Example

```python
from aceflow.utils import Vocabulary

# Create and build vocabulary
vocab = Vocabulary(name="english_vocab")
texts = ["hello world", "hello there", "test example", "another test"]
vocab.build_from_texts(texts, max_size=1000, min_freq=1)

# Usage
print(f"Vocabulary size: {len(vocab)}")
print(f"Most common: {vocab.most_common(3)}")

# Encode/decode
idx = vocab.encode_word("hello")
word = vocab.decode_idx(idx)
print(f"hello -> {idx} -> {word}")

# Check if word exists
if "world" in vocab:
    print("'world' is in vocabulary")

# Get word frequency
freq = vocab.get_frequency("test")
print(f"'test' appears {freq} times")

# Save vocabulary
vocab.save("vocabularies/english")
```

## Utility Functions

### Model Information

```python
from aceflow.trainers.training_utils import count_parameters, get_model_size

model = Seq2SeqModel(...)

# Count trainable parameters
num_params = count_parameters(model)
print(f"Trainable parameters: {num_params:,}")

# Get model size
size_str = get_model_size(model)
print(f"Model size: {size_str}")
```

### Training Visualization

```python
from aceflow.trainers.training_utils import plot_training_history

# After training
plot_training_history(history, "training_plots.png")

# Save comprehensive report
from aceflow.trainers.training_utils import save_training_report
save_training_report(history, config, "training_report.json")
```

## Best Practices

### 1. Efficient Data Pipeline

```python
from aceflow.utils import Tokenizer, create_data_loader

# Initialize tokenizers with appropriate max_length
src_tokenizer = Tokenizer(name="english", max_length=50)
tgt_tokenizer = Tokenizer(name="french", max_length=55)

# Build vocabulary on training data only
src_tokenizer.fit(train_english_texts)
tgt_tokenizer.fit(train_french_texts)

# Create efficient data loaders
train_loader = create_data_loader(
    train_english_texts, train_french_texts,
    src_tokenizer, tgt_tokenizer,
    batch_size=32,
    max_length=50,
    shuffle=True
)

val_loader = create_data_loader(
    val_english_texts, val_french_texts,
    src_tokenizer, tgt_tokenizer, 
    batch_size=64,  # Larger batches for validation
    max_length=50,
    shuffle=False   # No shuffle for consistent validation
)
```

### 2. Proper Serialization

```python
from aceflow import Seq2SeqModel

# Save models with versioning
model.save(f"models/translation_epoch_{epoch:03d}.ace")
model.save("models/translation_best.ace")

# Save tokenizers with models
src_tokenizer.save("tokenizers/english_tokenizer")
tgt_tokenizer.save("tokenizers/french_tokenizer")

# Load with error handling
try:
    model = Seq2SeqModel.load("models/translation_best.ace")
    src_tokenizer = Tokenizer.load("tokenizers/english_tokenizer")
    tgt_tokenizer = Tokenizer.load("tokenizers/french_tokenizer")
except FileNotFoundError as e:
    print(f"Failed to load: {e}")
    # Initialize new model/tokenizers
```

### 3. Memory Management

```python
# Use appropriate batch sizes based on available memory
batch_size = 16  # Reduce if you get OOM errors

# Clear cache periodically during training
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Use gradient checkpointing for large models
model.enable_gradient_checkpointing()
```

## Related Documentation

- [Seq2SeqModel API](seq2seqmodel.md) - Main model class documentation
- [Trainer API](trainer.md) - Training utilities and callbacks
- [Examples](../../examples/) - Complete working examples
- [Guides](../../guides/) - Practical usage guides and tutorials

## Support

For issues and questions:
1. Check the [examples](../../examples/) directory
2. Review existing [GitHub issues](https://github.com/42Wor/aceflow/issues)
3. Create a new issue with reproducible code
```