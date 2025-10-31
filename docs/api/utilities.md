 **docs/api/utilities.md**

```markdown
# Utilities API Reference

AceFlow provides various utility classes and functions for data loading, serialization, and model management.

## Table of Contents

- [Data Loader](#data-loader)
- [Serialization](#serialization)
- [Translation Dataset](#translation-dataset)
- [Preprocessor](#preprocessor)
- [Vocabulary](#vocabulary)

## Data Loader

### `create_data_loader`

Creates a PyTorch DataLoader for sequence-to-sequence tasks.

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
src_tokenizer = Tokenizer(name="english")
tgt_tokenizer = Tokenizer(name="french")

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
    src_texts = batch['src_text']     # Original source texts
    tgt_texts = batch['tgt_text']     # Original target texts
```

## Serialization

### `AceModelSerializer`

Handles saving and loading models in AceFlow's custom `.ace` format.

```python
from aceflow.utils import AceModelSerializer

serializer = AceModelSerializer()
```

#### Methods

##### `save_model(model, filepath)`

Saves a Seq2SeqModel to `.ace` format.

```python
serializer.save_model(
    model,          # Seq2SeqModel instance to save
    filepath        # Path where to save the model (.ace extension)
)
```

**Example:**

```python
from aceflow.utils import AceModelSerializer
from aceflow import Seq2SeqModel

# Create and train model
model = Seq2SeqModel(src_vocab_size=1000, tgt_vocab_size=1000)

# Save using serializer directly
serializer = AceModelSerializer()
serializer.save_model(model, "my_model.ace")

# Or use model's save method (recommended)
model.save("my_model.ace")  # This uses AceModelSerializer internally
```

##### `load_model(filepath)`

Loads a Seq2SeqModel from `.ace` format.

```python
model = serializer.load_model(filepath)  # Path to .ace file
```

**Returns:** `Seq2SeqModel` instance

**Example:**

```python
from aceflow.utils import AceModelSerializer

# Load using serializer directly
serializer = AceModelSerializer()
model = serializer.load_model("my_model.ace")

# Or use model's load method (recommended)
model = Seq2SeqModel.load("my_model.ace")  # This uses AceModelSerializer internally
```

## Translation Dataset

### `TranslationDataset`

PyTorch Dataset class for sequence-to-sequence tasks.

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

Returns a single data sample.

```python
sample = dataset[0]  # Get first sample
```

**Returns:** Dictionary with keys:
- `'src'`: `torch.Tensor` - Encoded source sequence
- `'tgt'`: `torch.Tensor` - Encoded target sequence  
- `'src_text'`: `str` - Original source text
- `'tgt_text'`: `str` - Original target text

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
for batch in dataloader:
    src_sequences = batch['src']      # Tensor: [32, 40]
    tgt_sequences = batch['tgt']      # Tensor: [32, 40]
    src_texts = batch['src_text']     # List of 32 source texts
    tgt_texts = batch['tgt_text']     # List of 32 target texts
    
    # Use in training...
```

## Preprocessor

### `Preprocessor`

Text preprocessing pipeline with configurable steps.

```python
from aceflow.utils import Preprocessor

preprocessor = Preprocessor(
    language="english"  # Language for preprocessing (default: "english")
)
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `language` | `str` | Language identifier for language-specific processing |

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `pipeline` | `List[Callable]` | List of preprocessing functions |
| `config` | `Dict` | Preprocessor configuration |
| `language` | `str` | Current language setting |

#### Methods

##### `process(text)`

Process a single text through the preprocessing pipeline.

```python
cleaned_text = preprocessor.process("Hello    WORLD!")
# Returns: "hello world"
```

##### `process_batch(texts)`

Process a batch of texts.

```python
texts = ["Hello WORLD!", "How ARE you?"]
cleaned_texts = preprocessor.process_batch(texts)
# Returns: ["hello world", "how are you"]
```

##### `add_step(step, position=None)`

Add a preprocessing step to the pipeline.

```python
def custom_cleaner(text):
    return text.replace('&', 'and')

preprocessor.add_step(custom_cleaner)  # Add to end
preprocessor.add_step(custom_cleaner, position=0)  # Add to beginning
```

##### `remove_step(step_name)`

Remove a step from the pipeline by function name.

```python
preprocessor.remove_step('lowercase')  # Remove lowercase step
```

##### `set_language(language)`

Set the language for language-specific processing.

```python
preprocessor.set_language("french")
```

##### `info()`

Get information about the preprocessor.

```python
info = preprocessor.info()
print(info)
# {
#   'language': 'english',
#   'pipeline_steps': ['decode_html', 'normalize_unicode', ...],
#   'has_contractions_lib': True,
#   'description': 'Preprocessor for english with 6 steps'
# }
```

#### Built-in Preprocessing Steps

1. **`decode_html`**: Decodes HTML entities
2. **`normalize_unicode`**: Normalizes unicode characters  
3. **`expand_contractions`**: Expands contractions (requires `contractions` library)
4. **`clean_special_chars`**: Removes special characters
5. **`normalize_whitespace`**: Normalizes whitespace
6. **`lowercase`**: Converts to lowercase

#### Example

```python
from aceflow.utils import Preprocessor

# Create custom preprocessor
preprocessor = Preprocessor(language="english")

# Remove default steps
preprocessor.remove_step('lowercase')
preprocessor.remove_step('expand_contractions')

# Add custom steps
def remove_numbers(text):
    import re
    return re.sub(r'\d+', '', text)

def capitalize_first(text):
    return text.capitalize()

preprocessor.add_step(remove_numbers)
preprocessor.add_step(capitalize_first)

# Use preprocessor
text = "hello 123 world!"
cleaned = preprocessor.process(text)
print(cleaned)  # "Hello world!"

# Process batch
texts = ["test 123", "another 456 example"]
cleaned_batch = preprocessor.process_batch(texts)
print(cleaned_batch)  # ["Test", "Another example"]
```

## Vocabulary

### `Vocabulary`

Manages word-to-index mapping and vocabulary statistics.

```python
from aceflow.utils import Vocabulary

vocab = Vocabulary(name="my_vocab")
```

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | Vocabulary name for identification |

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `word2idx` | `Dict[str, int]` | Word to index mapping |
| `idx2word` | `Dict[int, str]` | Index to word mapping |
| `word_freq` | `Counter` | Word frequency counter |
| `special_tokens` | `Dict[str, int]` | Special tokens mapping |
| `config` | `Dict` | Vocabulary configuration |

#### Methods

##### `add_word(word, freq=1)`

Add a single word to the vocabulary.

```python
vocab.add_word("hello")
vocab.add_word("world", freq=5)  # With frequency
```

##### `add_words(words)`

Add multiple words to the vocabulary.

```python
vocab.add_words(["hello", "world", "test"])
```

##### `build_from_texts(texts, max_size=50000, min_freq=2)`

Build vocabulary from a list of texts.

```python
vocab.build_from_texts(
    texts,          # List of texts to build vocabulary from
    max_size=10000, # Maximum vocabulary size (default: 50000)
    min_freq=1      # Minimum word frequency (default: 2)
)
```

##### `encode_word(word)`

Encode a single word to its index.

```python
index = vocab.encode_word("hello")  # Returns integer index
```

##### `decode_idx(idx)`

Decode a single index to its word.

```python
word = vocab.decode_idx(42)  # Returns word string
```

##### `get_frequency(word)`

Get the frequency of a word in the vocabulary.

```python
freq = vocab.get_frequency("hello")  # Returns integer frequency
```

##### `most_common(n=10)`

Get the n most common words.

```python
common_words = vocab.most_common(20)  # Returns list of words
```

##### `save(folder_path)`

Save vocabulary to a folder.

```python
vocab.save("vocabularies/english_vocab")
```

##### `load(folder_path)`

Load vocabulary from a folder.

```python
vocab = Vocabulary.load("vocabularies/english_vocab")
```

##### `info()`

Get vocabulary information.

```python
info = vocab.info()
print(info)
# {
#   'name': 'my_vocab',
#   'total_words': 1500,
#   'special_tokens': ['<pad>', '<start>', '<end>', '<unk>', '<mask>'],
#   'most_common_words': ['the', 'and', 'to', ...],
#   'config': {...}
# }
```

#### Special Tokens

The vocabulary automatically includes these special tokens:

| Token | Index | Description |
|-------|-------|-------------|
| `<pad>` | 0 | Padding token |
| `<start>` | 1 | Start of sequence |
| `<end>` | 2 | End of sequence |
| `<unk>` | 3 | Unknown word |
| `<mask>` | 4 | Mask token (for future use) |

#### Example

```python
from aceflow.utils import Vocabulary

# Create vocabulary
vocab = Vocabulary(name="english_vocab")

# Build from texts
texts = ["hello world", "hello there", "world peace", "test example"]
vocab.build_from_texts(texts, max_size=1000, min_freq=1)

# Check vocabulary
print(f"Vocabulary size: {len(vocab)}")
print(f"Most common words: {vocab.most_common(5)}")

# Encode/decode
encoded = vocab.encode_word("hello")
decoded = vocab.decode_idx(encoded)
print(f"hello -> {encoded} -> {decoded}")

# Check word frequency
freq = vocab.get_frequency("world")
print(f"Frequency of 'world': {freq}")

# Save vocabulary
vocab.save("my_vocabulary")

# Load vocabulary
loaded_vocab = Vocabulary.load("my_vocabulary")
```

## Utility Functions

### `calculate_sequence_lengths(texts, tokenizer=None)`

Calculate sequence length statistics.

```python
from aceflow.utils import calculate_sequence_lengths

stats = calculate_sequence_lengths(
    texts,          # List of texts
    tokenizer=None  # Optional tokenizer for token-level counting
)

print(stats)
# {
#   'max_length': 45,
#   'min_length': 2, 
#   'mean_length': 15.6,
#   'median_length': 14,
#   'percentile_95': 32
# }
```

### `analyze_vocabulary_coverage(tokenizer, texts)`

Analyze how well the vocabulary covers a set of texts.

```python
from aceflow.utils import analyze_vocabulary_coverage

coverage_info = analyze_vocabulary_coverage(
    tokenizer,  # Tokenizer instance
    texts       # List of texts to analyze
)

print(coverage_info)
# {
#   'coverage_rate': 0.956,
#   'total_tokens': 15000,
#   'covered_tokens': 14340,
#   'unknown_tokens': 660,
#   'unknown_words': ['rareword1', 'rareword2', ...]
# }
```

## Best Practices

### 1. Efficient Data Loading

```python
from aceflow.utils import create_data_loader

# Use appropriate batch sizes
train_loader = create_data_loader(
    train_src, train_tgt, src_tokenizer, tgt_tokenizer,
    batch_size=32,    # Adjust based on GPU memory
    max_length=50,    # Based on sequence length analysis
    shuffle=True      # For training
)

val_loader = create_data_loader(
    val_src, val_tgt, src_tokenizer, tgt_tokenizer, 
    batch_size=64,    # Larger batches for validation
    max_length=50,
    shuffle=False     # No shuffle for validation
)
```

### 2. Model Serialization

```python
from aceflow import Seq2SeqModel

# Save with descriptive names
model.save(f"models/translation_epoch_{epoch}.ace")
model.save("models/translation_best.ace")

# Load with error handling
try:
    model = Seq2SeqModel.load("models/translation_best.ace")
except FileNotFoundError:
    print("Model file not found, training new model...")
    model = Seq2SeqModel(...)
```

### 3. Vocabulary Management

```python
from aceflow.utils import Vocabulary

# Create vocabulary with appropriate sizes
vocab = Vocabulary(name="domain_specific")
vocab.build_from_texts(
    texts,
    max_size=20000,    # Based on domain complexity
    min_freq=2         # Filter very rare words
)

# Save vocabulary with model
vocab.save("vocabularies/domain_vocab")
```

## Related Documentation

- [Seq2SeqModel API](seq2seqmodel.md) - Main model class
- [Tokenizer API](tokenizer.md) - Text tokenization
- [Trainer API](trainer.md) - Model training utilities
- [Guides](../../guides/) - Practical usage guides
- [Examples](../../examples/) - Complete working examples
```