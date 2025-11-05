I'll list all files, classes, and functions in the AceFlow project structure.

## Project Structure

```
aceflow/
├── setup.py
├── requirements.txt
├── README.md
├── examples/
│   ├── basic_translation.py
│   ├── fast_translation.py
│   ├── rnn_comparison.py
│   └── enhanced_tokenizer_demo.py
└── aceflow/
    ├── __init__.py
    ├── core/
    │   ├── __init__.py
    │   ├── model.py
    │   ├── layers.py
    │   └── attention.py
    ├── utils/
    │   ├── __init__.py
    │   ├── tokenizer/
    │   │   ├── __init__.py
    │   │   ├── vocabulary.py
    │   │   ├── preprocessor.py
    │   │   └── tokenizer.py
    │   ├── data_loader.py
    │   └── serialization.py
    ├── trainers/
        ├── __init__.py
        ├── base_trainer.py
        ├── seq2seq_trainer.py
        ├── callback.py
        ├── metrics.py
        └── training_utils.py
```

## Core Module (`aceflow/core/`)

### `model.py`
**Classes:**
- `Seq2SeqModel` - Main sequence-to-sequence model class

**Methods in Seq2SeqModel:**
- `__init__()` - Initialize model with configurable parameters
- `forward()` - Forward pass through encoder-decoder
- `encode()` - Encode input sequences
- `decode()` - Decode with current hidden state
- `_init_decoder_hidden()` - Initialize decoder hidden state
- `beam_search()` - Beam search for inference
- `save()` - Save model to .ace format
- `load()` - Load model from .ace format (class method)
- `get_rnn_info()` - Get RNN configuration information
- `resize_token_embeddings()` - Resize for transfer learning

### `layers.py`
**Classes:**
- `RNNLayer` - Unified RNN layer supporting multiple types
- `Encoder` - Encoder module for Seq2Seq
- `Decoder` - Decoder module for Seq2Seq
- `Attention` - Basic attention mechanism

**Methods in RNNLayer:**
- `__init__()` - Initialize with RNN type
- `forward()` - Forward pass
- `get_output_size()` - Get output size considering bidirectional

**Methods in Encoder:**
- `__init__()` - Initialize encoder
- `forward()` - Forward pass
- `get_output_size()` - Get encoder output size

**Methods in Decoder:**
- `__init__()` - Initialize decoder
- `forward()` - Forward pass

**Methods in Attention:**
- `__init__()` - Initialize attention
- `forward()` - Compute attention
- `dot_score()` - Dot product attention
- `general_score()` - General attention
- `concat_score()` - Concatenative attention

### `attention.py`
**Classes:**
- `BahdanauAttention` - Bahdanau-style additive attention
- `MultiHeadAttention` - Multi-head attention (Transformer-like)
- `AttentionalDecoder` - Decoder with attention mechanism

**Methods in BahdanauAttention:**
- `__init__()` - Initialize attention
- `forward()` - Compute attention weights and context

**Methods in MultiHeadAttention:**
- `__init__()` - Initialize multi-head attention
- `forward()` - Compute multi-head attention

**Methods in AttentionalDecoder:**
- `__init__()` - Initialize decoder with attention
- `forward()` - Forward pass with attention

## Utils Module (`aceflow/utils/`)

### Tokenizer Submodule (`aceflow/utils/tokenizer/`)

#### `vocabulary.py`
**Classes:**
- `Vocabulary` - Vocabulary management class

**Methods in Vocabulary:**
- `__init__()` - Initialize vocabulary
- `_build_special_tokens()` - Initialize special tokens
- `add_word()` - Add single word to vocabulary
- `add_words()` - Add multiple words
- `build_from_texts()` - Build vocabulary from text corpus
- `encode_word()` - Encode single word to index
- `decode_idx()` - Decode index to word
- `__len__()` - Get vocabulary size
- `__contains__()` - Check if word in vocabulary
- `get_frequency()` - Get word frequency
- `most_common()` - Get most common words
- `save()` - Save vocabulary to folder
- `load()` - Load vocabulary from folder (class method)
- `info()` - Get vocabulary information
- `__str__()` - String representation
- `__repr__()` - Representation

#### `preprocessor.py`
**Classes:**
- `Preprocessor` - Text preprocessing pipeline

**Methods in Preprocessor:**
- `__init__()` - Initialize preprocessor
- `_get_default_pipeline()` - Get default processing steps
- `decode_html()` - Decode HTML entities
- `normalize_unicode()` - Normalize unicode characters
- `expand_contractions()` - Expand contractions
- `_expand_contractions_fallback()` - Fallback contraction expansion
- `clean_special_chars()` - Clean special characters
- `normalize_whitespace()` - Normalize whitespace
- `lowercase()` - Convert to lowercase
- `remove_punctuation()` - Remove punctuation
- `remove_numbers()` - Remove numbers
- `custom_replace()` - Create custom replacement function
- `add_step()` - Add processing step
- `remove_step()` - Remove processing step
- `process()` - Process single text
- `process_batch()` - Process batch of texts
- `set_language()` - Set language
- `info()` - Get preprocessor info
- `__call__()` - Make callable
- `__str__()` - String representation

#### `tokenizer.py`
**Classes:**
- `Tokenizer` - Main tokenizer class

**Methods in Tokenizer:**
- `__init__()` - Initialize tokenizer
- `fit()` - Fit tokenizer on texts
- `encode()` - Encode text to indices
- `encode_batch()` - Encode batch of texts
- `decode()` - Decode indices to text
- `decode_batch()` - Decode batch of indices
- `tokenize()` - Tokenize without encoding
- `get_vocab()` - Get vocabulary mapping
- `get_vocab_size()` - Get vocabulary size
- `add_special_tokens()` - Add special tokens
- `save()` - Save tokenizer to folder
- `load()` - Load tokenizer from folder (class method)
- `info()` - Get tokenizer information
- `__call__()` - Make callable
- `__len__()` - Get vocabulary size
- `__str__()` - String representation
- `__repr__()` - Representation

### `data_loader.py`
**Classes:**
- `TranslationDataset` - Dataset for translation tasks

**Methods in TranslationDataset:**
- `__init__()` - Initialize dataset
- `__len__()` - Get dataset size
- `__getitem__()` - Get item by index

**Functions:**
- `create_data_loader()` - Create DataLoader from texts

### `serialization.py`
**Classes:**
- `AceModelSerializer` - Serializer for .ace format

**Methods in AceModelSerializer:**
- `__init__()` - Initialize serializer
- `save_model()` - Save model to .ace format
- `load_model()` - Load model from .ace format

## Trainers Module (`aceflow/trainers/`)

### `trainer.py`
**Classes:**
- `Trainer` - Training management class

**Methods in Trainer:**
- `__init__()` - Initialize trainer
- `train_epoch()` - Train for one epoch
- `validate_epoch()` - Validate for one epoch
- `train()` - Main training loop
- `save_training_history()` - Save training history
- `load_training_history()` - Load training history

## Main Package (`aceflow/`)

### `__init__.py`
**Exports:**
- `Seq2SeqModel` from core
- `Tokenizer`, `Vocabulary`, `Preprocessor` from utils
- `Trainer` from trainers

## Example Files

### `examples/basic_translation.py`
**Functions:**
- Basic translation example workflow

### `examples/fast_translation.py`
**Functions:**
- Fast training demonstration

### `examples/rnn_comparison.py`
**Functions:**
- `compare_rnn_types()` - Compare different RNN types
- `get_recommended_rnn_type()` - Recommend RNN type based on task

### `examples/enhanced_tokenizer_demo.py`
**Functions:**
- Enhanced tokenizer usage examples

## Configuration Files

### `setup.py`
**Functions:**
- Package setup configuration

### `requirements.txt`
**Lists:**
- All Python dependencies

## Complete Function/Class List

**Main Classes (15):**
1. `Seq2SeqModel` - Main model
2. `RNNLayer` - RNN abstraction
3. `Encoder` - Encoder module
4. `Decoder` - Decoder module
5. `Attention` - Basic attention
6. `BahdanauAttention` - Additive attention
7. `MultiHeadAttention` - Multi-head attention
8. `AttentionalDecoder` - Decoder with attention
9. `Vocabulary` - Vocabulary management
10. `Preprocessor` - Text preprocessing
11. `Tokenizer` - Main tokenizer
12. `TranslationDataset` - Dataset class
13. `AceModelSerializer` - Model serialization
14. `Trainer` - Training management
15. `TrainingCallback` - Training callback (example)

**Main Functions (40+):**
- Model: forward, encode, decode, beam_search, save, load
- Layers: various forward passes and initializations
- Attention: different attention mechanisms
- Vocabulary: word management, encoding, decoding
- Preprocessor: text cleaning, pipeline management
- Tokenizer: encoding, decoding, batch processing
- Data: dataset management, data loader creation
- Training: epoch training, validation, history management
- Serialization: model saving/loading
- Utilities: RNN type recommendation, comparison

**Key Configuration Methods:**
- `info()` - Available in most classes for configuration inspection
- `save()`/`load()` - Available for models and tokenizers
- `__call__()` - Makes tokenizers and preprocessors callable

This comprehensive structure provides:
- **Modular design** with separate components
- **Flexible configuration** for different use cases
- **Multiple RNN types** support
- **Advanced tokenization** with preprocessing pipelines
- **Professional serialization** with .ace format
- **Comprehensive training** utilities with monitoring
- **Production-ready** examples and best practices

The library supports various sequence-to-sequence tasks including machine translation, text summarization, chatbot development, and more.