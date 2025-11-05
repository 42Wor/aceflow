I'll update the comprehensive file and class listing to reflect the current AceFlow project structure with the enhanced trainer system.

## Updated Project Structure

```
aceflow/
├── setup.py
├── requirements.txt
├── README.md
├── docs/
│   └── api/
│       └── utilities.md
├── examples/
│   ├── basic_translation.py
│   ├── fast_translation.py
│   ├── rnn_comparison.py
│   ├── enhanced_tokenizer_demo.py
│   ├── enhanced_training.py
│   └── clean_training.py
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
    └── trainers/
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

### `base_trainer.py`
**Classes:**
- `BaseTrainer` - Foundation training class with common functionality

**Methods in BaseTrainer:**
- `__init__()` - Initialize trainer with comprehensive configuration
- `_setup_device()` - Auto-detect and setup training device
- `_safe_print()` - Print without breaking tqdm progress bars
- `print_table_header()` - Display professional training table header
- `_get_table_row_str()` - Format table row with metrics
- `check_early_stopping()` - Early stopping logic
- `train()` - Main training loop with progress tracking
- `save_training_history()` - Save training metrics to JSON
- `load_training_history()` - Load training history
- `get_best_epoch()` - Find best performing epoch
- `get_learning_rate()` - Get current learning rate
- `set_learning_rate()` - Set learning rate
- `backward_pass()` - Mixed precision backward pass
- `optimizer_step()` - Optimizer step with AMP support
- `clip_gradients()` - Gradient clipping

### `seq2seq_trainer.py`
**Classes:**
- `Seq2SeqTrainer` - Specialized trainer for sequence-to-sequence models

**Methods in Seq2SeqTrainer:**
- `__init__()` - Initialize with Seq2Seq-specific parameters
- `train_epoch()` - Training epoch with teacher forcing
- `validate_epoch()` - Validation epoch without teacher forcing
- `_forward_pass()` - Model forward pass
- `_compute_metrics()` - Calculate loss and accuracy
- `set_teacher_forcing_ratio()` - Adjust teacher forcing
- `translate_batch()` - Batch translation for inspection

### `callback.py`
**Classes:**
- `Callback` - Base callback class
- `CallbackHandler` - Manages multiple callbacks
- `ModelCheckpoint` - Automatic model saving
- `LearningRateScheduler` - LR scheduling integration
- `EarlyStopping` - Early stopping implementation
- `ProgressLogger` - Training progress logging

**Methods in Callback:**
- `on_train_begin()` - Called when training starts
- `on_train_end()` - Called when training ends
- `on_epoch_begin()` - Called at start of each epoch
- `on_epoch_end()` - Called at end of each epoch
- `on_batch_begin()` - Called at start of each batch
- `on_batch_end()` - Called at end of each batch

### `metrics.py`
**Classes:**
- `Metric` - Base metric class
- `MetricTracker` - Tracks multiple metrics
- `AccuracyMetric` - Accuracy calculation
- `LossMetric` - Loss tracking

**Methods in MetricTracker:**
- `add_metric()` - Add new metric
- `update()` - Update metric value
- `compute_all()` - Compute all metrics
- `reset_all()` - Reset all metrics

### `training_utils.py`
**Functions:**
- `plot_training_history()` - Plot training curves
- `save_training_report()` - Save comprehensive training report
- `count_parameters()` - Count trainable parameters
- `get_model_size()` - Calculate model size in MB
- `setup_mixed_precision()` - Setup AMP training

## Main Package (`aceflow/`)

### `__init__.py`
**Exports:**
- `Seq2SeqModel` from core
- `Tokenizer`, `Vocabulary`, `Preprocessor` from utils
- `Seq2SeqTrainer`, `BaseTrainer` from trainers
- `ModelCheckpoint`, `EarlyStopping`, `ProgressLogger` from trainers
- `create_data_loader` from utils

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

### `examples/enhanced_training.py`
**Functions:**
- `demo_enhanced_training()` - Comprehensive training demonstration

### `examples/clean_training.py`
**Functions:**
- Clean training example without optional dependencies

## Configuration Files

### `setup.py`
**Functions:**
- Package setup configuration

### `requirements.txt`
**Lists:**
- All Python dependencies including optional ones

## Documentation Files

### `docs/api/utilities.md`
**Content:**
- Comprehensive API reference for all utility classes
- Usage examples and best practices
- Method documentation and parameters

## Complete Function/Class List

**Main Classes (25+):**
1. `Seq2SeqModel` - Main sequence-to-sequence model
2. `RNNLayer` - Unified RNN layer abstraction
3. `Encoder` - Encoder module
4. `Decoder` - Decoder module
5. `Attention` - Basic attention mechanism
6. `BahdanauAttention` - Additive attention
7. `MultiHeadAttention` - Multi-head attention
8. `AttentionalDecoder` - Attention-based decoder
9. `Vocabulary` - Vocabulary management
10. `Preprocessor` - Text preprocessing pipeline
11. `Tokenizer` - Main tokenizer class
12. `TranslationDataset` - PyTorch dataset for translation
13. `AceModelSerializer` - Model serialization
14. `BaseTrainer` - Foundation trainer class
15. `Seq2SeqTrainer` - Specialized Seq2Seq trainer
16. `Callback` - Base callback class
17. `CallbackHandler` - Callback management
18. `ModelCheckpoint` - Model saving callback
19. `LearningRateScheduler` - LR scheduling callback
20. `EarlyStopping` - Early stopping callback
21. `ProgressLogger` - Progress logging callback
22. `Metric` - Base metric class
23. `MetricTracker` - Metric management
24. `AccuracyMetric` - Accuracy calculation
25. `LossMetric` - Loss tracking

**Key Methods (60+):**
- **Model**: forward, encode, decode, beam_search, save, load, get_rnn_info
- **Layers**: various forward passes, get_output_size
- **Attention**: attention mechanisms, context calculation
- **Vocabulary**: word management, encoding, decoding, statistics
- **Preprocessor**: text cleaning, pipeline management, batch processing
- **Tokenizer**: encoding, decoding, batch processing, serialization
- **Data**: dataset management, data loader creation
- **Training**: epoch training, validation, progress tracking, metrics
- **Callbacks**: training hooks, model saving, early stopping
- **Serialization**: model saving/loading in .ace format
- **Utilities**: plotting, reporting, parameter counting

**Key Features:**
- `info()` - Configuration inspection (available in most classes)
- `save()`/`load()` - Serialization (models, tokenizers, vocabularies)
- `__call__()` - Callable interfaces (tokenizers, preprocessors)
- Professional progress reporting with tqdm integration
- Mixed precision training support
- Comprehensive error handling
- Flexible configuration system
- Production-ready serialization

## Enhanced Features in v1.5.0

### Training System
- **Professional output formatting** with clean tables
- **Robust progress tracking** without terminal breaking
- **Advanced callback system** for extensible training
- **Mixed precision training** for faster execution
- **Comprehensive metrics** tracking and visualization

### Tokenization System
- **Modular preprocessing** pipeline
- **Batch processing** support
- **Organized serialization** to folder structures
- **Language-specific** processing
- **Vocabulary statistics** and analysis

### Model Architecture
- **Multiple RNN types** support (RNN, LSTM, GRU, BiLSTM, BiGRU)
- **Attention mechanisms** (Bahdanau, Multi-Head)
- **Flexible configuration** for various tasks
- **Transfer learning** support with embedding resizing

### Production Features
- **Model serialization** in custom .ace format
- **Training history** tracking and visualization
- **Early stopping** and model checkpointing
- **Device auto-detection** (CPU/GPU/MPS)
- **Comprehensive error handling**

