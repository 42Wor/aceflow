# Tokenizers Guide

AceFlow provides advanced tokenization with flexible preprocessing pipelines for handling text data in sequence-to-sequence models.

## Overview

The tokenization system consists of three main components:
1. **Preprocessor**: Text cleaning and normalization pipeline
2. **Vocabulary**: Word-to-index mapping with frequency tracking
3. **Tokenizer**: Main interface for encoding/decoding text

## Basic Usage

### Initialization

```python
from aceflow.utils import Tokenizer

# Initialize tokenizer with custom settings
tokenizer = Tokenizer(
    name="my_tokenizer",
    language="english", 
    max_length=50,
    padding="post",    # or 'pre'
    truncation="post"  # or 'pre'
)

print(tokenizer)
# Output: Tokenizer(name='my_tokenizer', vocab_size=5, language='english')
```

### Fitting on Text Data

```python
# Sample texts
texts = [
    "Hello world!",
    "This is a sample text.",
    "Another example for tokenization.",
    "Let's test the tokenizer!"
]

# Fit tokenizer on data
tokenizer.fit(texts, max_vocab_size=10000, min_freq=1)

print(f"Vocabulary size: {len(tokenizer)}")
# Output: Vocabulary size: 25
```

### Encoding and Decoding

```python
# Encode text
encoded = tokenizer.encode("Hello world!")
print(encoded)
# Output: 
# {
#   'input_ids': [1, 34, 56, 2, 0, 0, 0, 0],
#   'attention_mask': [1, 1, 1, 1, 0, 0, 0, 0],
#   'token_count': 4
# }

# Decode back to text
decoded = tokenizer.decode(encoded['input_ids'])
print(decoded)  # Output: "hello world"
```

## Preprocessing Pipeline

### Default Pipeline Steps

The tokenizer applies this preprocessing pipeline by default:

1. **HTML Decoding**: Convert HTML entities (`&amp;` → `&`)
2. **Unicode Normalization**: Normalize unicode characters
3. **Contraction Expansion**: Expand contractions (`don't` → `do not`)
4. **Special Character Cleaning**: Remove unwanted characters
5. **Whitespace Normalization**: Collapse multiple spaces
6. **Lowercasing**: Convert to lowercase

### Customizing the Pipeline

```python
from aceflow.utils import Preprocessor

# Create custom preprocessor
preprocessor = Preprocessor(language="english")

# Remove default steps
preprocessor.remove_step('expand_contractions')
preprocessor.remove_step('lowercase')

# Add custom preprocessing steps
def remove_urls(text):
    import re
    return re.sub(r'http\S+', '', text)

def remove_emojis(text):
    import re
    # Remove emojis and other non-text characters
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def custom_cleaning(text):
    # Your custom cleaning logic
    text = text.replace('&', 'and')
    text = text.replace('@', 'at')
    text = text.replace('#', '')
    return text

# Add custom steps to pipeline
preprocessor.add_step(remove_urls)
preprocessor.add_step(remove_emojis)
preprocessor.add_step(custom_cleaning)

# Use custom preprocessor with tokenizer
tokenizer = Tokenizer(name="custom_tokenizer")
tokenizer.preprocessor = preprocessor
```

### Language-Specific Preprocessing

```python
# English with custom rules
english_preprocessor = Preprocessor(language="english")
english_preprocessor.add_step(lambda x: x.replace("'s", " 's"))  # Handle possessives

# French preprocessing
french_preprocessor = Preprocessor(language="french")
# French-specific rules can be added

# Chinese/Japanese (no contraction expansion)
cjk_preprocessor = Preprocessor(language="chinese")
cjk_preprocessor.remove_step('expand_contractions')
cjk_preprocessor.remove_step('lowercase')  # Keep case for proper nouns
```

## Vocabulary Management

### Building Vocabulary

```python
# Basic vocabulary building
texts = ["hello world", "hello there", "world peace", "machine learning"]
tokenizer.fit(texts, max_vocab_size=5000, min_freq=1)

# From file
with open("corpus.txt", "r", encoding="utf-8") as f:
    texts = [line.strip() for line in f.readlines() if line.strip()]

tokenizer.fit(texts, max_vocab_size=20000, min_freq=2)

# Check vocabulary information
vocab_info = tokenizer.vocab.info()
print(f"Vocabulary name: {vocab_info['name']}")
print(f"Total words: {vocab_info['total_words']}")
print(f"Most common: {vocab_info['most_common_words']}")
```

### Vocabulary Statistics

```python
def analyze_vocabulary(tokenizer, texts):
    """Analyze vocabulary coverage and statistics"""
    total_tokens = 0
    covered_tokens = 0
    unknown_words = set()
    
    for text in texts:
        tokens = tokenizer.tokenize(text, preprocess=True)
        total_tokens += len(tokens)
        for token in tokens:
            if token in tokenizer.vocab:
                covered_tokens += 1
            else:
                unknown_words.add(token)
    
    coverage = covered_tokens / total_tokens if total_tokens > 0 else 0
    
    print(f"Vocabulary Coverage: {coverage:.2%}")
    print(f"Total Tokens: {total_tokens}")
    print(f"Covered Tokens: {covered_tokens}")
    print(f"Unknown Tokens: {total_tokens - covered_tokens}")
    print(f"Unique Unknown Words: {len(unknown_words)}")
    print(f"Sample Unknown Words: {list(unknown_words)[:10]}")

# Usage
analyze_vocabulary(tokenizer, test_texts)
```

### Handling Rare and Unknown Words

```python
# Strategy 1: Replace rare tokens with placeholders
def replace_rare_tokens(text):
    tokens = text.split()
    processed_tokens = []
    
    for token in tokens:
        if token.isdigit():
            processed_tokens.append("<NUMBER>")
        elif '@' in token and '.' in token:
            processed_tokens.append("<EMAIL>")
        elif token.startswith('http'):
            processed_tokens.append("<URL>")
        elif len(token) > 20:  # Very long tokens
            processed_tokens.append("<LONG_TOKEN>")
        else:
            processed_tokens.append(token)
    
    return ' '.join(processed_tokens)

# Add to preprocessor
preprocessor.add_step(replace_rare_tokens)

# Strategy 2: Add custom tokens to vocabulary
custom_tokens = ["<NUMBER>", "<EMAIL>", "<URL>", "<LONG_TOKEN>", "<NAME>"]
tokenizer.vocab.add_words(custom_tokens)
```

## Advanced Encoding Options

### Batch Processing

```python
# Encode multiple texts efficiently
texts = [
    "hello world",
    "good morning", 
    "how are you today?",
    "this is a longer example text for demonstration"
]

batch_encoded = tokenizer.encode_batch(texts)

for i, (text, encoded) in enumerate(zip(texts, batch_encoded)):
    print(f"Text {i+1}: {text}")
    print(f"Encoded: {encoded['input_ids']}")
    print(f"Token count: {encoded['token_count']}")
    print("-" * 40)
```

### Different Encoding Modes

```python
# Without special tokens (no <start>, <end>)
encoded = tokenizer.encode("hello world", add_special_tokens=False)
# Output: [34, 56, 0, 0, 0] instead of [1, 34, 56, 2, 0, 0]

# Without preprocessing (keep original text)
encoded = tokenizer.encode("Hello World!", preprocess=False)
# Keeps original casing and punctuation

# Return as simple list
encoded_list = tokenizer.encode("hello world", return_tensors='list')
# Output: [1, 34, 56, 2, 0, 0] instead of dictionary

# Custom max_length for specific case
encoded = tokenizer.encode("hello world", max_length=10)  # Override tokenizer's max_length
```

### Attention Masks and Padding

```python
# Understanding attention masks
encoded = tokenizer.encode("hello world")
print(f"Input IDs: {encoded['input_ids']}")
print(f"Attention Mask: {encoded['attention_mask']}")
print(f"Token Count: {encoded['token_count']}")

# Output:
# Input IDs: [1, 34, 56, 2, 0, 0, 0, 0]
# Attention Mask: [1, 1, 1, 1, 0, 0, 0, 0]
# Token Count: 4

# The attention mask indicates:
# 1 = real token, 0 = padding token
# Useful for models to ignore padding during computation
```

## Tokenization Without Encoding

```python
# Just tokenize without converting to indices
tokens = tokenizer.tokenize("Hello world! How are you?")
print(tokens)
# Output: ['hello', 'world', '!', 'how', 'are', 'you', '?']

# With and without preprocessing
tokens_raw = tokenizer.tokenize("Hello World!", preprocess=False)
print(tokens_raw)  # Output: ['Hello', 'World!']

tokens_processed = tokenizer.tokenize("Hello World!", preprocess=True)
print(tokens_processed)  # Output: ['hello', 'world', '!']
```

## Multi-Language Support

```python
# English tokenizer
english_tokenizer = Tokenizer(
    name="english",
    language="english",
    max_length=50
)

# French tokenizer  
french_tokenizer = Tokenizer(
    name="french",
    language="french", 
    max_length=50
)

# Chinese tokenizer with custom preprocessing
chinese_preprocessor = Preprocessor(language="chinese")
chinese_preprocessor.remove_step('expand_contractions')  # No contractions in Chinese
chinese_preprocessor.remove_step('lowercase')  # Keep Chinese characters as is

chinese_tokenizer = Tokenizer(
    name="chinese",
    language="chinese",
    max_length=100
)
chinese_tokenizer.preprocessor = chinese_preprocessor

# Using multiple tokenizers
def create_multilingual_tokenizers():
    tokenizers = {}
    
    for lang in ['english', 'french', 'german', 'spanish']:
        tokenizers[lang] = Tokenizer(
            name=f"{lang}_tokenizer",
            language=lang,
            max_length=60
        )
    
    return tokenizers

multilingual_tokenizers = create_multilingual_tokenizers()
```

## Saving and Loading

### Saving Tokenizers

```python
# Save tokenizer to folder
tokenizer.save("tokenizers/english_tokenizer")

# Directory structure created:
# tokenizers/english_tokenizer/
#   ├── tokenizer_config.json
#   ├── tokenizer_info.json
#   ├── preprocessor_config.json
#   └── vocabulary/
#       ├── vocabulary.pkl
#       └── vocabulary_info.json

# Verify saved tokenizer info
import json
with open("tokenizers/english_tokenizer/tokenizer_info.json", "r") as f:
    info = json.load(f)
    print(info)
```

### Loading Tokenizers

```python
from aceflow.utils import Tokenizer

# Load tokenizer from folder
loaded_tokenizer = Tokenizer.load("tokenizers/english_tokenizer")

# Verify loading
print(loaded_tokenizer.info())

# Test functionality
encoded = loaded_tokenizer.encode("test message")
print(f"Encoded: {encoded['input_ids']}")

# Load with error handling
import os
def safe_load_tokenizer(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Tokenizer path not found: {path}")
    
    try:
        tokenizer = Tokenizer.load(path)
        print(f"✅ Tokenizer loaded successfully: {tokenizer.name}")
        return tokenizer
    except Exception as e:
        print(f"❌ Error loading tokenizer: {e}")
        return None

english_tokenizer = safe_load_tokenizer("tokenizers/english_tokenizer")
```

## Best Practices

### 1. Vocabulary Size Optimization

```python
def optimize_vocabulary_size(texts, target_coverage=0.98):
    """Find optimal vocabulary size for target coverage"""
    tokenizer = Tokenizer(name="optimizer")
    
    # Test different vocabulary sizes
    vocab_sizes = [1000, 5000, 10000, 20000, 50000]
    coverages = []
    
    for size in vocab_sizes:
        tokenizer.fit(texts, max_vocab_size=size, min_freq=2)
        coverage = calculate_coverage(tokenizer, texts)
        coverages.append(coverage)
        print(f"Vocab Size: {size:5d} | Coverage: {coverage:.3f}")
        
        if coverage >= target_coverage:
            print(f"✅ Target coverage reached at vocab size {size}")
            return size
    
    print(f"⚠️  Maximum coverage: {max(coverages):.3f}")
    return vocab_sizes[coverages.index(max(coverages))]

def calculate_coverage(tokenizer, texts):
    total, covered = 0, 0
    for text in texts:
        tokens = tokenizer.tokenize(text)
        total += len(tokens)
        covered += sum(1 for token in tokens if token in tokenizer.vocab)
    return covered / total if total > 0 else 0

# Usage
optimal_size = optimize_vocabulary_size(training_texts, target_coverage=0.98)
```

### 2. Sequence Length Analysis

```python
def analyze_sequence_lengths(texts, percentiles=[50, 90, 95, 99]):
    """Analyze sequence lengths to set appropriate max_length"""
    lengths = []
    
    for text in texts:
        # Tokenize without encoding to get actual token count
        tokens = text.split()  # Simple split for analysis
        lengths.append(len(tokens))
    
    lengths.sort()
    
    print("Sequence Length Analysis:")
    print(f"Min length: {min(lengths)}")
    print(f"Max length: {max(lengths)}")
    print(f"Average length: {sum(lengths)/len(lengths):.1f}")
    
    for p in percentiles:
        idx = int(p/100 * len(lengths))
        print(f"{p}th percentile: {lengths[idx]}")
    
    # Recommended max_length (cover 95% of sequences with some buffer)
    recommended = lengths[int(0.95 * len(lengths))] + 5
    print(f"Recommended max_length: {recommended}")
    
    return recommended

# Usage
recommended_max_length = analyze_sequence_lengths(training_texts)
tokenizer.max_length = recommended_max_length
```

### 3. Handling Different Text Domains

```python
def create_domain_specific_tokenizer(domain, texts):
    """Create tokenizer optimized for specific text domain"""
    
    tokenizer = Tokenizer(
        name=f"{domain}_tokenizer",
        max_length=100  # Default, can be adjusted
    )
    
    # Domain-specific preprocessing
    preprocessor = Preprocessor(language="english")
    
    if domain == "technical":
        # Keep case for technical terms, remove some cleaning
        preprocessor.remove_step('lowercase')
        preprocessor.add_step(lambda x: x.replace("_", " "))  # Handle snake_case
        
    elif domain == "social_media":
        # Handle social media specific patterns
        preprocessor.add_step(lambda x: x.replace(" u ", " you "))
        preprocessor.add_step(lambda x: x.replace(" ur ", " your "))
        preprocessor.add_step(lambda x: x.replace(" lol ", " laughing out loud "))
        
    elif domain == "medical":
        # Medical text often has specific formatting
        preprocessor.remove_step('lowercase')  # Keep medical terminology case
        preprocessor.add_step(lambda x: re.sub(r'\d+\.\d+', '<FLOAT>', x))  # Replace floats
        
    elif domain == "legal":
        # Legal documents need careful handling
        preprocessor.remove_step('lowercase')
        preprocessor.remove_step('expand_contractions')  # Legal contractions matter
        
    tokenizer.preprocessor = preprocessor
    tokenizer.fit(texts)
    
    return tokenizer

# Usage
technical_tokenizer = create_domain_specific_tokenizer("technical", technical_texts)
social_tokenizer = create_domain_specific_tokenizer("social_media", social_texts)
```

## Troubleshooting

### Common Issues and Solutions

**1. Low Vocabulary Coverage**
```python
# Problem: Many unknown words
# Solution: Adjust vocabulary parameters
tokenizer.fit(texts, min_freq=1, max_vocab_size=50000)

# Or improve preprocessing
def better_tokenization(text):
    # Handle hyphenated words, apostrophes, etc.
    text = re.sub(r"(\w+)'(\w+)", r"\1 '\2", text)  # Split apostrophes
    text = re.sub(r"(\w+)-(\w+)", r"\1 - \2", text)  # Split hyphens
    return text

preprocessor.add_step(better_tokenization)
```

**2. Sequence Length Issues**
```python
# Problem: Texts being truncated too much
# Solution: Analyze and adjust max_length
recommended_length = analyze_sequence_lengths(texts)
tokenizer.max_length = recommended_length

# Or use dynamic padding in data loader
```

**3. Memory Issues with Large Vocabularies**
```python
# Problem: Tokenizer using too much memory
# Solution: Use smaller vocabulary or filter rare words
tokenizer.fit(texts, max_vocab_size=20000, min_freq=3)

# Or use subword tokenization (advanced)
```

**4. Special Characters Not Handled Properly**
```python
# Problem: Special characters causing issues
# Solution: Add custom cleaning
def handle_special_chars(text):
    # Replace or remove problematic characters
    text = text.replace('�', '')  # Remove replacement character
    text = text.replace('…', '...')  # Normalize ellipsis
    text = ''.join(char for char in text if ord(char) < 65536)  # Remove very rare chars
    return text

preprocessor.add_step(handle_special_chars)
```

### Debugging Tokenization

```python
def debug_tokenization(tokenizer, text):
    """Debug each step of tokenization"""
    print(f"Original text: {text}")
    
    # Test preprocessing steps
    preprocessed = text
    for step in tokenizer.preprocessor.pipeline:
        preprocessed = step(preprocessed)
        print(f"After {step.__name__}: {preprocessed}")
    
    # Tokenize
    tokens = tokenizer.tokenize(text, preprocess=True)
    print(f"Tokens: {tokens}")
    
    # Encode
    encoded = tokenizer.encode(text)
    print(f"Encoded IDs: {encoded['input_ids']}")
    print(f"Attention Mask: {encoded['attention_mask']}")
    
    # Decode
    decoded = tokenizer.decode(encoded['input_ids'])
    print(f"Decoded: {decoded}")

# Usage
debug_tokenization(tokenizer, "Hello world! Don't worry.")
```

## Performance Tips

```python
# Batch processing for better performance
def process_large_dataset(texts, tokenizer, batch_size=1000):
    """Process large datasets in batches"""
    all_encoded = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_encoded = tokenizer.encode_batch(batch_texts)
        all_encoded.extend(batch_encoded)
        
        if i % 5000 == 0:
            print(f"Processed {i}/{len(texts)} texts")
    
    return all_encoded

# Cache processed results
import pickle
def get_cached_encodings(texts, tokenizer, cache_file):
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    encodings = tokenizer.encode_batch(texts)
    
    with open(cache_file, 'wb') as f:
        pickle.dump(encodings, f)
    
    return encodings
```

This comprehensive guide covers all aspects of AceFlow tokenizers from basic usage to advanced optimization techniques. The tokenizer system is designed to be flexible and powerful for handling various text processing scenarios in sequence-to-sequence models.