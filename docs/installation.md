# Installation Guide

Get AceFlow up and running on your system with this comprehensive installation guide.

## 🎯 Quick Install

```bash
# Install AceFlow from source
git clone https://github.com/42Wor/aceflow.git
cd aceflow
pip install -e .
```

## 📋 Prerequisites

Before installing AceFlow, ensure you have:

- **Python**: 3.7 or higher
- **PyTorch**: 1.9 or higher
- **Operating System**: Windows, macOS, or Linux

### Checking Your Environment

```bash
# Check Python version
python --version
# Python 3.8.10 or higher

# Check if Python is available
python -c "import sys; print(sys.version_info)"
```

## 🔧 Installation Methods

### Method 1: Basic Installation (Recommended)

#### Step 1: Install PyTorch

Visit [pytorch.org](https://pytorch.org/get-started/locally/) and select your configuration:

**For CUDA (GPU support):**
```bash
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**For CPU only:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```


#### Step 2: Install AceFlow

```bash
# Clone the repository
git clone https://github.com/42Wor/aceflow.git
cd aceflow

# Install in development mode
pip install -e .
```

### Method 2: Using Conda

```bash
# Create a new conda environment
conda create -n aceflow python=3.9
conda activate aceflow

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install AceFlow
git clone https://github.com/42Wor/aceflow.git
cd aceflow
pip install -e .
```


## 🚀 Verification

After installation, verify everything works:

### Basic Verification

```python

import torch
import aceflow

print(f"✅ Python version check passed")
print(f"✅ PyTorch version: {torch.__version__}")
print(f"✅ AceFlow version: {aceflow.__version__}")

# Test CUDA availability
if torch.cuda.is_available():
    print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
else:
    print("ℹ️  CUDA not available - using CPU")

# Test basic imports
from aceflow import Seq2SeqModel
from aceflow.utils import Tokenizer
from aceflow.trainers import Trainer

print("✅ All imports successful!")
print("🎉 AceFlow installed successfully!")
```

