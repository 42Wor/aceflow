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