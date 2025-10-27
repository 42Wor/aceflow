from .core.model import Seq2SeqModel
from .utils.tokenizer import Tokenizer
from .trainers.trainer import Trainer
from importlib.metadata import version as _version

version = _version("aceflow")  # Call the function and store the result
__version__ = version
__all__ = ["Seq2SeqModel", "Tokenizer", "Trainer", "version"]