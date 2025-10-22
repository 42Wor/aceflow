"""
AceFlow: A high-performance Seq2Seq library with a Rust core.
"""
__version__ = "0.1.0"

# Import the compiled Rust module
from . import aceflow_core

# Expose the high-level Python classes to the user
from .model import Seq2Seq

# Define what gets imported when a user does `from aceflow import *`
__all__ = [
    "aceflow_core",
    "Seq2Seq",
    "__version__",
]