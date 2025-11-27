from setuptools import setup, find_packages, Extension 
from setuptools_rust import RustExtension
import numpy as np
import sys

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip()]

# Check if we're on Windows and adjust compilation flags
is_windows = sys.platform.startswith('win')

if is_windows:
    # Windows compilation flags (no OpenMP for simplicity)
    compile_args = ['/O2', '/GL']  # Optimization flags for Windows
    link_args = []
    macros = []
else:
    # Linux/Mac compilation flags
    compile_args = ['-O3', '-march=native', '-fopenmp']
    link_args = ['-fopenmp']
    macros = [('_OPENMP', None)]

# C extensions - only build if source files exist
extensions = []
try:
    # Check if C source files exist before adding the extension
    import os
    c_sources = [
        'aceflow/core/c_core/_rnn_ops.c',
        'aceflow/core/c_core/_rnn_extension.c'
    ]
    
    # Check if all source files exist
    if all(os.path.exists(source) for source in c_sources):
        extensions.append(
            Extension(
                'aceflow._rnn_ops',
                sources=c_sources,
                include_dirs=[np.get_include(), 'aceflow/core/c_core'],
                libraries=['m'] if not is_windows else [],  # math library (not on Windows)
                extra_compile_args=compile_args,
                extra_link_args=link_args,
                define_macros=macros
            )
        )
    else:
        print("Warning: C extension source files not found. Skipping C extension build.")
except Exception as e:
    print(f"Warning: Could not configure C extensions: {e}")

# Rust extensions - only build if Cargo.toml exists
rust_extensions = []
try:
    if os.path.exists("aceflow-core/Cargo.toml"):
        rust_extensions.append(
            RustExtension(
                "aceflow_core",
                path="aceflow-core/Cargo.toml",
                binding=pyo3.PyO3,  # Fixed: removed the incorrect reference
                native=False,
                py_limited_api=False,
                features=[],
            )
        )
    else:
        print("Warning: aceflow-core/Cargo.toml not found. Skipping Rust extension build.")
except Exception as e:
    print(f"Warning: Could not configure Rust extensions: {e}")

setup(
    name="aceflow",
    version="1.6.0",
    author="Maaz waheed",
    author_email="wwork4287@gmail.com",
    ext_modules=extensions,
    rust_extensions=rust_extensions,
    description="A Python library for building and training Seq2Seq models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/42Wor/aceflow",
    project_urls={
        "Bug Tracker": "https://github.com/42Wor/aceflow/issues",
        "Source Code": "https://github.com/42Wor/aceflow",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    keywords=[
        "seq2seq",
        "deep learning",
        "nlp",
        "machine translation",
        "artificial intelligence",
        "neural networks",
        "pytorch",
        "transformer",
    ],
    include_package_data=True,
    zip_safe=False,
    platforms="any",
)