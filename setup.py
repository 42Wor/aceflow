import sys
import os
from setuptools import setup, Extension
import numpy as np

# 1. Handle Rust Import Safely
try:
    from setuptools_rust import RustExtension, Binding
except ImportError:
    import warnings
    warnings.warn("setuptools-rust not found. Rust extensions will not be built.")
    RustExtension = None
    Binding = None

# 2. Configuration for C Extensions
# Because 'numpy' is listed in [build-system] in pyproject.toml, 
# it is guaranteed to be present during the build.
include_dirs = [np.get_include(), 'aceflow/core/c_core']

# Check OS for compilation flags
is_windows = sys.platform.startswith('win')

if is_windows:
    # Windows compilation flags
    compile_args = ['/O2', '/GL']
    link_args = []
    macros = []
else:
    # Linux/Mac compilation flags (OpenMP support)
    compile_args = ['-O3', '-march=native', '-fopenmp']
    link_args = ['-fopenmp']
    macros = [('_OPENMP', None)]

# Define C Extensions
extensions = [
    Extension(
        'aceflow._rnn_ops',
        sources=[
            'aceflow/core/c_core/_rnn_ops.c',
            'aceflow/core/c_core/_rnn_extension.c'
        ],
        include_dirs=include_dirs,
        libraries=['m'] if not is_windows else [],
        extra_compile_args=compile_args,
        extra_link_args=link_args,
        define_macros=macros
    ),
    Extension(
        'aceflow._attention_ops',
        sources=[
            'aceflow/core/c_core/_attention_ops.c',
            'aceflow/core/c_core/_attention_extension.c'
        ],
        include_dirs=include_dirs,
        libraries=['m'] if not is_windows else [],
        extra_compile_args=compile_args,
        extra_link_args=link_args,
        define_macros=macros
    )
]

# Define Rust extensions
rust_extensions = []
if RustExtension:
    cargo_path = "aceflow-core/Cargo.toml"
    if os.path.exists(cargo_path):
        rust_extensions.append(
            RustExtension(
                "aceflow_core",
                path=cargo_path,
                binding=Binding.PyO3,
                native=False,
                py_limited_api=False,
                features=[],
            )
        )
    else:
        print(f"Warning: {cargo_path} not found. Skipping Rust extension build.")

# 3. Final Setup Call
# Metadata is handled by pyproject.toml, so we only pass the dynamic extensions here.
setup(
    ext_modules=extensions,
    rust_extensions=rust_extensions,
)