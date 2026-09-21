import os
from setuptools import setup

try:
    from setuptools_rust import Binding, RustExtension
except ImportError:
    RustExtension = None
    Binding = None

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
        print(f"Warning: {cargo_path} not found. Skipping optional Rust beam-search build.")

setup(rust_extensions=rust_extensions)
