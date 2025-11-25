from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip()]

setup(
    name="aceflow",
    version="1.6.0",
    author="Maaz waheed",
    author_email="wwork4287@gmail.com",
    description="A Python library for building and training Seq2Seq models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/42Wor/aceflow",  # Add your repository URL
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