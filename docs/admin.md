### Build & Publish Wheels with `maturin`

1. Build the wheel:
   ```bash
   pip install build

   python -m build
   ```
2. (Optional) Create a `wheelhouse`:
   ```bash
   mkdir -p wheelhouse
   ```
3. Copy the wheel:
   ```bash
   mkdir -Force wheelhouse
   #cp target/wheels/omniregress-*.whl wheelhouse/


   cp dist/omniregress-*.tar.gz wheelhouse/

   #.venv
   Move-Item -Path dist/omniregress-*.whl -Destination wheelhouse/

   #.venv_Ubuntu
   mv dist/omniregress-*.whl wheelhouse/


   ```
4. Upload to PyPI:
   ```bash
   pip install twine
   twine upload dist/*
   ```

