# SOC

**Spin-Orbit Coupling and Quantum Chemistry Tools**  
_A hybrid Python/C++ package for advanced spin-orbit coupling calculations, leveraging Libint, Eigen, pybind11, and the scientific Python stack._

---

## Features

- High-performance C++ backend (Libint2, Eigen, OpenMP)
- Seamless Python interface via pybind11
- Tools for molecular integrals, band structures, population analysis, and more
- Easy integration with Python workflows (numpy, scipy, matplotlib, pymatgen)
- Modern build system—install everything with one command!

---

## Requirements

- Linux, macOS, or WSL (Windows Subsystem for Linux)
- Python ≥3.9
- C++17 compiler (provided by conda/mamba)
- [Libint2](https://github.com/evaleev/libint) ≥2.6
- [Eigen3](https://eigen.tuxfamily.org/) ≥3.4
- [pybind11](https://pybind11.readthedocs.io/) ≥2.10
- (All dependencies handled via conda/mamba)

---

## Installation

### 1. Create and activate the environment

With **conda** or **mamba** (recommended):

```bash
mamba env create -f environment.yml
mamba activate soc
```

### 2. Build and install the package

From the root of the repository, just run:

```bash
pip install .
```

- This will compile the C++ extension using your environment's compilers and libraries.
- After this, you can import `libint_cpp` and all Python modules directly from your scripts.

---

## Project Structure

```
SOC/
├── libint/           # C++ extension sources (pybind11, Libint)
│   ├── CMakeLists.txt
│   ├── bindings.cpp
│   ├── integrals_core.cpp
│   └── integrals_core.hpp
├── src/              # Python modules
│   ├── analysis.py
│   ├── config.py
│   ├── hamiltonian.py
│   ├── parsers.py
│   ├── soc_driver.py
│   └── utils.py
├── environment.yml
├── pyproject.toml
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

---

## Example Usage

```python
import libint_cpp                     # C++ extension module
from src import analysis, soc_driver  # Your Python analysis scripts

# Example: compute something, analyze results...
```

For detailed examples and API docs, see [docs/](docs/) (if available) or the script docstrings.

---

## Development & Contributing

- Source code: [https://github.com/YOURUSERNAME/SOC](https://github.com/YOURUSERNAME/SOC)
- To contribute, fork the repo and submit a pull request.
- Please open an issue for bug reports, feature requests, or questions.

### Building C++ manually (advanced/dev):

```bash
cd libint
mkdir -p build && cd build
cmake ..
make
```
_Normally this is handled automatically by `pip install .`—no manual build needed!_

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Citing

If you use **SOC** in your research, please cite:

> Ivan Infante, SOC: Spin-Orbit Coupling Python/C++ Package (2025)  
> [GitHub repo link here]

---

## Support

- For help, open a [GitHub Issue](https://github.com/YOURUSERNAME/SOC/issues).
- For scientific consulting or collaboration, contact [YOUR EMAIL HERE].

---

*Happy computing!*