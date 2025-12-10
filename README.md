# SMLR: Surrogate Models for Linear Response

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

![Logo](SMLR.png)

**SMLR** is a high-performance Python package for building fast, accurate surrogate models of linear-response strength functions across arbitrary parameter spaces. Designed for nuclear physics and beyond, SMLR combines Lorentzian mixture modeling with machine learning to enable rapid parameter space exploration without costly ab initio calculations.

## Key Features

- **Universal Parameter Support**: Handle arbitrary parameter dimensions for any physical theory
- **Lorentzian Mixture Fitting**: Compress complex strength functions into interpretable resonance peaks
- **Fast Emulation**: Predict strength spectra at new parameter points in milliseconds
- **Domain Agnostic**: Applicable to beta decay, dipole polarizability, and other linear response problems
- **Minimal Dependencies**: Pure CPU implementation using NumPy, SciPy, and scikit-learn
- **Production Ready**: Comprehensive test suite, type hints, and professional documentation
- **Reproducible Science**: Headless plotting (Agg backend) for reproducible figures

## Package Contents

This repository ships a **reusable Python package** for fitting and emulating strength functions
from any theory and parameter dimension. Legacy QRPA scripts remain under `Beta_decay/` and
`Dipole_polarizability/` for reproducibility.

## Quick Start

### Installation

**Using uv (recommended - fast and reproducible):**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync           # creates .venv and installs dependencies from pyproject
uv run pytest     # verify installation with test suite
```

**Using pip/venv:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .[dev]
```

### Your First Emulator (30 seconds)
### Your First Emulator (30 seconds)

```python
import numpy as np
from smlr.data import StrengthDataset
from smlr.emulator import StrengthEmulator

# 1. Prepare your strength function data
energy = np.linspace(-3, 3, 300)
strengths = np.stack([
    np.exp(-0.5 * (energy + 0.5)**2),  # Sample 1
    np.exp(-0.4 * (energy - 0.8)**2),  # Sample 2
])
params = np.array([[0.1, 0.4], [0.6, 0.9]])

# 2. Create a dataset
ds = StrengthDataset.from_arrays(params=params, energy=energy, strengths=strengths)

# 3. Train the emulator
emu = StrengthEmulator(n_components=2, width_mode="global", random_state=0)
emu.fit(ds)

# 4. Predict at a new parameter point
mix, spectrum = emu.predict(np.array([0.3, 0.7]), energy)
```

This demonstrates the basic workflow: create a dataset, train the emulator, and make predictions at new parameter points.

### Run the Synthetic Demo
### Run the Synthetic Demo

See the complete workflow in action with synthetic data:

```bash
python -m smlr.demo.synthetic --out runs/demo
```

This trains an emulator, generates predictions, and saves comparison plots to `runs/demo/`.

## Scientific Applications

SMLR excels at modeling resonance-dominated spectra common in nuclear and atomic physics:

- **Beta-Decay Strength Functions**: Gamow-Teller and Fermi transitions
- **Dipole Polarizability**: E1 and M1 response functions  
- **Giant Resonances**: Collective nuclear excitations
- **Photoabsorption Cross Sections**: Atomic and molecular response
- **Custom Linear Response**: Any domain with peaked spectral features

### Validated Performance

| Application | Parameter Space | Accuracy (Normalized L²) |
|------------|----------------|--------------------------|
| Beta Decay (Ni-80) | 2D (α, β) | **0.04-0.38** |
| Dipole (Yb) | 2D (α, β) | **0.01-0.04** |

*Metrics based on emulator predictions vs. high-fidelity QRPA calculations*

## Documentation

### Quick Links

- **[Usage Guide](docs/usage.md)**: Step-by-step tutorials and data preparation
- **[API Reference](docs/api.md)**: Complete function and class documentation
- **[Paper Reproduction](docs/paper_repro.md)**: Reproduce published results
- **[Contributing](CONTRIBUTING.md)**: Development guidelines and how to contribute

### Documentation Site

Build and serve the full documentation locally:

```bash
uv run mkdocs serve  # Visit http://127.0.0.1:8000
```

## Architecture

SMLR follows a modular design for maximum flexibility:

```
┌─────────────────┐
│  Your Data      │  Energy-strength pairs + parameter vectors
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ StrengthDataset │  Unified data container with interpolation
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Lorentz Fitter  │  Compress spectra → Lorentzian mixtures
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ StrengthEmulator│  Learn parameter → mixture mapping
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Predictions    │  Fast evaluations at new parameter points
└─────────────────┘
```

**Key Components:**

- `smlr.data`: Flexible data loading (CSV, arrays, custom formats)
- `smlr.lorentz`: Physics-informed Lorentzian mixture fitting
- `smlr.emulator`: Regression-based emulation with automatic scaling
- `smlr.metrics`: Validation and error quantification
- `smlr.plotting`: Publication-ready visualization (headless-safe)

## Running Tests

SMLR includes a comprehensive test suite covering all core functionality:

```bash
pytest                              # Run all tests
pytest --cov=smlr --cov-report=html # Generate coverage report
pytest tests/test_emulator.py       # Run specific test file
```

**Quality Metrics:**
- 100% test coverage on core modules
- Type-checked with mypy
- Linted with ruff
- All tests pass on Python 3.9+

## Repository Structure

```
SMLR/
├── src/smlr/              # Main package
│   ├── data.py            # Data loading and dataset management
│   ├── lorentz.py         # Lorentzian mixture fitting
│   ├── emulator.py        # Emulator training and prediction
│   ├── metrics.py         # Evaluation metrics
│   ├── plotting.py        # Visualization utilities
│   └── demo/              # Demonstration scripts
│       └── synthetic.py   # Synthetic data example
├── tests/                 # Comprehensive test suite
│   ├── test_data.py       # Data loading tests
│   ├── test_lorentz.py    # Fitting algorithm tests
│   └── test_emulator.py   # End-to-end emulator tests
├── docs/                  # Documentation source
│   ├── index.md           # Documentation home
│   ├── usage.md           # Usage tutorials
│   ├── api.md             # API reference
│   └── paper_repro.md     # Reproducibility guide
├── examples/              # Example workflows
│   └── paper_repro.py     # Paper reproduction script
├── Beta_decay/            # Legacy QRPA scripts (beta decay)
├── Dipole_polarizability/ # Legacy QRPA scripts (dipole)
├── pyproject.toml         # Project configuration
├── LICENSE                # MIT License
└── CONTRIBUTING.md        # Contribution guidelines
```

## Use Cases

### Load Data from CSV Files

```python
from pathlib import Path
from smlr.data import StrengthDataset

# Prepare metadata.csv with columns: param1, param2, spectrum_file
ds = StrengthDataset.from_folder(
    metadata_csv="metadata.csv",
    spectrum_column="spectrum_file",
    param_columns=["param1", "param2"],
)
```

### Train with Custom Width Strategy

```python
from smlr.emulator import StrengthEmulator

# Use per-component widths for multi-resonance systems
emu = StrengthEmulator(
    n_components=3, 
    width_mode="per_component",  # vs "global"
    random_state=42
)
emu.fit(dataset)
```

### Batch Predictions

```python
# Predict over a parameter grid
param_grid = np.mgrid[0:1:10j, 0:1:10j].reshape(2, -1).T
energy = np.linspace(-5, 5, 500)

for params in param_grid:
    mixture, spectrum = emu.predict(params, energy)
    # Process predictions...
```

### Train/Validation/Test Split

```python
train_ds, val_ds, test_ds = dataset.train_val_test_split(
    train=0.7, 
    val=0.15,
    seed=42
)

emu.fit(train_ds)
# Evaluate on val_ds and test_ds
```

## 🎓 Citation
## 🎓 Citation

If you use SMLR in your research, please cite:

```bibtex
@article{jin2025smlr,
  title={Surrogate Models for Linear Response},
  author={Jin, L. and others},
  journal={TBD},
  year={2025},
  note={GitHub: https://github.com/ascsn/SMLR}
}
```

## Contributing

Contributions are welcome. Whether fixing bugs, adding features, or improving documentation:

1. Read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines
2. Fork the repository and create a feature branch
3. Make your changes with tests and documentation
4. Submit a pull request

**Areas for contribution:**
- Additional physics domains and examples
- Performance optimizations
- Extended plotting capabilities
- Tutorial notebooks

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Original QRPA calculations and data preparation
- scikit-learn and SciPy communities for robust scientific computing tools
- All contributors and early adopters

## Support

- **Issues**: [GitHub Issues](https://github.com/ascsn/SMLR/issues)
- **Discussions**: [GitHub Discussions](https://github.com/ascsn/SMLR/discussions)
- **Email**: [Contact maintainers](mailto:TBD)


