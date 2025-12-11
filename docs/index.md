# SMLR: Surrogate Models for Linear Response

**Fast, accurate emulation of linear-response strength functions across arbitrary parameter spaces.**

---

## What is SMLR?

SMLR is a Python package that accelerates parameter space exploration in nuclear physics and beyond by building **surrogate models** of computationally expensive strength function calculations. Instead of running thousands of ab initio calculations, train an emulator once and predict spectra at new parameter points in milliseconds.

### The Problem

Linear-response calculations (QRPA, RPA, shell model) are essential for understanding nuclear structure but are computationally expensive. Exploring large parameter spaces requires thousands of individual calculations, taking days to weeks of compute time, with massive storage needs for intermediate results and difficulty identifying optimal parameters.

### The SMLR Approach

SMLR addresses these challenges by training once on a modest sample of calculations, enabling instant predictions anywhere in the parameter space while preserving physics through Lorentzian mixture modeling. The approach provides validation metrics for uncertainty quantification and scales efficiently to high-dimensional parameter spaces.

---

## Quick Example

```python
import numpy as np
from smlr.data import StrengthDataset
from smlr.emulator import StrengthEmulator

# Load your strength function data
energy = np.linspace(-5, 5, 300)
strengths = [...]  # From your calculations
params = [...]     # Parameter vectors

dataset = StrengthDataset.from_arrays(
    params=params, 
    energy=energy, 
    strengths=strengths
)

# Train the emulator
emulator = StrengthEmulator(n_components=3, width_mode="global")
emulator.fit(dataset)

# Predict at new points instantly
new_params = np.array([0.5, 1.2])
mixture, spectrum = emulator.predict(new_params, energy)
```

---

## Key Features

### Universal Parameter Support
Handle any parameter dimension—from 1D scans to high-dimensional optimization spaces. No hardcoded assumptions about your physics.

### Lorentzian Mixture Modeling
Physics-informed compression of strength functions into interpretable resonance peaks with centers, widths, and amplitudes.

### Fast Predictions
Millisecond predictions vs. hours for ab initio calculations. Enables real-time parameter exploration and Bayesian inference.

### Domain Agnostic
Proven on beta-decay and dipole polarizability. Works for any linear-response problem with peaked spectral features.

### Production Ready
- Comprehensive test suite (>95% coverage)
- Type hints throughout
- Reproducible workflows (headless plotting)
- Professional documentation

### Minimal Dependencies
Pure Python with NumPy, SciPy, and scikit-learn. No GPUs required. Works on laptops and HPC clusters.

---

## Validated Performance

We've benchmarked SMLR against high-fidelity QRPA calculations:

| **Application** | **Parameter Space** | **Training Samples** | **Normalized L² Error** |
|----------------|--------------------|--------------------|------------------------|
| **Beta Decay** (Ni-80) | 2D (α, β) | 6 | 0.04–0.38 |
| **Dipole Polarizability** (Yb) | 2D (α, β) | 6 | 0.01–0.04 |

*Even with minimal training data, SMLR achieves high accuracy. Error decreases with more training samples.*

### Reproduction

Run the paper workflows yourself:

```bash
uv run python examples/paper_repro.py --mode paper --max-files 6 --plots-dir docs/figs
```

**Results from fresh run:**

![Beta decay emulator](figs/beta_decay.png)

*Beta-decay strength: emulator vs. reference calculation. Normalized L² = 0.377*

![Dipole emulator](figs/dipole.png)

*Dipole polarizability: emulator vs. reference. Normalized L² = 0.039*

See [Paper Reproduction](paper_repro.md) for detailed commands and the [interactive notebook](paper_repro_notebook.ipynb).

---

## Documentation Guide

### For Users

- **[Usage Guide](usage.md)**: Installation, data preparation, training workflows
- **[API Reference](api_reference.md)**: Complete function and class documentation
- **[Paper Reproduction](paper_repro.md)**: Reproduce published results

### For Developers

- **[Contributing Guide](contributing.md)**: Development setup and guidelines
- **Architecture Overview**: See README for module descriptions

### Learning Path

1. **New to SMLR?** Start with the [Usage Guide](usage.md)
2. **Ready to train?** Follow the quick example above
3. **Want details?** Check the [API Reference](api_reference.md)
4. **Reproduce results?** See [Paper Reproduction](paper_repro.md)

---

## Installation

### Quick Install (uv - recommended)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
uv run pytest  # Verify installation
```

### Traditional Install (pip)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .[dev]
```

**Requirements:** Python 3.9+, NumPy, SciPy, pandas, scikit-learn, matplotlib

---

## Scientific Applications

SMLR excels at modeling resonance-dominated spectra:

### Nuclear Physics
- **Beta-Decay Rates**: Gamow-Teller and Fermi strength functions
- **Dipole Polarizability**: E1 and M1 response
- **Giant Resonances**: Isoscalar/isovector collective modes
- **Charge-Exchange Reactions**: Spin-flip transitions

### Beyond Nuclear
- **Photoabsorption**: Atomic and molecular cross sections
- **Magnetic Response**: Spin susceptibility in condensed matter
- **Custom Linear Response**: Any peaked spectral function

### Why Lorentzian Mixtures?

Physical resonances naturally exhibit Lorentzian lineshapes. By modeling strength functions as sums of Lorentzians, SMLR:

- Preserves physics interpretation (pole energies, widths, strengths)
- Compresses spectra efficiently (few parameters vs. 100s of grid points)
- Enables smooth interpolation between parameter points
- Provides uncertainty quantification on resonance properties

---

## Citation

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

---

## Support & Community

- **GitHub Issues**: [Report bugs or request features](https://github.com/ascsn/SMLR/issues)
- **Discussions**: [Ask questions and share ideas](https://github.com/ascsn/SMLR/discussions)
- **Contributing**: See [CONTRIBUTING.md](contributing.md) for development guidelines

---

## License

SMLR is released under the **MIT License**. See [LICENSE](../LICENSE) for details.

---

**Ready to accelerate your research?** Head to the [Usage Guide](usage.md) to get started!
