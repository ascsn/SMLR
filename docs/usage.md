# Usage guide

This page is a practical recipe for installing SMLR, loading your own spectra, fitting emulators,
and regenerating the paper figures.

## 1) Create and activate an environment

venv:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .[dev]
```

uv (fast, reproducible):
```bash
curl -Ls https://astral.sh/uv/install.sh | sh
uv sync --group dev --group docs   # installs deps + docs extras into .venv
uv run pytest                      # run tests
uv run mkdocs serve                # serve docs locally
```

### Optional Optimization Backends

For high-dimensional problems or GPU acceleration, install optional backends:

```bash
# TensorFlow backend (Adam optimizer, GPU support)
pip install -e .[tensorflow]

# JAX/Optax backend (modern autodiff, JIT compilation)
pip install -e .[jax]

# All optional dependencies
pip install -e .[all]
```

## 2) Prepare your data (any domain)
- Put each spectrum in a text file with two columns: `energy strength` (CSV/TSV/whitespace all OK).
- Collect a metadata CSV with one row per spectrum, parameter columns, and a `spectrum` column
	pointing to the file:

```csv
param_g0,param_V0,spectrum,label
0.2,1.0,data/lorm_0.2_1.0.out,baseline
0.5,1.2,data/lorm_0.5_1.2.out,scan-1
```

**High-dimensional parameters**: SMLR supports any number of parameters (2, 5, 10, 15+):

```csv
p1,p2,p3,p4,p5,spectrum,label
0.1,0.2,0.3,0.4,0.5,data/spectrum_001.out,sample-1
0.2,0.3,0.4,0.5,0.6,data/spectrum_002.out,sample-2
```

## 3) Load data and fit an emulator
```python
from pathlib import Path
import numpy as np
from smlr import Surrogate, StrengthDataset

# Load dataset (works for any parameter dimension: 2D, 5D, 10D, ...)
meta = Path("metadata.csv")
ds = StrengthDataset.from_folder(
		meta,
		spectrum_column="spectrum",
		param_columns=["param_g0", "param_V0"],  # or ["p1", "p2", "p3", "p4", "p5"]
		label_column="label",
)

# Fit emulator with chosen backend and regression method
model = Surrogate(
    "regression",                 # Backend: "regression" or "pmm"
    n_components=4,
    width_mode="global",
    regression_method="ridge",    # or "linear", "polynomial", "gp"
    random_state=0,
)
model.fit(ds, verbose=True)

# Predict on a new parameter point
energy_grid = np.linspace(-3, 3, 300)
result = model.predict(np.array([0.35, 1.1]), energy_grid)
print(f"Spectrum shape: {result.spectrum.shape}")
print(f"Poles: {result.poles}")

# Batch prediction for multiple points
params_batch = np.random.rand(100, 2)  # 100 points in 2D space
results = model.predict_batch(params_batch, energy_grid)
spectra = np.array([r.spectrum for r in results])
```

## 4) Synthetic demo (smoke test)
```bash
uv run python -m smlr.demo.synthetic --out runs/demo
```
Outputs `synthetic_demo.png` and a compact `.npz` with the learned mixture.

## 5) High-dimensional emulation example

For nuclear physics or other domains with many parameters:

```bash
# 5-parameter emulation
uv run python examples/high_dim_emulation.py --n-params 5

# 10-parameter emulation with Gaussian Process regression
uv run python examples/high_dim_emulation.py --n-params 10 --method gp

# 15-parameter emulation (requires more training samples)
uv run python examples/high_dim_emulation.py --n-params 15 --n-train 200
```

## 6) Domain-specific examples

### Acoustic/Vibrational Spectroscopy
```bash
uv run python examples/acoustic_resonance.py
```
Demonstrates emulation of room acoustics, musical instruments, or structural vibrations.

### Materials Science Spectroscopy
```bash
# X-ray absorption (XANES)
uv run python examples/materials_spectroscopy.py --type xanes

# Raman spectroscopy
uv run python examples/materials_spectroscopy.py --type raman
```

## 7) Paper reproduction via packaged workflow
```bash
uv run python examples/paper_repro.py --mode paper --max-files 6 --plots-dir docs/figs
```
- Uses the bundled Ni-80 beta-decay and Yb dipole datasets if present.
- Saves comparison plots to `docs/figs/` and prints normalized L2 errors.
- See the live numbers and plots on [Paper Reproduction](paper_repro.md).

## 8) Physics-informed training with observables

SMLR supports physics-informed training using observable constraints:

```python
from smlr.observables import SumRule, DipolePolarizability, ObservableSet

# Define observables with target values
obs_set = ObservableSet([
    SumRule(k=1, name="TRK", target=875.0, weight=1.0),
    DipolePolarizability(target=18.5, weight=0.5),
])

# Compute observables from a spectrum
results = obs_set.compute_all(energy, strength)
total_loss = obs_set.total_loss(energy, strength)
```

## 9) Using alternative optimization backends

```python
from smlr.optimization import create_optimizer, get_available_backends

# Check available backends
print(get_available_backends())  # ['scipy', 'tensorflow', 'jax']

# Create optimizers for different backends
scipy_opt = create_optimizer("scipy", method="L-BFGS-B", max_iterations=1000)
adam_opt = create_optimizer("jax", method="adam", learning_rate=0.01)
tf_opt = create_optimizer("tensorflow", method="adamw", learning_rate=0.001)
```

## 10) Run tests
```bash
uv run pytest
```

## Tips
- **Regression method**: Start with `"ridge"` for most cases. Use `"polynomial"` for nonlinear parameter dependence. Use `"gp"` for uncertainty quantification (but slower).
- **High dimensions**: For >10 parameters, increase training samples and consider polynomial or GP regression.
- Normalize spectra when scales differ; denormalize only if you need absolute strengths.
- Use `width_mode="per_component"` when different resonance families have distinct widths.
- Plots are headless (Agg); always saved to disk for reproducibility.
