# SMLR Copilot Instructions

## Project Overview

SMLR (Surrogate Models for Linear Response) builds fast emulators for strength functions in nuclear physics and beyond. The core workflow: fit Lorentzian mixtures to training spectra → learn a parameter-to-pole mapping → predict at new parameter points.

## Architecture & Data Flow

```
StrengthDataset → fit_lorentzian_mixture() → StrengthEmulator.fit() → predict()
                        or
StrengthDataset → ParametricMatrixModel.fit() → predict()
```

**Two emulation backends:**
- `StrengthEmulator` (regression-based): Fast training, good for smooth parameter dependence
- `ParametricMatrixModel` (PMM in `pmm.py`): Physics-based, better extrapolation, sum rule preservation

Select via `smlr.backends.get_emulator("regression")` or `get_emulator("pmm")`.

## Key Modules (in `src/smlr/`)

| Module | Purpose |
|--------|---------|
| `data.py` | `StrengthDataset`, `StrengthSample` - data containers with interpolation |
| `lorentz.py` | `fit_lorentzian_mixture()` - compress spectra to pole parameters |
| `emulator.py` | `StrengthEmulator` - regression-based emulation (linear/ridge/poly/GP) |
| `pmm.py` | `ParametricMatrixModel` - physics-based matrix model emulation |
| `backends.py` | `get_emulator()` factory for backend selection |
| `observables.py` | Physics observables (sum rules, half-lives) for loss terms |
| `optimization.py` | Multi-backend optimizer (scipy/tensorflow/jax) |
| `metrics.py` | `normalized_l2()` - primary validation metric |
| `plotting.py` | Headless-safe (`Agg` backend) visualization |

## Development Commands

```bash
uv sync                           # Install dependencies (creates .venv)
uv run pytest                     # Run tests
uv run pytest --cov=smlr          # Tests with coverage
uv run ruff check src/ tests/     # Lint
uv run mkdocs serve               # Local docs at http://127.0.0.1:8000
python -m smlr.demo.synthetic --out runs/demo  # Quick smoke test
```

## Code Patterns

**Creating datasets:**
```python
# From arrays
ds = StrengthDataset.from_arrays(params=params, energy=energy, strengths=strengths)
# From CSV metadata
ds = StrengthDataset.from_folder("metadata.csv", param_columns=["alpha", "beta"])
```

**Emulator training (supports arbitrary parameter dimensions):**
```python
emu = StrengthEmulator(n_components=4, width_mode="global", regression_method="ridge")
emu.fit(dataset)
mixture, spectrum = emu.predict(new_params, energy_grid)
```

**Width modes:** `"global"` (one width for all poles) or `"per_component"` (separate widths).

**Regression methods:** `"linear"`, `"ridge"`, `"polynomial"`, `"gp"` (Gaussian process).

## Testing Conventions

- Tests in `tests/test_*.py`, functions named `test_*`
- Use `random_state` or `seed` for reproducibility
- Primary metric: `normalized_l2(pred, truth, energy)` should be < 0.05 for good fits
- See `tests/test_emulator.py::test_emulator_interpolates_spectrum` as reference

## File Naming & Data Patterns

- Training data: `{dataset}_data_{nucleus}/` directories
- Strength files: two-column format (energy, strength)
- Metadata CSVs: parameter columns + `spectrum_file` column pointing to data files

## Important Considerations

- Plotting uses `matplotlib.use("Agg")` for headless environments
- Optional backends (TensorFlow, JAX) installed via `pip install smlr[tensorflow]` or `smlr[jax]`
- Legacy scripts in `Beta_decay/` and `Dipole_polarizability/` - prefer `src/smlr/` for new code
- Use NumPy-style docstrings with type hints
- Commit messages: conventional format (`feat:`, `fix:`, `docs:`, `test:`)
