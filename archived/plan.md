# Packaging plan: generalized strength-function emulator

## Objectives
- Provide a reusable Python package that fits and emulates strength functions for arbitrary theories and parameter dimensions.
- Keep a clean API for (a) loading strength data, (b) fitting Lorentzian mixtures, (c) learning a parametric emulator, and (d) evaluating/plotting predictions.
- Ship with tests, docs, and examples so users can onboard quickly.

## Deliverables
- `pyproject.toml` with dependencies and editable install support; recommendation to use a local venv.
- `src/smlr/` package with:
  - `data.py`: loaders/utilities for strength tables; dataset abstraction.
  - `lorentz.py`: Lorentzian mixture model + fitting helpers.
  - `emulator.py`: regression-based emulator (Gaussian process or ridge) that maps parameters → mixture parameters and reconstructs spectra.
  - `metrics.py`: helpers (normalized L2, MAE on half-lives, etc.).
  - `plotting.py`: optional plotting utilities (matplotlib) that never require a display backend.
  - `examples/` data + scripts (small synthetic demo).
- Tests under `tests/` using pytest (no GPU needed) that cover loaders, fitting stability, emulator training on synthetic data, and API contracts.
- Documentation: updated top-level `README.md` + `docs/usage.md` and `docs/api.md` with CLI/venv instructions and examples.

## Target workflow
1) User arranges their strength functions as CSV/TSV (energy, strength) files + a CSV describing parameter vectors.
2) `StrengthDataset.from_folder(...)` ingests metadata and spectra.
3) `LorentzianMixture.fit_strength(...)` compresses each spectrum to `n_components` poles (+ width strategy).
4) `StrengthEmulator.fit(dataset, ...)` trains regressors for energies, strengths, and width; stores scalers.
5) `StrengthEmulator.predict(params, energy_grid)` returns reconstructed spectra and mixture parameters; can also compute half-life proxy via numeric integration.

## Notes on generality
- Parameter dimension is arbitrary (`d >= 1`), stored as `np.ndarray` of shape `(d,)`.
- Width handling: either fixed scalar, per-component scalar, or global learned regressor.
- Fitting is CPU-only (SciPy + scikit-learn); avoids TensorFlow/GPUs.
- Normalization: spectra can be optionally normalized to unit integral to ease regression.

## Testing strategy
- Synthetic sine/Lorentz mixture data to validate fitting accuracy (<5% normalized L2).
- Emulator round-trip: train on synthetic grid and ensure predictions interpolate with small error on held-out points.
- Deterministic seeds; small grids to keep test runtime < 10s.

## CLI (initial)
- `python -m smlr.demo.synthetic` trains an emulator on synthetic data and writes plots to `./runs/demo`.

## Migration of legacy scripts
- Keep originals under `Beta_decay/` and `Dipole_polarizability/` untouched.
- Provide a short compatibility note in `README` pointing to new package API.

## Open questions / future work
- Add half-life domain-specific pieces (phase-space polynomials) as optional add-ons.
- Support batching/parallel fitting for large datasets.
- Provide HDF5-based cache of fitted mixtures.
