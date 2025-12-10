# Development Guide

Technical documentation for SMLR developers and contributors.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Design Principles](#design-principles)
- [Module Details](#module-details)
- [Extension Points](#extension-points)
- [Performance Optimization](#performance-optimization)
- [Release Process](#release-process)

---

## Architecture Overview

### High-Level Design

SMLR follows a **pipeline architecture** with three main stages:

```
┌─────────────┐
│   Data      │  smlr.data
│  Loading    │  StrengthDataset, StrengthSample
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Lorentzian │  smlr.lorentz
│   Fitting   │  fit_lorentzian_mixture, LorentzianMixture
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Emulator   │  smlr.emulator
│  Training   │  StrengthEmulator
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ Prediction  │  smlr.emulator.predict()
│ Evaluation  │  smlr.metrics, smlr.plotting
└─────────────┘
```

### Core Abstractions

**`StrengthSample`** (dataclass):
- Immutable container for single spectrum
- Parameters (array), energy grid (array), strength values (array)
- Optional label for identification

**`StrengthDataset`** (collection):
- Manages multiple samples
- Handles interpolation to common grids
- Provides train/val/test splitting

**`LorentzianMixture`** (dataclass + methods):
- Represents fitted resonance parameters
- `evaluate()` method for spectrum reconstruction
- Serializable/deserializable

**`StrengthEmulator`** (stateful model):
- Fits regression models (parameter → mixture parameters)
- Stores scalers for normalization
- Provides prediction API

---

## Design Principles

### 1. **Separation of Concerns**

Each module has a single responsibility:
- `data.py`: I/O and data management only
- `lorentz.py`: Physics-informed fitting only
- `emulator.py`: Machine learning only
- `plotting.py`: Visualization only

**Why**: Easier testing, maintenance, and extension.

### 2. **Immutability Where Possible**

Dataclasses (`StrengthSample`, `LorentzianMixture`) are immutable by design. Datasets and emulators maintain internal state but expose functional APIs.

**Why**: Reduces bugs from side effects, enables safe parallelization.

### 3. **Type Hints Everywhere**

All public functions and methods have complete type annotations.

**Why**: Static analysis, IDE support, self-documenting code.

### 4. **Fail Fast**

Validation happens at API boundaries (constructors, function entry). Explicit `ValueError` / `RuntimeError` with clear messages.

**Why**: Better debugging experience, clearer error messages.

### 5. **No Side Effects in Computation**

Functions like `fit_lorentzian_mixture()` never modify inputs. Emulator methods return new objects.

**Why**: Predictable behavior, easier reasoning about code.

### 6. **Headless by Default**

All plotting uses Agg backend, never requires display. Figures are returned, not shown.

**Why**: Server-safe, reproducible, CI/CD friendly.

---

## Module Details

### `smlr.data`

**Responsibilities**:
- Load strength functions from various formats (CSV, arrays)
- Manage collections of spectra with heterogeneous energy grids
- Provide interpolation and normalization utilities

**Key Classes**:

```python
@dataclass
class StrengthSample:
    params: Array
    energy: Array
    strength: Array
    label: Optional[str] = None
    
    def normalized(self) -> StrengthSample: ...

class StrengthDataset:
    def __init__(self, samples: Sequence[StrengthSample]): ...
    def parameters(self) -> Array: ...
    def to_matrix(self, energy_grid=None, normalize=False): ...
    def train_val_test_split(self, train=0.7, val=0.15, seed=0): ...
    
    @classmethod
    def from_folder(cls, metadata_csv, ...): ...
    
    @classmethod
    def from_arrays(cls, params, energy, strengths, ...): ...
```

**Design Decisions**:
- `from_folder` uses pandas for CSV parsing (robust, handles edge cases)
- Interpolation uses `np.interp` (fast, no scipy dependency)
- Normalization via trapezoidal integration (numerical stability)

**Extension Points**:
- Add `from_hdf5()` for binary format
- Implement `StrengthDataset.__getitem__()` for indexing
- Add custom interpolation methods (spline, etc.)

### `smlr.lorentz`

**Responsibilities**:
- Define Lorentzian lineshape and mixture evaluation
- Fit mixtures to data via nonlinear optimization
- Enforce physical constraints (positivity, ordering)

**Key Functions**:

```python
@dataclass
class LorentzianMixture:
    energies: Array
    strengths: Array
    widths: Array
    
    def evaluate(self, energy_grid: Array) -> Array: ...

def fit_lorentzian_mixture(
    energy: Array,
    strength: Array,
    n_components: int,
    width_mode: Literal["global", "per_component"] = "global",
    eta_init: float = 0.5,
    min_spacing: float = 0.1,
    l2: float = 1e-3,
) -> LorentzianMixture: ...
```

**Design Decisions**:
- SciPy `least_squares` with "trf" (trust-region reflective) method
  - Handles bounds implicitly via parameterization
  - Robust to poor initialization
- Inverse softplus transformation ensures positivity:
  ```python
  def _inv_softplus(y): return np.log(np.expm1(y))
  def _softplus(x): return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)
  ```
- Ordered energies via cumulative sum of gaps:
  ```python
  energies[0] = E_min + softplus(z[0])
  energies[i] = energies[i-1] + softplus(z[i]) + min_spacing
  ```

**Extension Points**:
- Alternative optimizers (BFGS, gradient-free methods)
- Bayesian fitting with `emcee` or `PyMC`
- Voigt profiles (Gaussian + Lorentzian convolution)
- Multi-channel fitting (multiple spectra simultaneously)

### `smlr.emulator`

**Responsibilities**:
- Learn parameter → mixture mappings via regression
- Handle feature scaling and normalization
- Provide prediction API

**Key Class**:

```python
class StrengthEmulator:
    def __init__(self, n_components, width_mode="global", random_state=0): ...
    
    def fit(
        self, 
        dataset: StrengthDataset,
        mixtures: Optional[Sequence[LorentzianMixture]] = None,
        fitter: Callable = fit_lorentzian_mixture,
        fit_kwargs: Optional[dict] = None,
        normalize_strengths: bool = True,
    ) -> StrengthEmulator: ...
    
    def predict_mixture(self, params: Array) -> LorentzianMixture: ...
    def predict_spectrum(self, params: Array, energy_grid: Array) -> Array: ...
    def predict(self, params, energy_grid=None): ...
```

**Design Decisions**:
- `MultiOutputRegressor` wraps linear regression for parallel training
  - Each mixture parameter (energy, strength, width) gets independent model
- `StandardScaler` for feature normalization (zero mean, unit variance)
  - Applied to parameters, energies, strengths, widths separately
- Log-transform for strengths when `normalize_strengths=True`:
  ```python
  S_scaled = log1p(S)  # Handles zeros gracefully
  S_predicted = expm1(S_scaled)  # Inverse transform
  ```
- Sorting energies post-prediction ensures physical ordering

**Extension Points**:
- Replace `LinearRegression` with `Ridge`, `Lasso`, `ElasticNet`
- Implement `GaussianProcessRegressor` for uncertainty quantification
- Add `predict_batch()` for vectorized predictions
- Serialize/deserialize with `pickle` or `joblib`

### `smlr.metrics`

**Responsibilities**:
- Quantify prediction accuracy
- Domain-agnostic error measures

**Key Functions**:

```python
def normalized_l2(y_pred: Array, y_true: Array, x: Array) -> float:
    """||y_pred - y_true||_L2 / ||y_true||_L2"""
    num = integrate.trapezoid((y_pred - y_true)**2, x)
    den = integrate.trapezoid(y_true**2, x)
    return float(num / (den + 1e-16))

def mean_absolute_relative_error(y_pred: Array, y_true: Array) -> float:
    """Mean of |y_pred - y_true| / |y_true|"""
    return float(np.mean(np.abs(y_pred - y_true) / (np.abs(y_true) + 1e-16)))
```

**Extension Points**:
- Add `r2_score`, `mean_squared_error`
- Implement energy-weighted errors (penalize peak regions more)
- Add sum-rule validation metrics

### `smlr.plotting`

**Responsibilities**:
- Generate publication-ready figures
- Headless operation (Agg backend)

**Key Functions**:

```python
def plot_spectrum(energy, strength, label="data") -> plt.Figure: ...
def plot_comparison(energy, truth, prediction, title="...", labels=...) -> plt.Figure: ...
def plot_mixture_components(mixture, energy) -> plt.Figure: ...
```

**Design Decisions**:
- Always return `Figure` objects (caller controls saving)
- Set `matplotlib.use("Agg")` at module import
- Consistent styling (grid, labels, legends)

**Extension Points**:
- Add `plot_residuals()`
- Implement `plot_parameter_space()` for 2D heatmaps
- Support custom styles/themes

---

## Extension Points

### Adding New Data Formats

To support HDF5, NetCDF, etc.:

```python
# In smlr/data.py

class StrengthDataset:
    @classmethod
    def from_hdf5(cls, filepath: Path) -> "StrengthDataset":
        import h5py
        samples = []
        with h5py.File(filepath, 'r') as f:
            for key in f.keys():
                params = f[key]['params'][:]
                energy = f[key]['energy'][:]
                strength = f[key]['strength'][:]
                samples.append(StrengthSample(params, energy, strength, key))
        return cls(samples)
```

### Custom Regression Models

To use Ridge regression instead of linear:

```python
from sklearn.linear_model import Ridge
from smlr.emulator import StrengthEmulator

# Monkey-patch (quick hack) or subclass
class RidgeEmulator(StrengthEmulator):
    def fit(self, dataset, **kwargs):
        # Replace LinearRegression with Ridge
        from sklearn.multioutput import MultiOutputRegressor
        lr_energy = Ridge(alpha=1.0)
        lr_strength = Ridge(alpha=1.0)
        self.energy_reg = MultiOutputRegressor(lr_energy)
        self.strength_reg = MultiOutputRegressor(lr_strength)
        # Continue with parent fit logic...
        return super().fit(dataset, **kwargs)
```

Better: make regressor configurable:

```python
class StrengthEmulator:
    def __init__(self, ..., regressor_class=LinearRegression, regressor_kwargs=None):
        self.regressor_class = regressor_class
        self.regressor_kwargs = regressor_kwargs or {}
```

### Parallelization

For large-scale fitting:

```python
from multiprocessing import Pool
from smlr.lorentz import fit_lorentzian_mixture

def parallel_fit(dataset, n_components, n_workers=4):
    def fit_one(sample):
        return fit_lorentzian_mixture(
            sample.energy, sample.strength, n_components
        )
    
    with Pool(n_workers) as pool:
        mixtures = pool.map(fit_one, dataset.samples)
    
    return mixtures
```

### Model Serialization

Save/load trained emulators:

```python
import pickle

# Save
with open("emulator.pkl", "wb") as f:
    pickle.dump(emu, f)

# Load
with open("emulator.pkl", "rb") as f:
    emu = pickle.load(f)
```

For production: use `joblib` (better compression) or `dill` (more types).

---

## Performance Optimization

### Profiling

Find bottlenecks:

```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Your code here
emu.fit(dataset)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumtime')
stats.print_stats(20)
```

### Common Bottlenecks

1. **Lorentzian fitting** (SciPy `least_squares`):
   - Pre-fit offline if reusing mixtures
   - Reduce grid resolution during fitting
   - Use coarse initial scan + refinement

2. **Data interpolation** (`np.interp` in `to_matrix`):
   - Use common grid to avoid repeated interpolation
   - Cache interpolated results

3. **Repeated predictions**:
   - Vectorize with broadcasting:
     ```python
     # Instead of loop:
     for params in param_grid:
         pred = emu.predict(params, energy)
     
     # Vectorize (future feature):
     preds = emu.predict_batch(param_grid, energy)
     ```

### Memory Optimization

For datasets with >1000 samples:

1. **Generator-based iteration**:
   ```python
   # Instead of loading all at once
   def sample_generator(metadata_csv):
       for row in pd.read_csv(metadata_csv).iterrows():
           yield load_sample(row)
   ```

2. **Chunked processing**:
   ```python
   batch_size = 100
   for i in range(0, len(dataset), batch_size):
       batch = dataset.samples[i:i+batch_size]
       # Process batch
   ```

3. **Sparse grids**: Downsample energy grids to ~100-200 points

---

## Release Process

### Version Numbering

Follow [Semantic Versioning](https://semver.org/):
- **MAJOR**: Incompatible API changes
- **MINOR**: New features, backward compatible
- **PATCH**: Bug fixes

### Pre-Release Checklist

- [ ] All tests pass: `pytest`
- [ ] Code formatted: `ruff check src/`
- [ ] Type-checked (if using mypy): `mypy src/`
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Version bumped in `pyproject.toml` and `src/smlr/__init__.py`

### Release Steps

1. **Create release branch**:
   ```bash
   git checkout -b release/v0.2.0
   ```

2. **Update version**:
   ```toml
   # pyproject.toml
   version = "0.2.0"
   ```
   
   ```python
   # src/smlr/__init__.py
   __version__ = "0.2.0"
   ```

3. **Update CHANGELOG**:
   ```markdown
   ## [0.2.0] - 2025-MM-DD
   ### Added
   - New feature X
   ### Fixed
   - Bug Y
   ```

4. **Commit and tag**:
   ```bash
   git add .
   git commit -m "Release v0.2.0"
   git tag -a v0.2.0 -m "Version 0.2.0"
   ```

5. **Merge to main**:
   ```bash
   git checkout main
   git merge release/v0.2.0
   git push origin main --tags
   ```

6. **Build and publish** (future - when on PyPI):
   ```bash
   python -m build
   twine upload dist/*
   ```

### Post-Release

- [ ] GitHub release with notes
- [ ] Announce on relevant forums/channels
- [ ] Update documentation site
- [ ] Close milestone (if applicable)

---

## Contributing to Development

See [CONTRIBUTING.md](contributing.md) for:
- Code style guidelines
- Pull request process
- Issue templates
- Community guidelines

**Questions?** Open a discussion on [GitHub](https://github.com/ascsn/SMLR/discussions).
