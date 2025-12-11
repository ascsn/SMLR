# API Reference

Complete documentation for all public SMLR classes and functions.

## Table of Contents

- [smlr.data](#smlrdata)
- [smlr.lorentz](#smlrlorentz)
- [smlr.emulator](#smlremulator)
- [smlr.pmm](#smlrpmm)
- [smlr.backends](#smlrbackends)
- [smlr.metrics](#smlrmetrics)
- [smlr.plotting](#smlrplotting)
- [smlr.observables](#smlrobservables)
- [smlr.optimization](#smlroptimization)

---

## `smlr.data`

Data loading, management, and preprocessing for strength functions.

### `StrengthSample`

```python
@dataclass
class StrengthSample:
    params: Array
    energy: Array
    strength: Array
    label: Optional[str] = None
```

Container for a single strength function spectrum.

**Attributes:**

- **params** (`np.ndarray`): Parameter vector of shape `(n_params,)`. Represents the physics parameters (e.g., coupling constants, masses) for this calculation.

- **energy** (`np.ndarray`): Energy grid of shape `(n_points,)`. Should be monotonically increasing.

- **strength** (`np.ndarray`): Strength function values of shape `(n_points,)`. Corresponds to the strength at each energy point.

- **label** (`Optional[str]`): Optional identifier for the sample (e.g., filename, run ID).

**Methods:**

#### `normalized()`

```python
def normalized(self) -> StrengthSample
```

Return a copy with strength normalized to unit integral (trapezoidal rule).

**Returns:**
- `StrengthSample`: New sample with normalized strength. If integral ≤ 0, returns copy unchanged.

**Example:**
```python
from smlr.data import StrengthSample
import numpy as np

energy = np.linspace(-3, 3, 100)
strength = np.exp(-0.5 * energy**2)  # Gaussian
sample = StrengthSample(params=np.array([0.5]), energy=energy, strength=strength)

normalized = sample.normalized()
integral = np.trapz(normalized.strength, normalized.energy)
print(f"Integral: {integral:.3f}")  # Should be ~1.0
```

---

### `StrengthDataset`

```python
class StrengthDataset:
    def __init__(self, samples: Sequence[StrengthSample])
```

Collection of strength function samples with utilities for manipulation and conversion.

**Parameters:**

- **samples** (`Sequence[StrengthSample]`): List or sequence of `StrengthSample` objects. Must contain at least one sample. All samples must have the same parameter dimension.

**Raises:**

- `ValueError`: If `samples` is empty or samples have inconsistent parameter dimensions.

**Attributes:**

- **samples** (`List[StrengthSample]`): List of all samples in the dataset.
- **param_dim** (`int`): Dimension of parameter space (e.g., 2 for 2D parameter scans).

**Methods:**

#### `__len__()`

```python
def __len__(self) -> int
```

Return the number of samples in the dataset.

---

#### `parameters()`

```python
def parameters(self) -> Array
```

Return stacked parameter matrix.

**Returns:**
- `np.ndarray`: Array of shape `(n_samples, n_params)` containing all parameter vectors.

**Example:**
```python
params = dataset.parameters()
print(f"Parameter ranges: {params.min(axis=0)} to {params.max(axis=0)}")
```

---

#### `energy_grids()`

```python
def energy_grids(self) -> List[Array]
```

Return list of energy grids for all samples.

**Returns:**
- `List[np.ndarray]`: List of energy arrays, one per sample. Grids may differ in length and values.

---

#### `strength_arrays()`

```python
def strength_arrays(self) -> List[Array]
```

Return list of strength arrays for all samples.

**Returns:**
- `List[np.ndarray]`: List of strength arrays, one per sample.

---

#### `to_matrix()`

```python
def to_matrix(
    self,
    energy_grid: Optional[Array] = None,
    *,
    normalize: bool = False,
) -> Tuple[Array, Array, Array]
```

Convert dataset to matrix form with optional interpolation and normalization.

**Parameters:**

- **energy_grid** (`Optional[np.ndarray]`): Target energy grid for interpolation. If `None`, assumes all samples share identical energy grids (raises `ValueError` if not).

- **normalize** (`bool`, default=`False`): If `True`, normalize each strength function to unit integral using trapezoidal rule.

**Returns:**
- **Tuple[np.ndarray, np.ndarray, np.ndarray]**:
  - `params`: Shape `(n_samples, n_params)` - Parameter matrix
  - `energy`: Shape `(n_energy,)` - Common energy grid
  - `strengths`: Shape `(n_samples, n_energy)` - Strength matrix

**Raises:**
- `ValueError`: If `energy_grid=None` and samples have differing energy grids.

**Example:**
```python
# Interpolate to common grid
energy_common = np.linspace(-5, 5, 200)
params, energy, strengths = dataset.to_matrix(energy_common, normalize=True)

print(f"Matrix shape: {strengths.shape}")
# Verify normalization
integrals = [np.trapz(s, energy) for s in strengths]
print(f"Integrals: {integrals}")  # Should all be ~1.0
```

---

#### `train_val_test_split()`

```python
def train_val_test_split(
    self,
    train: float = 0.7,
    val: float = 0.15,
    *,
    seed: int = 0
) -> Tuple[StrengthDataset, Optional[StrengthDataset], Optional[StrengthDataset]]
```

Split dataset into training, validation, and test sets with reproducible shuffling.

**Parameters:**

- **train** (`float`, default=`0.7`): Fraction of data for training. Must be in (0, 1).

- **val** (`float`, default=`0.15`): Fraction of data for validation. Must be in [0, 1). Remaining data goes to test set.

- **seed** (`int`, default=`0`): Random seed for reproducibility.

**Returns:**
- **Tuple**: `(train_ds, val_ds_or_None, test_ds_or_None)`
  - `train_ds`: Training dataset (always non-empty)
  - `val_ds_or_None`: Validation dataset or `None` if `val=0` or resulting split is empty
  - `test_ds_or_None`: Test dataset or `None` if `train+val≈1` or resulting split is empty

**Raises:**
- `ValueError`: If `train` or `val` are outside valid ranges, or `train + val >= 1`.

**Example:**
```python
train_ds, val_ds, test_ds = dataset.train_val_test_split(
    train=0.7, val=0.15, seed=42
)

print(f"Train: {len(train_ds)}, Val: {len(val_ds) if val_ds else 0}, "
      f"Test: {len(test_ds) if test_ds else 0}")
```

---

#### `from_folder()` (classmethod)

```python
@classmethod
def from_folder(
    cls,
    metadata_csv: Path | str,
    *,
    spectrum_column: str = "spectrum",
    label_column: Optional[str] = None,
    param_columns: Optional[Sequence[str]] = None,
    root: Optional[Path | str] = None,
    sep: str = ",",
) -> StrengthDataset
```

Load dataset from CSV metadata file pointing to spectrum files.

**Parameters:**

- **metadata_csv** (`Path | str`): Path to CSV file containing metadata. Must include parameter columns and a column with paths to spectrum files.

- **spectrum_column** (`str`, default=`"spectrum"`): Name of CSV column containing relative paths to spectrum files.

- **label_column** (`Optional[str]`): Name of CSV column containing sample labels. If `None`, no labels are assigned.

- **param_columns** (`Optional[Sequence[str]]`): List of column names to use as parameters. If `None`, auto-detects all columns except `spectrum_column` and `label_column`.

- **root** (`Optional[Path | str]`): Base directory for resolving relative spectrum paths. If `None`, uses directory containing `metadata_csv`.

- **sep** (`str`, default=`","`): CSV delimiter.

**Returns:**
- `StrengthDataset`: Loaded dataset.

**Raises:**
- `ValueError`: If required columns are missing from metadata.
- `FileNotFoundError`: If spectrum files cannot be found.

**Spectrum File Format:**
Spectrum files should contain two whitespace/comma/tab-separated columns:
```
# Energy  Strength
-3.0     0.001
-2.9     0.003
...
```

**Example:**

Metadata CSV (`data/metadata.csv`):
```csv
alpha,beta,spectrum,label
0.1,0.5,spectra/run001.txt,baseline
0.2,0.6,spectra/run002.txt,variant1
```

Python:
```python
from pathlib import Path
from smlr.data import StrengthDataset

dataset = StrengthDataset.from_folder(
    metadata_csv="data/metadata.csv",
    spectrum_column="spectrum",
    param_columns=["alpha", "beta"],
    label_column="label",
    root="data"
)

print(f"Loaded {len(dataset)} samples")
```

**See Also:**
- [Usage Guide](usage.md#2-prepare-your-data-any-domain) for detailed data preparation instructions.

---

#### `from_arrays()` (classmethod)

```python
@classmethod
def from_arrays(
    cls,
    params: Array,
    energy: Array,
    strengths: Array,
    labels: Optional[Sequence[str]] = None,
    *,
    normalize: bool = False,
) -> StrengthDataset
```

Create dataset from NumPy arrays (all samples share same energy grid).

**Parameters:**

- **params** (`np.ndarray`): Parameter matrix of shape `(n_samples, n_params)`.

- **energy** (`np.ndarray`): Common energy grid of shape `(n_energy,)`.

- **strengths** (`np.ndarray`): Strength matrix of shape `(n_samples, n_energy)`.

- **labels** (`Optional[Sequence[str]]`): Optional list of labels, one per sample.

- **normalize** (`bool`, default=`False`): If `True`, normalize each spectrum to unit integral.

**Returns:**
- `StrengthDataset`: Constructed dataset.

**Raises:**
- `ValueError`: If `params` and `strengths` have mismatched leading dimensions.

**Example:**
```python
import numpy as np
from smlr.data import StrengthDataset

energy = np.linspace(-3, 3, 200)
params = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
strengths = np.array([
    np.exp(-0.5 * (energy - 0.1)**2),
    np.exp(-0.5 * (energy - 0.3)**2),
    np.exp(-0.5 * (energy - 0.5)**2),
])

dataset = StrengthDataset.from_arrays(
    params=params,
    energy=energy,
    strengths=strengths,
    labels=["sample1", "sample2", "sample3"],
    normalize=True
)
```

**See Also:**
- [Tutorial 1](tutorials.md#tutorial-1-your-first-emulator) for complete workflow example.

---

## `smlr.lorentz`

Lorentzian mixture modeling for strength function compression.

### `LorentzianMixture`

```python
@dataclass
class LorentzianMixture:
    energies: Array
    strengths: Array
    widths: Array
```

Represents a sum of Lorentzian resonances.

**Attributes:**

- **energies** (`np.ndarray`): Resonance center energies, shape `(n_components,)`.

- **strengths** (`np.ndarray`): Resonance amplitudes (integrated strengths), shape `(n_components,)`.

- **widths** (`np.ndarray`): Resonance widths (FWHM), shape `(n_components,)` for per-component widths or `(1,)` for global width.

**Methods:**

#### `evaluate()`

```python
def evaluate(self, energy_grid: Array) -> Array
```

Evaluate the mixture on a given energy grid.

**Parameters:**
- **energy_grid** (`np.ndarray`): Energy points to evaluate at, shape `(n_points,)`.

**Returns:**
- `np.ndarray`: Strength values at each energy point, shape `(n_points,)`.

**Formula:**

$$S(E) = \sum_{i=1}^K A_i \cdot \frac{\Gamma_i / (2\pi)}{(E - E_i)^2 + \Gamma_i^2 / 4}$$

**Example:**
```python
from smlr.lorentz import LorentzianMixture
import numpy as np

mix = LorentzianMixture(
    energies=np.array([-1.0, 1.0]),
    strengths=np.array([2.0, 1.5]),
    widths=np.array([0.3, 0.4])
)

energy = np.linspace(-3, 3, 300)
spectrum = mix.evaluate(energy)

import matplotlib.pyplot as plt
plt.plot(energy, spectrum)
plt.xlabel('Energy (MeV)')
plt.ylabel('Strength')
plt.savefig('mixture.png')
```

---

#### `as_tuple()`

```python
def as_tuple(self) -> Tuple[Array, Array, Array]
```

Return mixture parameters as a tuple.

**Returns:**
- `Tuple[np.ndarray, np.ndarray, np.ndarray]`: `(energies, strengths, widths)`

---

### `fit_lorentzian_mixture()`

```python
def fit_lorentzian_mixture(
    energy: Array,
    strength: Array,
    n_components: int,
    *,
    width_mode: Literal["global", "per_component"] = "global",
    eta_init: float = 0.5,
    min_spacing: float = 0.1,
    l2: float = 1e-3,
) -> LorentzianMixture
```

Fit a Lorentzian mixture to a strength function using nonlinear least squares.

**Parameters:**

- **energy** (`np.ndarray`): Energy grid, shape `(n_points,)`. Must be 1D and match `strength` length.

- **strength** (`np.ndarray`): Strength values, shape `(n_points,)`.

- **n_components** (`int`): Number of Lorentzian peaks to fit. Typically 2-5.

- **width_mode** (`Literal["global", "per_component"]`, default=`"global"`):
  - `"global"`: All resonances share a single width parameter (faster, fewer DOF).
  - `"per_component"`: Each resonance has independent width (more flexible).

- **eta_init** (`float`, default=`0.5`): Initial guess for width(s) in same units as `energy`.

- **min_spacing** (`float`, default=`0.1`): Minimum energy separation between adjacent resonances (prevents degeneracy).

- **l2** (`float`, default=`1e-3`): L2 regularization strength to prevent overfitting.

**Returns:**
- `LorentzianMixture`: Fitted mixture with optimized parameters.

**Raises:**
- `ValueError`: If `energy` and `strength` have mismatched shapes or dimensions, or if `n_components < 1`.

**Algorithm:**
- Uses SciPy `least_squares` with trust-region reflective method
- Enforces positivity via inverse softplus transformation
- Enforces energy ordering via cumulative gap construction
- Includes small L2 penalty for numerical stability

**Example:**
```python
import numpy as np
from smlr.lorentz import fit_lorentzian_mixture

# Generate synthetic data
energy = np.linspace(-3, 3, 200)
true_centers = np.array([-1.0, 1.0])
true_strengths = np.array([2.0, 1.5])
true_widths = np.array([0.3, 0.4])

from smlr.lorentz import lorentzian_sum
truth = lorentzian_sum(energy, true_centers, true_strengths, true_widths)

# Add noise
noisy = truth + 0.05 * np.random.randn(len(energy))

# Fit mixture
mixture = fit_lorentzian_mixture(
    energy, noisy, n_components=2,
    width_mode="per_component",
    eta_init=0.4,
    l2=1e-2
)

print(f"Fitted centers: {mixture.energies}")
print(f"Fitted strengths: {mixture.strengths}")
print(f"Fitted widths: {mixture.widths}")

# Evaluate fit quality
from smlr.metrics import normalized_l2
error = normalized_l2(mixture.evaluate(energy), truth, energy)
print(f"Fit error: {error:.4f}")
```

**See Also:**
- [Theory](theory.md#lorentzian-representation) for mathematical background.
- [Tutorial 5](tutorials.md#tutorial-5-custom-domains-and-advanced-fitting) for advanced usage.

---

### `lorentzian_sum()`

```python
def lorentzian_sum(
    energy: Array,
    centers: Array,
    strengths: Array,
    widths: Array
) -> Array
```

Evaluate a sum of Lorentzians (low-level function).

**Parameters:**
- **energy** (`np.ndarray`): Energy grid, shape `(n_points,)`.
- **centers** (`np.ndarray`): Resonance energies, shape `(n_components,)`.
- **strengths** (`np.ndarray`): Amplitudes, shape `(n_components,)`.
- **widths** (`np.ndarray`): Widths, shape `(n_components,)` or `(1,)`.

**Returns:**
- `np.ndarray`: Spectrum values, shape `(n_points,)`.

**Note:** Typically use `LorentzianMixture.evaluate()` instead. This is a lower-level utility.

---

## `smlr.emulator`

Surrogate model for predicting strength functions from parameters.

### `StrengthEmulator`

```python
class StrengthEmulator:
    def __init__(
        self,
        n_components: int,
        *,
        width_mode: Literal["global", "per_component"] = "global",
        regression_method: Literal["linear", "ridge", "polynomial", "gp"] = "linear",
        poly_degree: int = 2,
        alpha: float = 1e-3,
        random_state: int = 0,
    )
```

Learn parameter → strength function mapping via Lorentzian mixture emulation.

**Supports arbitrary parameter dimensions** (2D, 5D, 10D, 15D, etc.) for high-dimensional nuclear physics applications.

**Parameters:**

- **n_components** (`int`): Number of Lorentzian components to use in mixture representation.

- **width_mode** (`Literal["global", "per_component"]`, default=`"global"`):
  - `"global"`: All resonances share width (faster training, assumes similar resonance characteristics).
  - `"per_component"`: Independent widths per resonance (more flexible, use for multi-scale physics).

- **regression_method** (`Literal["linear", "ridge", "polynomial", "gp"]`, default=`"linear"`):
  - `"linear"`: Simple linear regression (fast, may underfit for nonlinear dependence).
  - `"ridge"`: L2-regularized regression (recommended default).
  - `"polynomial"`: Polynomial features + ridge (captures nonlinearity).
  - `"gp"`: Gaussian Process regression (best for uncertainty, slower).

- **poly_degree** (`int`, default=`2`): Polynomial degree when using `regression_method="polynomial"`.

- **alpha** (`float`, default=`1e-3`): Regularization strength for ridge/polynomial regression.

- **random_state** (`int`, default=`0`): Random seed for reproducibility.

**Attributes (after fitting):**

- **param_dim** (`int`): Dimension of the parameter space.
- **n_samples_seen** (`int`): Number of training samples used.
- **param_scaler** (`StandardScaler`): Scaler for input parameters.
- **energy_reg**, **strength_reg**, **width_reg**: Trained regression models.

**Methods:**

#### `fit()`

```python
def fit(
    self,
    dataset: StrengthDataset,
    *,
    mixtures: Optional[Sequence[LorentzianMixture]] = None,
    fitter: Callable[..., LorentzianMixture] = fit_lorentzian_mixture,
    fit_kwargs: Optional[dict] = None,
    normalize_strengths: bool = True,
    verbose: bool = False,
) -> StrengthEmulator
```

Train the emulator on a dataset.

**Parameters:**

- **dataset** (`StrengthDataset`): Training data containing spectra and parameter vectors. **Works with any parameter dimension**.

- **mixtures** (`Optional[Sequence[LorentzianMixture]]`): Pre-fitted Lorentzian mixtures (one per sample). If provided, skips fitting step (faster). Must align with `dataset.samples`.

- **fitter** (`Callable`, default=`fit_lorentzian_mixture`): Function to fit mixtures if `mixtures=None`. Should have signature `fitter(energy, strength, n_components, **kwargs) -> LorentzianMixture`.

- **fit_kwargs** (`Optional[dict]`): Keyword arguments passed to `fitter` (e.g., `{"l2": 1e-2, "eta_init": 0.6}`).

- **normalize_strengths** (`bool`, default=`True`): If `True`, apply log-transform to strengths during regression.

- **verbose** (`bool`, default=`False`): Whether to print progress during fitting.

**Returns:**
- `StrengthEmulator`: Returns `self` for method chaining.

**Example (2D parameters):**
```python
from smlr.data import StrengthDataset
from smlr.emulator import StrengthEmulator

emu = StrengthEmulator(n_components=3, regression_method="ridge")
emu.fit(dataset, verbose=True)
```

**Example (10D high-dimensional parameters):**
```python
# Works with any number of parameters
ds = StrengthDataset.from_folder(
    "metadata.csv",
    param_columns=["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10"]
)
emu = StrengthEmulator(n_components=5, regression_method="polynomial")
emu.fit(ds)

# Predict at a 10D point
new_params = np.random.rand(10)
mixture = emu.predict_mixture(new_params)
```

---

#### `predict_mixture()`

```python
def predict_mixture(self, params: Array) -> LorentzianMixture
```

Predict Lorentzian mixture parameters at a new parameter point.

**Parameters:**
- **params** (`np.ndarray`): Parameter vector, shape `(param_dim,)`. Must match training dimension.

**Returns:**
- `LorentzianMixture`: Predicted mixture with `n_components` resonances.

**Raises:**
- `RuntimeError`: If called before `fit()`.
- `ValueError`: If params dimension doesn't match training.

---

#### `predict_batch()`

```python
def predict_batch(
    self, 
    params_batch: Array, 
    energy_grid: Optional[Array] = None
) -> Tuple[List[LorentzianMixture], Optional[Array]]
```

Predict for multiple parameter points at once.

**Parameters:**
- **params_batch** (`np.ndarray`): Shape `(n_points, param_dim)`.
- **energy_grid** (`Optional[np.ndarray]`): If provided, also evaluate spectra.

**Returns:**
- **mixtures**: List of `LorentzianMixture`.
- **spectra**: Array of shape `(n_points, len(energy_grid))` if grid provided.

**Example:**
```python
# Predict at 1000 random parameter points
params_batch = np.random.rand(1000, 2)
mixtures, spectra = emu.predict_batch(params_batch, energy_grid)
```

---

#### `score()`

```python
def score(self, dataset: StrengthDataset, metric: str = "l2") -> float
```

Evaluate emulator accuracy on a test dataset.

**Parameters:**
- **dataset** (`StrengthDataset`): Test dataset.
- **metric** (`str`): `"l2"`, `"mse"`, or `"mae"`.

**Returns:**
- `float`: Mean error across all samples.

---

#### `predict_spectrum()`

```python
def predict_spectrum(self, params: Array, energy_grid: Array) -> Array
```

Predict strength function spectrum at a new parameter point.

**Parameters:**
- **params** (`np.ndarray`): Parameter vector, shape `(n_params,)`.
- **energy_grid** (`np.ndarray`): Energy points to evaluate at, shape `(n_points,)`.

**Returns:**
- `np.ndarray`: Predicted strength values, shape `(n_points,)`.

**Raises:**
- `RuntimeError`: If called before `fit()`.

**Example:**
```python
new_params = np.array([0.4, 0.8])
energy = np.linspace(-5, 5, 400)
spectrum = emu.predict_spectrum(new_params, energy)

import matplotlib.pyplot as plt
plt.plot(energy, spectrum)
plt.xlabel('Energy (MeV)')
plt.ylabel('Predicted Strength')
plt.savefig('prediction.png')
```

---

#### `predict()`

```python
def predict(
    self,
    params: Array,
    energy_grid: Optional[Array] = None
) -> Tuple[LorentzianMixture, Optional[Array]]
```

Unified prediction method.

**Parameters:**
- **params** (`np.ndarray`): Parameter vector, shape `(n_params,)`.
- **energy_grid** (`Optional[np.ndarray]`): Energy grid for spectrum evaluation. If `None`, returns mixture only.

**Returns:**
- **Tuple**:
  - `LorentzianMixture`: Predicted mixture parameters.
  - `Optional[np.ndarray]`: Predicted spectrum (if `energy_grid` provided) or `None`.

**Example:**
```python
# Get both mixture and spectrum
mixture, spectrum = emu.predict(new_params, energy_grid)

# Get mixture only
mixture, _ = emu.predict(new_params)
```

---

## `smlr.pmm`

Parametric Matrix Model for physics-based emulation.

### `PMMConfig`

```python
@dataclass
class PMMConfig:
    n_poles: int = 10
    width_mode: Literal["global", "per_component", "parametric"] = "global"
    optimizer: str = "scipy"
    max_iterations: int = 2000
    tolerance: float = 1e-8
    regularization: float = 1e-6
    verbose: bool = False
```

Configuration for Parametric Matrix Model.

**Attributes:**

- **n_poles** (`int`): Size of the response matrix (number of poles).
- **width_mode** (`str`): Width handling: "global", "per_component", or "parametric".
- **optimizer** (`str`): Optimization backend: "scipy", "tensorflow", "jax".
- **max_iterations** (`int`): Maximum optimization iterations.
- **tolerance** (`float`): Convergence tolerance.
- **regularization** (`float`): L2 regularization on matrix elements.
- **verbose** (`bool`): Print progress during training.

---

### `PMMResult`

```python
@dataclass
class PMMResult:
    eigenvalues: Array
    strengths: Array
    width: Union[float, Array]
    spectrum: Array
    energy: Array
```

Container for PMM prediction results.

**Attributes:**

- **eigenvalues** (`np.ndarray`): Pole energies from diagonalization.
- **strengths** (`np.ndarray`): Transition strengths (squared projections).
- **width** (`float` or `np.ndarray`): Pole width(s).
- **spectrum** (`np.ndarray`): Reconstructed strength function.
- **energy** (`np.ndarray`): Energy grid used.

---

### `ParametricMatrixModel`

```python
class ParametricMatrixModel:
    def __init__(
        self,
        config: Optional[PMMConfig] = None,
        n_poles: Optional[int] = None,
        **kwargs,
    )
```

Parametric Matrix Model emulator for strength functions.

This emulator learns a reduced-order response matrix whose eigenvalue
structure reproduces the training spectra. Unlike regression-based
approaches, PMM maintains the algebraic structure of linear response
theory, leading to better extrapolation and sum rule preservation.

**Parameters:**

- **config** (`PMMConfig`, optional): Configuration object.
- **n_poles** (`int`, optional): Number of poles (matrix dimension).
- **\*\*kwargs**: Additional configuration options passed to `PMMConfig`.

**Attributes:**

- **param_dim** (`int`): Dimension of the parameter space (set after fitting).
- **reference_point** (`np.ndarray`): Central parameter point p₀ for the expansion.
- **D** (`np.ndarray`): Diagonal matrix (n_poles,).
- **S** (`list`): Perturbation matrices, one per parameter dimension.
- **v0** (`np.ndarray`): External field vector (n_poles,).
- **eta** (`float` or `np.ndarray`): Global or per-component width.

**Methods:**

#### `fit()`

```python
def fit(
    self,
    dataset: StrengthDataset,
    reference_point: Optional[Array] = None,
    sample_indices: Optional[Sequence[int]] = None,
    observable_weight: float = 0.0,
    observable_targets: Optional[Dict[str, Array]] = None,
) -> "ParametricMatrixModel"
```

Fit the PMM to a dataset of strength functions.

**Parameters:**

- **dataset** (`StrengthDataset`): Training dataset.
- **reference_point** (`np.ndarray`, optional): Central parameter point p₀.
- **sample_indices** (`Sequence[int]`, optional): Indices for subsampling.
- **observable_weight** (`float`): Weight for observable-based loss.
- **observable_targets** (`dict`, optional): Target observable values.

**Returns:**
- `ParametricMatrixModel`: Fitted model (self).

#### `predict()`

```python
def predict(
    self,
    params: Array,
    energy: Optional[Array] = None,
) -> PMMResult
```

Predict strength function at a new parameter point.

**Parameters:**

- **params** (`np.ndarray`): Parameter vector.
- **energy** (`np.ndarray`, optional): Energy grid.

**Returns:**
- `PMMResult`: Prediction with spectrum and pole information.

#### `get_eigenvalues()`

```python
def get_eigenvalues(self, params: Array) -> Tuple[Array, Array]
```

Get eigenvalues and strengths without computing full spectrum.

**Returns:**
- Tuple of (eigenvalues, strengths).

#### `model_diagnostics()`

```python
def model_diagnostics(self) -> Dict[str, Any]
```

Return diagnostic information about the fitted model.

**Example:**
```python
from smlr.pmm import ParametricMatrixModel
import numpy as np

# Create and fit PMM
pmm = ParametricMatrixModel(n_poles=10)
pmm.fit(dataset, reference_point=np.array([0.5, 0.5]))

# Predict at new point
result = pmm.predict(np.array([0.7, 0.3]), energy_grid)
print(f"Poles: {result.eigenvalues}")
print(f"Strengths: {result.strengths}")
```

---

### `compare_emulation_methods()`

```python
def compare_emulation_methods(
    dataset: StrengthDataset,
    n_components: int = 4,
    test_fraction: float = 0.2,
    seed: int = 42,
) -> Dict[str, Any]
```

Compare regression-based and PMM emulation on a dataset.

**Parameters:**

- **dataset** (`StrengthDataset`): Dataset to use.
- **n_components** (`int`): Number of poles/components.
- **test_fraction** (`float`): Fraction for testing.
- **seed** (`int`): Random seed.

**Returns:**
- `dict`: Comparison results with errors and timing.

---

## `smlr.backends`

Factory functions for creating emulators with different backends.

### `get_emulator()`

```python
def get_emulator(
    backend: str = "regression",
    n_components: Optional[int] = None,
    n_poles: Optional[int] = None,
    **kwargs,
) -> Union[StrengthEmulator, ParametricMatrixModel]
```

Create an emulator with the specified backend.

**Parameters:**

- **backend** (`str`): Emulation method:
    - `"regression"` or `"lorentzian"`: Regression-based emulator.
    - `"pmm"` or `"matrix"`: Parametric Matrix Model.
- **n_components** (`int`, optional): Number of Lorentzian components (regression).
- **n_poles** (`int`, optional): Matrix dimension (PMM).
- **\*\*kwargs**: Additional arguments for the emulator.

**Returns:**
- `StrengthEmulator` or `ParametricMatrixModel`: Configured emulator.

**Example:**
```python
from smlr.backends import get_emulator

# Fast regression-based emulation
emu = get_emulator("regression", n_components=4, regression_method="ridge")

# Physics-based PMM emulation
emu = get_emulator("pmm", n_poles=10, max_iterations=5000)
```

---

### `list_backends()`

```python
def list_backends() -> dict
```

List available emulation backends with descriptions.

**Returns:**
- `dict`: Backend names mapped to descriptions.

---

## `smlr.metrics`

Evaluation metrics for quantifying emulator accuracy.

### `normalized_l2()`

```python
def normalized_l2(y_pred: Array, y_true: Array, x: Array) -> float
```

Compute normalized L² error between predicted and true spectra.

**Parameters:**
- **y_pred** (`np.ndarray`): Predicted values, shape `(n_points,)`.
- **y_true** (`np.ndarray`): True values, shape `(n_points,)`.
- **x** (`np.ndarray`): Grid points (e.g., energy), shape `(n_points,)`.

**Returns:**
- `float`: Normalized L² error: $\|y_{\text{pred}} - y_{\text{true}}\|_2 / \|y_{\text{true}}\|_2$.

**Formula:**

$$\epsilon = \frac{\sqrt{\int (y_{\text{pred}} - y_{\text{true}})^2 dx}}{\sqrt{\int y_{\text{true}}^2 dx}}$$

Numerical integration via trapezoidal rule.

**Interpretation:**
- < 0.01: Excellent agreement
- 0.01-0.1: Good  
- 0.1-0.5: Acceptable
- \> 0.5: Poor

**Example:**
```python
from smlr.metrics import normalized_l2

error = normalized_l2(prediction, ground_truth, energy)
print(f"Normalized L² error: {error:.4f}")
```

---

### `mean_absolute_relative_error()`

```python
def mean_absolute_relative_error(y_pred: Array, y_true: Array) -> float
```

Compute mean absolute relative error.

**Parameters:**
- **y_pred** (`np.ndarray`): Predicted values.
- **y_true** (`np.ndarray`): True values.

**Returns:**
- `float`: MARE = mean(|y_pred - y_true| / |y_true|).

**Example:**
```python
from smlr.metrics import mean_absolute_relative_error

mare = mean_absolute_relative_error(prediction, ground_truth)
print(f"MARE: {mare:.4f}")
```

---

## `smlr.plotting`

Visualization utilities for strength functions (headless, Agg backend).

**Note:** All functions return `matplotlib.pyplot.Figure` objects. You must save them explicitly:

```python
fig = plot_spectrum(energy, strength)
fig.savefig("output.png", dpi=150, bbox_inches="tight")
```

### `plot_spectrum()`

```python
def plot_spectrum(
    energy: Array,
    strength: Array,
    *,
    label: str = "data"
) -> plt.Figure
```

Plot a single strength function.

**Parameters:**
- **energy** (`np.ndarray`): Energy grid.
- **strength** (`np.ndarray`): Strength values.
- **label** (`str`, default=`"data"`): Legend label.

**Returns:**
- `plt.Figure`: Matplotlib figure object.

**Example:**
```python
from smlr.plotting import plot_spectrum

fig = plot_spectrum(energy, strength, label="QRPA calculation")
fig.savefig("spectrum.png")
```

---

### `plot_comparison()`

```python
def plot_comparison(
    energy: Array,
    truth: Array,
    prediction: Array,
    *,
    title: str = "Strength comparison",
    labels: Tuple[str, str] = ("reference", "predicted"),
) -> plt.Figure
```

Plot predicted vs. true strength functions.

**Parameters:**
- **energy** (`np.ndarray`): Energy grid.
- **truth** (`np.ndarray`): True/reference strength.
- **prediction** (`np.ndarray`): Predicted strength.
- **title** (`str`): Plot title.
- **labels** (`Tuple[str, str]`): Legend labels for (truth, prediction).

**Returns:**
- `plt.Figure`: Figure with both spectra.

**Example:**
```python
from smlr.plotting import plot_comparison
from smlr.metrics import normalized_l2

error = normalized_l2(pred, truth, energy)
fig = plot_comparison(
    energy, truth, pred,
    title=f"Emulator (L²={error:.3f})",
    labels=("QRPA", "Emulator")
)
fig.savefig("comparison.png")
```

---

### `plot_mixture_components()`

```python
def plot_mixture_components(
    mixture: LorentzianMixture,
    energy: Array
) -> plt.Figure
```

Plot individual Lorentzian components and their sum.

**Parameters:**
- **mixture** (`LorentzianMixture`): Fitted mixture.
- **energy** (`np.ndarray`): Energy grid for evaluation.

**Returns:**
- `plt.Figure`: Figure showing all components plus total.

**Example:**
```python
from smlr.plotting import plot_mixture_components

fig = plot_mixture_components(mixture, energy)
fig.savefig("components.png")
```

---

## Type Aliases

Throughout the API, `Array` refers to `np.ndarray` from NumPy.

---

## `smlr.observables`

Physics-informed observable functions for computing derived quantities from strength functions.

### `Observable`

Abstract base class for all observables.

```python
class Observable(abc.ABC):
    def __init__(
        self,
        name: str,
        weight: float = 1.0,
        target: Optional[float] = None,
        normalize_loss: bool = True,
    )
```

**Methods:**

- `compute(energy, strength, **kwargs) -> float`: Compute the observable value.
- `loss(energy, strength, **kwargs) -> float`: Compute loss term for optimization (weighted squared error if target is set).

---

### `SumRule`

Energy-weighted sum rule $m_k = \int E^k S(E) dE$.

```python
class SumRule(Observable):
    def __init__(
        self,
        k: int = 0,
        name: Optional[str] = None,
        energy_min: Optional[float] = None,
        energy_max: Optional[float] = None,
        **kwargs
    )
```

**Parameters:**
- **k** (`int`): Power of energy weighting (-1, 0, 1, 2, ...).
- **energy_min/max** (`float`, optional): Integration limits.

**Example:**
```python
from smlr.observables import SumRule

# Thomas-Reiche-Kuhn sum rule (m_1)
trk = SumRule(k=1, name="TRK", target=875.0, weight=1.0)
value = trk.compute(energy, strength)
```

---

### `DipolePolarizability`

Dipole polarizability $\alpha_D = \frac{8\pi \alpha_c \hbar c}{9} \int \frac{S(E)}{E} dE$.

```python
class DipolePolarizability(Observable):
    def __init__(
        self,
        hbar_c: float = 197.33,
        alpha_c: float = 1/137,
        energy_min: float = 1.0,
        name: str = "alpha_D",
        **kwargs
    )
```

**Example:**
```python
from smlr.observables import DipolePolarizability

alphaD = DipolePolarizability(target=18.5, weight=0.5)
value = alphaD.compute(energy, strength)

# More efficient for pole decomposition:
value = alphaD.compute_from_poles(pole_energies, pole_strengths)
```

---

### `BetaDecayHalfLife`

Beta decay half-life from Gamow-Teller strength function.

```python
class BetaDecayHalfLife(Observable):
    def __init__(
        self,
        Z: int,
        A: int,
        Q_value: float,
        g_A: float = 1.27,
        kappa: float = 6147.0,
        name: str = "t_half",
        **kwargs
    )
```

**Example:**
```python
from smlr.observables import BetaDecayHalfLife

half_life = BetaDecayHalfLife(Z=28, A=80, Q_value=5.0, target=0.1)
t_half = half_life.compute_from_poles(pole_energies, pole_strengths)
```

---

### `ObservableSet`

Collection of observables for multi-objective optimization.

```python
@dataclass
class ObservableSet:
    observables: List[Observable]
    
    def compute_all(self, energy, strength, **kwargs) -> List[ObservableResult]
    def total_loss(self, energy, strength, **kwargs) -> float
    def to_dict(self, energy, strength, **kwargs) -> Dict[str, float]
```

**Example:**
```python
from smlr.observables import ObservableSet, SumRule, DipolePolarizability

obs_set = ObservableSet([
    SumRule(k=1, target=875.0, weight=1.0),
    DipolePolarizability(target=18.5, weight=0.5),
])

results = obs_set.compute_all(energy, strength)
total_loss = obs_set.total_loss(energy, strength)
```

---

## `smlr.optimization`

Multiple optimization backends for emulator training.

### `OptimizerBackend`

Enum of available backends.

```python
class OptimizerBackend(Enum):
    SCIPY = auto()      # Always available
    TENSORFLOW = auto() # Requires pip install smlr[tensorflow]
    JAX = auto()        # Requires pip install smlr[jax]
```

---

### `OptimizerConfig`

Configuration dataclass for optimizers.

```python
@dataclass
class OptimizerConfig:
    backend: OptimizerBackend = OptimizerBackend.SCIPY
    method: str = "L-BFGS-B"  # Backend-specific method
    max_iterations: int = 1000
    learning_rate: float = 0.01  # For TF/JAX
    tolerance: float = 1e-6
    patience: int = 100  # Early stopping
    min_delta: float = 1e-4
    verbose: bool = False
    random_seed: int = 42
```

---

### `create_optimizer()`

Factory function to create optimizers.

```python
def create_optimizer(
    backend: Union[str, OptimizerBackend] = "scipy",
    method: str = "L-BFGS-B",
    **kwargs
) -> BaseOptimizer
```

**Parameters:**
- **backend**: `"scipy"`, `"tensorflow"`, or `"jax"`.
- **method**: Optimization method.
  - Scipy: `"L-BFGS-B"`, `"TNC"`, `"SLSQP"`
  - TF/JAX: `"adam"`, `"sgd"`, `"rmsprop"`, `"adamw"`

**Example:**
```python
from smlr.optimization import create_optimizer, get_available_backends

# Check what's available
print(get_available_backends())  # ['scipy', 'tensorflow', 'jax']

# Create optimizers
scipy_opt = create_optimizer("scipy", method="L-BFGS-B", max_iterations=1000)
adam_opt = create_optimizer("jax", method="adam", learning_rate=0.001)

# Run optimization
result = adam_opt.minimize(cost_fn, initial_params)
print(f"Converged: {result.converged}, Cost: {result.cost:.4f}")
```

---

### `get_available_backends()`

Return list of available optimization backends.

```python
def get_available_backends() -> List[str]
```

---

## See Also

- **[Usage Guide](usage.md)**: Practical workflow examples
- **[Tutorials](tutorials.md)**: Step-by-step guides
- **[Theory](theory.md)**: Mathematical background
- **[FAQ](faq.md)**: Common questions and troubleshooting

---

**Last Updated**: 2025-01-XX
