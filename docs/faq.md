# Frequently Asked Questions (FAQ)

Common questions and troubleshooting for SMLR.

## Table of Contents

- [General](#general)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Training and Fitting](#training-and-fitting)
- [Prediction and Evaluation](#prediction-and-evaluation)
- [Performance](#performance)
- [Troubleshooting](#troubleshooting)

---

## General

### What is SMLR best suited for?

SMLR excels at emulating **resonance-dominated** spectral functions where:
- Peaks have approximate Lorentzian shapes (finite-lifetime resonances)
- You need fast predictions across parameter space
- Interpretability matters (want to track resonance positions/widths)
- Training data is limited (works with 10-100 samples)

**Good fits**: Beta decay, dipole response, giant resonances, photoabsorption  
**Poor fits**: Smooth continuum spectra without clear peaks, highly oscillatory functions

### How many training samples do I need?

**Rule of thumb**: 10-20 samples per parameter dimension.

- **1D parameter**: 10-20 spectra
- **2D parameter**: 50-100 spectra
- **3D parameter**: 200+ spectra

Quality matters more than quantity. Sample the parameter space uniformly or use Latin hypercube sampling.

### Can SMLR handle extrapolation?

**Limited extrapolation** is possible but risky. Emulator accuracy degrades rapidly outside the training region. Best practice:
- Train on a slightly larger parameter range than needed
- Validate extrapolation performance explicitly
- Use ensemble methods for uncertainty estimates

### How do I choose the number of components (K)?

Start with K = number of visible peaks in your spectra. Then:
1. Try K ± 1 and compare validation errors
2. Use cross-validation (see [Tutorial 3](tutorials.md#tutorial-3-cross-validation-and-model-selection))
3. Inspect residuals: systematic structure suggests K is too low

Typical range: K = 2-5. Going beyond K = 7 rarely helps and may overfit.

---

## Installation

### `uv` vs. `pip`: which should I use?

**uv (recommended)**:
- Faster dependency resolution
- Reproducible lockfiles
- Better environment management
- Modern tool actively maintained

**pip (fallback)**:
- More familiar to many users
- Works everywhere
- ⚠️ Slower dependency resolution

Both work fine. Use `uv` for development, `pip` for quick installs.

### Why do I get import errors after installation?

**Common causes**:
1. **Wrong environment**: Ensure you activated the venv:
   ```bash
   source .venv/bin/activate  # macOS/Linux
   .venv\Scripts\activate     # Windows
   ```

2. **Editable install missing**: Run:
   ```bash
   pip install -e .  # or: uv sync
   ```

3. **Old Python version**: SMLR requires Python ≥ 3.9:
   ```bash
   python --version
   ```

### Can I install SMLR from PyPI?

Not yet. SMLR is currently installable via:
```bash
git clone https://github.com/ascsn/SMLR.git
cd SMLR
pip install -e .
```

A PyPI release is planned for v1.0.

---

## Data Preparation

### My energy grids differ across samples. Is that okay?

**Yes!** SMLR automatically interpolates when you call `to_matrix()` or fit the emulator. Just ensure grids span similar energy ranges.

### Should I normalize my strength functions?

**Use `normalize=True` when**:
- Strengths vary by orders of magnitude across samples
- You care about spectral *shape* more than absolute magnitude
- Regression convergence is slow

**Keep normalization off (`normalize=False`) when**:
- Absolute strengths are physically meaningful
- All spectra have similar scales
- You'll denormalize after prediction anyway

### How do I handle missing data or gaps?

**Options**:
1. **Interpolation**: Fill gaps with linear/spline interpolation before creating dataset
2. **Masking**: Exclude problematic energy regions from all samples
3. **Regularization**: Increase L2 penalty in fitting to smooth over gaps

```python
# Example: Mask energy region
mask = (energy > -5) & (energy < 5)
energy_masked = energy[mask]
strength_masked = strength[mask]
```

### Can I use SMLR for 1D parameter scans?

**Absolutely!** SMLR handles arbitrary parameter dimensions, including 1D:

```python
params = np.array([[0.1], [0.2], [0.3], [0.4]])  # Shape: (4, 1)
dataset = StrengthDataset.from_arrays(params, energy, strengths)
```

---

## Training and Fitting

### Fitting fails with "optimization did not converge"

**Causes and solutions**:

1. **Poor initialization**:
   ```python
   # Try different eta_init (initial width guess)
   from smlr.lorentz import fit_lorentzian_mixture
   mixture = fit_lorentzian_mixture(energy, strength, n_components=3, eta_init=1.0)
   ```

2. **Too many components**: Reduce K
3. **Noisy data**: Increase L2 regularization:
   ```python
   mixture = fit_lorentzian_mixture(energy, strength, n_components=2, l2=1e-2)
   ```

4. **Narrow peaks**: Decrease `min_spacing`:
   ```python
   mixture = fit_lorentzian_mixture(energy, strength, n_components=3, min_spacing=0.05)
   ```

### Training is very slow. How can I speed it up?

**Strategies**:

1. **Pre-fit mixtures** (one-time cost):
   ```python
   from smlr.lorentz import fit_lorentzian_mixture
   
   mixtures = []
   for sample in dataset.samples:
       mix = fit_lorentzian_mixture(sample.energy, sample.strength, n_components=3)
       mixtures.append(mix)
   
   # Pass to emulator (skips fitting during training)
   emu.fit(dataset, mixtures=mixtures)
   ```

2. **Reduce grid resolution** during fitting
3. **Use fewer components** (K)
4. **Parallel fitting** (see [Development Guide](development.md))

### What does `width_mode` do?

**`width_mode="global"`** (default):
- All resonances share a single width parameter
- Faster training, fewer parameters
- Use when resonances are similar (e.g., same physics)

**`width_mode="per_component"`**:
- Each resonance has independent width
- More flexible, can capture multi-scale physics
- Use for disparate resonance families (e.g., narrow valence + broad giant resonances)

Example:
```python
# Giant dipole (broad) + pygmy dipole (narrow)
emu = StrengthEmulator(n_components=2, width_mode="per_component")
```

---

## Prediction and Evaluation

### Predictions look nothing like training data!

**Debugging checklist**:

1. **Check parameter ranges**: Are test parameters inside the training region?
   ```python
   train_params = dataset.parameters()
   print(f"Training range: {train_params.min(axis=0)} to {train_params.max(axis=0)}")
   print(f"Test params: {test_params}")
   ```

2. **Inspect fitted mixtures**:
   ```python
   # Look at what was actually fitted
   sample = dataset.samples[0]
   from smlr.lorentz import fit_lorentzian_mixture
   mix = fit_lorentzian_mixture(sample.energy, sample.strength, n_components=3)
   print(f"Centers: {mix.energies}, Widths: {mix.widths}")
   ```

3. **Visualize training fit**:
   ```python
   from smlr.plotting import plot_mixture_components
   fig = plot_mixture_components(mix, sample.energy)
   fig.savefig("debug_fit.png")
   ```

4. **Check for NaNs**:
   ```python
   assert not np.any(np.isnan(spectrum)), "NaN in prediction!"
   ```

### How do I quantify emulator accuracy?

**Validation workflow**:

```python
from smlr.metrics import normalized_l2

# 1. Split data
train_ds, _, test_ds = dataset.train_val_test_split(train=0.7, val=0.0, seed=42)

# 2. Train
emu.fit(train_ds)

# 3. Evaluate on held-out data
errors = []
for sample in test_ds.samples:
    _, pred = emu.predict(sample.params, sample.energy)
    error = normalized_l2(pred, sample.strength, sample.energy)
    errors.append(error)

mean_error = np.mean(errors)
print(f"Test set error: {mean_error:.4f} ± {np.std(errors):.4f}")
```

**Interpreting errors**:
- < 0.01: Excellent
- 0.01-0.1: Good
- 0.1-0.5: Acceptable for many applications
- \> 0.5: Poor, needs more data or components

### Can I get prediction uncertainties?

**Bootstrap ensemble** approach:

```python
n_bootstrap = 20
predictions = []

for i in range(n_bootstrap):
    # Resample training data
    indices = np.random.choice(len(dataset), size=len(dataset), replace=True)
    bootstrap_samples = [dataset.samples[i] for i in indices]
    bootstrap_ds = StrengthDataset(bootstrap_samples)
    
    # Train and predict
    emu_i = StrengthEmulator(n_components=3, random_state=i)
    emu_i.fit(bootstrap_ds)
    _, pred_i = emu_i.predict(test_params, energy)
    predictions.append(pred_i)

# Compute statistics
predictions = np.array(predictions)
mean_pred = predictions.mean(axis=0)
std_pred = predictions.std(axis=0)

# Plot with uncertainty
import matplotlib.pyplot as plt
plt.plot(energy, mean_pred, label='Mean')
plt.fill_between(energy, mean_pred - 2*std_pred, mean_pred + 2*std_pred, alpha=0.3)
```

---

## Performance

### How fast are predictions?

**Typical performance** (laptop CPU):
- Single prediction: **~0.1-1 ms**
- 1000 predictions: **~1 second**
- vs. QRPA calculation: **~1000-10000× speedup**

Depends on:
- Number of components (K)
- Energy grid size
- Parameter dimension

### Can I use GPUs?

**Not needed.** SMLR is designed for CPU-only execution because:
- Predictions are already very fast
- Training is one-time, small-scale
- Avoids GPU dependencies

For truly massive-scale problems (>1M predictions), consider:
- Vectorizing predictions with NumPy broadcasting
- Parallelizing across CPU cores with `multiprocessing`
- Using Numba for JIT compilation

### How do I parallelize training across many spectra?

**Pre-fit mixtures in parallel**:

```python
from multiprocessing import Pool
from smlr.lorentz import fit_lorentzian_mixture

def fit_one(sample):
    return fit_lorentzian_mixture(sample.energy, sample.strength, n_components=3)

with Pool(processes=8) as pool:
    mixtures = pool.map(fit_one, dataset.samples)

emu.fit(dataset, mixtures=mixtures)
```

**Note**: Emulator training itself (regression) is already very fast (< 1 second).

---

## Troubleshooting

### Import error: "No module named 'smlr'"

**Solution**: Install in editable mode:
```bash
pip install -e .
# or
uv sync
```

### Tests fail with "FileNotFoundError"

**Cause**: Running tests from wrong directory.

**Solution**:
```bash
cd /path/to/SMLR  # Repository root
pytest
```

### Plots don't show up

**Expected behavior.** SMLR uses headless matplotlib (Agg backend) for reproducibility. Plots are saved to files:

```python
fig = plot_comparison(energy, truth, pred)
fig.savefig("output.png")  # ← Always save
```

To view interactively (development only):
```python
import matplotlib
matplotlib.use("TkAgg")  # Before importing pyplot
import matplotlib.pyplot as plt
```

### "LinAlgError: SVD did not converge"

**Cause**: Ill-conditioned regression problem.

**Solutions**:
1. **Normalize strengths**:
   ```python
   emu.fit(dataset, normalize_strengths=True)
   ```

2. **Reduce parameter range**:
   - Remove outliers from training data

3. **Add regularization**:
   - Already included by default (L2 in Lorentzian fitting)

4. **Check for duplicate samples**:
   ```python
   params = dataset.parameters()
   assert len(np.unique(params, axis=0)) == len(params), "Duplicate parameter points!"
   ```

### Emulator predicts negative strengths

**Cause**: Extrapolation or poor mixture fit.

**Solutions**:
1. **Clip to zero** (post-processing):
   ```python
   _, spectrum = emu.predict(params, energy)
   spectrum = np.maximum(spectrum, 0)
   ```

2. **Check training region**:
   - Ensure test parameters are within training bounds

3. **Inspect mixture**:
   ```python
   mix = emu.predict_mixture(params)
   print(mix.strengths)  # Should all be positive
   ```

4. **Increase L2 regularization** during fitting

### Memory error with large datasets

**Solutions**:

1. **Reduce grid resolution**:
   ```python
   energy_coarse = np.linspace(energy.min(), energy.max(), 100)  # vs. 1000
   ```

2. **Process in batches**:
   ```python
   batch_size = 10
   for i in range(0, len(dataset), batch_size):
       batch_samples = dataset.samples[i:i+batch_size]
       batch_ds = StrengthDataset(batch_samples)
       # Fit on batch
   ```

3. **Use sparse grids** or adaptive sampling

---

## Still Need Help?

- **Check documentation**: [Usage Guide](usage.md), [API Reference](api.md), [Tutorials](tutorials.md)
- **Search issues**: [GitHub Issues](https://github.com/ascsn/SMLR/issues)
- **Ask a question**: [GitHub Discussions](https://github.com/ascsn/SMLR/discussions)
- **Report a bug**: [New Issue](https://github.com/ascsn/SMLR/issues/new)

When asking for help, please provide:
- SMLR version: Check `smlr.__version__`
- Python version: `python --version`
- OS and hardware
- Minimal reproducible example
- Full error traceback
