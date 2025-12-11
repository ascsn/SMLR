# Tutorials

Step-by-step guides for common SMLR workflows.

## Table of Contents

- [Tutorial 1: Your First Emulator](#tutorial-1-your-first-emulator)
- [Tutorial 2: Loading Data from Files](#tutorial-2-loading-data-from-files)
- [Tutorial 3: Cross-Validation and Model Selection](#tutorial-3-cross-validation-and-model-selection)
- [Tutorial 4: Batch Predictions and Parameter Scans](#tutorial-4-batch-predictions-and-parameter-scans)
- [Tutorial 5: Custom Domains and Advanced Fitting](#tutorial-5-custom-domains-and-advanced-fitting)

---

## Tutorial 1: Your First Emulator

**Goal**: Train an emulator on synthetic data and visualize predictions.

**Time**: 5 minutes

### Step 1: Generate Synthetic Data

```python
import numpy as np
from smlr.lorentz import lorentzian_sum

# Define a parameter-dependent strength function
def synthetic_strength(params, energy):
    """Two-resonance system with parameter-dependent positions."""
    alpha, beta = params
    centers = np.array([-1.0 + 0.5 * alpha, 1.0 + 0.3 * beta])
    strengths = np.array([2.0 + 0.4 * alpha, 1.5 + 0.2 * beta])
    widths = np.array([0.3, 0.4])
    return lorentzian_sum(energy, centers, strengths, widths)

# Create a training grid
energy = np.linspace(-3, 3, 200)
param_grid = np.array([[0.2, 0.3], [0.5, 0.6], [0.8, 0.9], [0.3, 0.8]])
strengths = np.array([synthetic_strength(p, energy) for p in param_grid])
```

### Step 2: Create a Dataset

```python
from smlr.data import StrengthDataset

dataset = StrengthDataset.from_arrays(
    params=param_grid,
    energy=energy,
    strengths=strengths,
    labels=[f"point_{i}" for i in range(len(param_grid))]
)

print(f"Dataset size: {len(dataset)}")
print(f"Parameter dimension: {dataset.param_dim}")
```

### Step 3: Train the Emulator

```python
from smlr import Surrogate

# Create and fit emulator using the Surrogate wrapper
model = Surrogate(
    "regression",             # Backend: "regression" or "pmm"  
    n_components=2,           # Number of Lorentzian peaks
    width_mode="global",      # Shared width across resonances
    random_state=42           # Reproducibility
)

model.fit(dataset)
print("Emulator trained!")
```

### Step 4: Make Predictions

```python
# Predict at a new parameter point
new_params = np.array([0.4, 0.5])
result = model.predict(new_params, energy)

# Ground truth for comparison
true_spectrum = synthetic_strength(new_params, energy)

print(f"Predicted spectrum shape: {result.spectrum.shape}")
print(f"Pole positions: {result.poles}")
print(f"Pole strengths: {result.strengths}")
print(f"Pole widths: {result.widths}")
```

### Step 5: Visualize Results

```python
from smlr.plotting import plot_comparison
from smlr.metrics import normalized_l2

# Compute error
error = normalized_l2(result.spectrum, true_spectrum, energy)
print(f"Normalized L² error: {error:.4f}")

# Plot comparison
fig = plot_comparison(
    energy, 
    true_spectrum, 
    result.spectrum,
    title=f"Emulator Performance (L² = {error:.3f})",
    labels=("Ground Truth", "Emulator")
)
fig.savefig("tutorial1_result.png", dpi=150, bbox_inches="tight")
print("Plot saved to tutorial1_result.png")
```

**Expected output**: Error < 0.05 (5% normalized L²)

---

## Tutorial 2: Loading Data from Files

**Goal**: Load strength functions from CSV files with metadata.

### Step 1: Prepare Your Data Files

Create a directory structure:
```
my_data/
├── metadata.csv
└── spectra/
    ├── strength_0.1_0.5.txt
    ├── strength_0.2_0.6.txt
    └── strength_0.3_0.7.txt
```

**metadata.csv**:
```csv
param_alpha,param_beta,spectrum_file,label
0.1,0.5,spectra/strength_0.1_0.5.txt,run1
0.2,0.6,spectra/strength_0.2_0.6.txt,run2
0.3,0.7,spectra/strength_0.3_0.7.txt,run3
```

**Spectrum files** (whitespace-separated):
```
# Energy  Strength
-3.0     0.001
-2.9     0.002
...
```

### Step 2: Load the Dataset

```python
from pathlib import Path
from smlr.data import StrengthDataset

dataset = StrengthDataset.from_folder(
    metadata_csv="my_data/metadata.csv",
    spectrum_column="spectrum_file",
    param_columns=["param_alpha", "param_beta"],
    label_column="label",
    root="my_data",  # Base directory for relative paths
    sep=","  # CSV separator
)

print(f"Loaded {len(dataset)} samples")
```

### Step 3: Inspect the Data

```python
# Check parameter matrix
params = dataset.parameters()
print(f"Parameter matrix shape: {params.shape}")
print(f"Parameter ranges:")
print(f"  Alpha: [{params[:, 0].min():.2f}, {params[:, 0].max():.2f}]")
print(f"  Beta:  [{params[:, 1].min():.2f}, {params[:, 1].max():.2f}]")

# Check energy grids
grids = dataset.energy_grids()
print(f"Energy grid lengths: {[len(g) for g in grids]}")

# Access individual samples
sample = dataset.samples[0]
print(f"Sample label: {sample.label}")
print(f"Sample params: {sample.params}")
```

### Step 4: Normalize if Needed

```python
# If strengths vary by orders of magnitude
params, energy, strengths = dataset.to_matrix(normalize=True)
print(f"Normalized strength matrix shape: {strengths.shape}")
```

**Key Points**:
- Spectrum files can be tab/space/comma-separated
- Energy grids can differ across samples (auto-interpolated)
- Use `normalize=True` when absolute scales vary
- Labels are optional but helpful for debugging

---

## Tutorial 3: Cross-Validation and Model Selection

**Goal**: Choose the optimal number of Lorentzian components.

### Step 1: Create a Validation Function

```python
import numpy as np
from smlr import Surrogate, StrengthDataset
from smlr.metrics import normalized_l2

def cross_validate(dataset, n_components, n_folds=3, random_state=42):
    """Perform K-fold cross-validation."""
    n_samples = len(dataset)
    indices = np.arange(n_samples)
    rng = np.random.default_rng(random_state)
    rng.shuffle(indices)
    
    fold_size = n_samples // n_folds
    errors = []
    
    for fold in range(n_folds):
        # Split into train/validation
        val_start = fold * fold_size
        val_end = val_start + fold_size
        val_indices = indices[val_start:val_end]
        train_indices = np.concatenate([indices[:val_start], indices[val_end:]])
        
        train_samples = [dataset.samples[i] for i in train_indices]
        val_samples = [dataset.samples[i] for i in val_indices]
        
        train_ds = StrengthDataset(train_samples)
        val_ds = StrengthDataset(val_samples)
        
        # Train emulator
        model = Surrogate(
            "regression",
            n_components=n_components,
            width_mode="global",
            random_state=random_state
        )
        model.fit(train_ds)
        
        # Evaluate on validation set
        energy = val_ds.energy_grids()[0]
        for sample in val_ds.samples:
            result = model.predict(sample.params, energy)
            truth = np.interp(energy, sample.energy, sample.strength)
            error = normalized_l2(result.spectrum, truth, energy)
            errors.append(error)
    
    return np.mean(errors), np.std(errors)
```

### Step 2: Grid Search Over Components

```python
# Assuming you have a dataset loaded
component_range = range(1, 8)  # Test 1 to 7 components
results = []

for K in component_range:
    mean_error, std_error = cross_validate(dataset, K, n_folds=3)
    results.append((K, mean_error, std_error))
    print(f"K={K}: Error = {mean_error:.4f} ± {std_error:.4f}")

# Find optimal K
optimal_K = min(results, key=lambda x: x[1])[0]
print(f"\nOptimal number of components: {optimal_K}")
```

### Step 3: Visualize Model Selection

```python
import matplotlib.pyplot as plt

Ks, means, stds = zip(*results)

fig, ax = plt.subplots(figsize=(8, 5))
ax.errorbar(Ks, means, yerr=stds, marker='o', capsize=5)
ax.set_xlabel("Number of Components (K)")
ax.set_ylabel("Normalized L² Error")
ax.set_title("Cross-Validation: Model Selection")
ax.grid(True, alpha=0.3)
ax.axvline(optimal_K, color='r', linestyle='--', label=f'Optimal K={optimal_K}')
ax.legend()
fig.savefig("model_selection.png", dpi=150, bbox_inches="tight")
```

### Step 4: Train Final Model

```python
final_model = Surrogate(
    "regression",
    n_components=optimal_K,
    width_mode="global",
    random_state=42
)
final_model.fit(dataset)
print(f"Final emulator trained with K={optimal_K}")
```

**Tips**:
- Start with K=2-3 for simple spectra
- Increase K if residuals show systematic structure
- Diminishing returns typically occur beyond K=5-7
- Use `width_mode="per_component"` if resonances have very different widths

---

## Tutorial 4: Batch Predictions and Parameter Scans

**Goal**: Efficiently evaluate the emulator over a parameter grid.

### Step 1: Define a Parameter Scan

```python
import numpy as np

# Create a 2D parameter grid
alpha_values = np.linspace(0.1, 1.0, 20)
beta_values = np.linspace(0.2, 0.8, 15)
alpha_grid, beta_grid = np.meshgrid(alpha_values, beta_values)

# Flatten for batch processing
param_points = np.column_stack([alpha_grid.ravel(), beta_grid.ravel()])
print(f"Total parameter points: {len(param_points)}")
```

### Step 2: Batch Prediction

```python
# Assuming model is already trained
energy = np.linspace(-5, 5, 300)
spectra = []

for params in param_points:
    result = model.predict(params, energy)
    spectra.append(result.spectrum)

spectra = np.array(spectra)
print(f"Predicted spectra shape: {spectra.shape}")
```

### Step 3: Analyze Results

```python
# Find maximum strength location for each parameter point
peak_energies = energy[np.argmax(spectra, axis=1)]
peak_energies = peak_energies.reshape(alpha_grid.shape)

# Compute integrated strength
integrated = np.trapz(spectra, energy, axis=1)
integrated = integrated.reshape(alpha_grid.shape)
```

### Step 4: Visualize Parameter Space

```python
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot peak energy map
c1 = ax1.contourf(alpha_grid, beta_grid, peak_energies, levels=15, cmap='viridis')
ax1.set_xlabel('α')
ax1.set_ylabel('β')
ax1.set_title('Peak Energy Position')
plt.colorbar(c1, ax=ax1, label='Energy (MeV)')

# Plot integrated strength
c2 = ax2.contourf(alpha_grid, beta_grid, integrated, levels=15, cmap='plasma')
ax2.set_xlabel('α')
ax2.set_ylabel('β')
ax2.set_title('Integrated Strength')
plt.colorbar(c2, ax=ax2, label='Strength')

plt.tight_layout()
fig.savefig("parameter_scan.png", dpi=150, bbox_inches="tight")
```

### Step 5: Extract Mixture Properties

```python
# Analyze how resonance positions evolve with parameters
mixture_params = []

for params in param_points:
    result = model.predict(params, energy)
    mixture_params.append({
        'params': params,
        'energies': result.poles,
        'strengths': result.strengths,
        'widths': result.widths
    })

# Example: Track first resonance energy vs. alpha
first_resonance = [mp['energies'][0] for mp in mixture_params]
first_resonance = np.array(first_resonance).reshape(alpha_grid.shape)

plt.figure(figsize=(8, 6))
plt.contourf(alpha_grid, beta_grid, first_resonance, levels=20, cmap='coolwarm')
plt.colorbar(label='First Resonance Energy (MeV)')
plt.xlabel('α')
plt.ylabel('β')
plt.title('Evolution of First Resonance')
plt.savefig("resonance_tracking.png", dpi=150, bbox_inches="tight")
```

**Performance Note**: 1000 predictions typically take < 1 second on a laptop.

---

## Tutorial 5: Custom Domains and Advanced Fitting

**Goal**: Adapt SMLR to a custom physics problem.

### Scenario: Atomic Photoabsorption Cross-Section

Suppose you have calculated photoabsorption cross-sections $\sigma(E)$ for different atomic species or ionization states, parameterized by charge $Z$ and screening parameter $\lambda$.

### Step 1: Define Your Physical Problem

```python
import numpy as np

# Example: Hydrogen-like photoionization
def hydrogen_cross_section(energy, Z, lambda_screen):
    """Simplified model with resonance structure."""
    # Ionization threshold
    E_thresh = 13.6 * Z**2 / lambda_screen**2
    
    # Only above threshold
    sigma = np.zeros_like(energy)
    mask = energy > E_thresh
    
    # Add resonances (Rydberg series)
    for n in range(2, 5):
        E_n = E_thresh * (1 - 1/n**2)
        Gamma_n = 0.1 * n  # Width increases with n
        A_n = 10 / n**3    # Strength decreases
        sigma[mask] += A_n * Gamma_n / ((energy[mask] - E_n)**2 + Gamma_n**2)
    
    return sigma
```

### Step 2: Generate Training Data

```python
from smlr.data import StrengthSample, StrengthDataset

energy = np.linspace(10, 50, 400)  # eV
Z_values = [1, 2, 3]  # H, He, Li
lambda_values = [0.8, 1.0, 1.2]

samples = []
for Z in Z_values:
    for lam in lambda_values:
        sigma = hydrogen_cross_section(energy, Z, lam)
        params = np.array([Z, lam])
        label = f"Z{Z}_lam{lam}"
        samples.append(StrengthSample(params, energy, sigma, label))

dataset = StrengthDataset(samples)
```

### Step 3: Custom Fitting with Per-Component Widths

```python
from smlr import Surrogate

# Use per-component widths for Rydberg series
model = Surrogate(
    "regression",
    n_components=3,  # Match number of resonances
    width_mode="per_component",  # Each has different width
    random_state=42
)

# Fit with custom tolerance
model.fit(dataset, normalize_strengths=True)
```

### Step 4: Physics-Aware Validation

```python
# Test on interpolated point
test_Z = 2.5
test_lambda = 0.9
test_params = np.array([test_Z, test_lambda])

result = model.predict(test_params, energy)

# Check sum rule (total oscillator strength)
f_total = np.trapz(result.spectrum, energy)
print(f"Total oscillator strength: {f_total:.3f}")

# Compare resonance positions to theoretical expectation
E_thresh = 13.6 * test_Z**2 / test_lambda**2
print(f"Predicted threshold: {result.poles[0]:.2f} eV")
print(f"Theoretical threshold: {E_thresh:.2f} eV")
```

### Step 5: Uncertainty Quantification

```python
# Ensemble of emulators with different random seeds
n_ensemble = 10
predictions = []

for seed in range(n_ensemble):
    model_i = Surrogate("regression", n_components=3, width_mode="per_component", random_state=seed)
    model_i.fit(dataset)
    result_i = model_i.predict(test_params, energy)
    predictions.append(result_i.spectrum)

predictions = np.array(predictions)
mean_pred = predictions.mean(axis=0)
std_pred = predictions.std(axis=0)

# Plot with uncertainty band
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.plot(energy, mean_pred, label='Mean Prediction')
plt.fill_between(energy, mean_pred - 2*std_pred, mean_pred + 2*std_pred, 
                 alpha=0.3, label='95% Confidence')
plt.xlabel('Energy (eV)')
plt.ylabel('Cross Section (arb. units)')
plt.title(f'Photoabsorption: Z={test_Z}, λ={test_lambda}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("uncertainty_quantification.png", dpi=150, bbox_inches="tight")
```

**Advanced Topics**:
- Use `fit_kwargs` to pass custom tolerances to the Lorentzian fitter
- Pre-fit mixtures offline and pass via `mixtures=` for large datasets
- Implement custom metrics for domain-specific validation
- Export trained emulators using `pickle` for reuse

---

## Next Steps

- **Optimize Performance**: See [Development Guide](development.md) for parallelization
- **Deploy Models**: Learn about model serialization and serving
- **Contribute**: Share your domain-specific examples in [CONTRIBUTING.md](contributing.md)

**Questions?** Open an issue on [GitHub](https://github.com/ascsn/SMLR/issues) or check the [FAQ](faq.md).
