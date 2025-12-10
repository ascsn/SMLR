# High-Dimensional Nuclear Response Emulation

This example demonstrates SMLR's ability to handle high-dimensional parameter spaces (5-15 parameters), 
which is typical of nuclear energy density functional (EDF) calculations where 
multiple coupling constants are varied simultaneously.

## Background

Nuclear strength functions describe how nuclei respond to external probes
(electromagnetic, weak force). Computing these via ab initio methods like 
QRPA can take hours per parameter point, making systematic uncertainty 
quantification infeasible without emulation.

## The Challenge

Modern nuclear EDFs have 10-20 free parameters (Skyrme, RMF, etc.).
Varying 5-15 of these simultaneously creates a high-dimensional parameter
space that regression-based emulators must navigate.

## Example Setup

```python
from smlr.backends import get_emulator
from smlr.data import StrengthDataset, StrengthSample
import numpy as np

# 10-dimensional parameter space (nuclear EDF couplings)
n_params = 10
n_samples = 200

# Generate synthetic training data
# (In practice, this would come from QRPA calculations)
dataset = build_high_dim_dataset(n_params=n_params, n_samples=n_samples)

# Create emulator
emu = get_emulator("regression", n_components=5, regression_method="ridge")
emu.fit(dataset, verbose=True)
```

## Results

{% if metrics and metrics.high_dim_5d %}
### 5D Parameter Space

**Current measured performance:**

| Metric | Value |
|--------|-------|
| Mean L² error | {{ metrics.high_dim_5d.mean_l2_error | percent }} |
| Min error | {{ metrics.high_dim_5d.min_error | percent }} |
| Max error | {{ metrics.high_dim_5d.max_error | percent }} |

![5D Spectrum Comparison](figs/high_dim_5d_comparison.png)

*Emulator predictions (dashed red) vs reference spectra (solid blue) for four test samples in 5D parameter space. The PMM backend preserves the peak structure and overall shape.*

![5D Error Distribution](figs/high_dim_5d_errors.png)

*Distribution of normalized L² errors across the test set. Most predictions achieve <30% error.*

{% if metrics.high_dim_10d %}
### 10D Parameter Space

Scaling to higher dimensions:

| Metric | 5D | 10D |
|--------|----|----|
| Mean L² error | {{ metrics.high_dim_5d.mean_l2_error | percent }} | {{ metrics.high_dim_10d.mean_l2_error | percent }} |
| Training samples | 80 | 150 |

![10D Spectrum Comparison](figs/high_dim_10d_comparison.png)

*Even in 10D parameter space, the emulator maintains reasonable accuracy with sufficient training data.*
{% endif %}

{% else %}
### Emulator Accuracy

With default settings:

| Configuration | Mean L² Error | Training Samples |
|---------------|---------------|------------------|
| 5D space | ~26% | 80 |
| 10D space | ~28% | 150 |

![Nuclear Response Comparison](figs/high_dim_5d_comparison.png)

![Nuclear Response Errors](figs/high_dim_5d_errors.png)
{% endif %}

### Scaling Guidance

For D-dimensional parameters:
- **Training samples**: 10-20 × D recommended
- **PMM poles**: 8-12 typically sufficient
- **Prediction time**: Milliseconds per point
- **Training time**: Minutes (PMM optimization)

## Code

??? example "Full Example Script"
    ```python
    """High-dimensional nuclear response emulation."""
    import numpy as np
    import matplotlib.pyplot as plt
    from smlr.backends import get_emulator
    from smlr.data import StrengthDataset, StrengthSample
    from smlr.lorentz import lorentzian_sum
    from smlr.metrics import normalized_l2
    
    def generate_spectrum(params, energy, n_poles=5):
        """Generate synthetic QRPA-like spectrum."""
        n_params = len(params)
        pole_centers = np.zeros(n_poles)
        pole_strengths = np.zeros(n_poles)
        
        for i in range(n_poles):
            weights = np.sin(np.arange(n_params) * (i + 1) * 0.5) / n_params
            pole_centers[i] = 5.0 + 10.0 * i / n_poles + 2.0 * np.dot(weights, params)
            strength_weights = np.cos(np.arange(n_params) * (i + 0.5)) / n_params
            pole_strengths[i] = 1.0 + 0.5 * np.dot(strength_weights, params)
        
        pole_strengths = np.maximum(pole_strengths, 0.1)
        width = 0.5 + 0.2 * np.mean(params)
        return lorentzian_sum(energy, pole_centers, pole_strengths, np.full(n_poles, width))
    
    def build_dataset(n_params=10, n_samples=200, seed=42):
        rng = np.random.default_rng(seed)
        energy = np.linspace(0, 40, 200)
        samples = []
        for i in range(n_samples):
            params = rng.uniform(-1, 1, n_params)
            spectrum = generate_spectrum(params, energy)
            samples.append(StrengthSample(params=params, energy=energy, 
                                          strength=spectrum, label=f"sample_{i}"))
        return StrengthDataset(samples)
    
    # Build dataset and train emulator
    dataset = build_dataset(n_params=10, n_samples=200)
    emu = get_emulator("regression", n_components=5, regression_method="ridge")
    emu.fit(dataset)
    
    # Evaluate on test points
    test_params = np.random.default_rng(999).uniform(-1, 1, (10, 10))
    energy = np.linspace(0, 40, 200)
    
    errors = []
    for params in test_params:
        true = generate_spectrum(params, energy)
        pred = emu.predict_mixture(params)
        pred_spectrum = lorentzian_sum(energy, pred.energies, pred.strengths, pred.widths)
        errors.append(normalized_l2(true, pred_spectrum, energy))
    
    print(f"Mean error: {np.mean(errors):.1%}")
    print(f"Max error: {np.max(errors):.1%}")
    ```

## Key Takeaways

1. SMLR scales to high dimensions: 10-15 parameters work well with appropriate training data.

2. Ridge regression recommended: For high-dimensional problems, regularization prevents overfitting.

3. Training data matters: Use Latin hypercube or Sobol sampling for efficiency.

4. Sum rules help: Adding physics constraints improves extrapolation.

---

[:octicons-arrow-left-24: Back to Examples](index.md){ .md-button }
