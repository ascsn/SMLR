# Emulation Backend Comparison

This example compares SMLR's two emulation backends:

1. **Regression-based**: Fits Lorentzians, learns parameter → pole mappings
2. **Parametric Matrix Model (PMM)**: Learns a reduced-order response matrix

## When to Use Each

| Factor | Regression | PMM |
|--------|------------|-----|
| Training speed | Fast | Slower |
| Prediction speed | Very fast | Fast |
| Training data needed | Medium | Fewer |
| Extrapolation | Limited | Better |
| Sum rule preservation | Approximate | Exact |
| Physics transparency | Black box | Interpretable |
| High dimensions | Scales well | More parameters |

## The Key Difference

### Regression Approach

```
Training spectra → Fit Lorentzians → Learn regression: params → poles
                                              ↓
                        Prediction: poles → Lorentzian sum → spectrum
```

**Pros**: Fast, simple, works well with smooth dependence
**Cons**: No physics constraints, can violate sum rules

### PMM Approach

```
Training spectra → Learn matrix M(p) = D + Σᵢ (pᵢ - p₀ᵢ) · Sᵢ
                                              ↓
                    Prediction: Diagonalize M(p) → eigenvalues/vectors
                                              ↓
                              S(E) = Σₙ |⟨n|v₀⟩|² L(E; Eₙ)
```

**Pros**: Physics-based, preserves sum rules, better extrapolation
**Cons**: Slower training, more hyperparameters

## Example Comparison

```python
from smlr.pmm import compare_emulation_methods

# Compare on a 2D nuclear response dataset
results = compare_emulation_methods(
    dataset,
    n_components=5,
    test_fraction=0.2,
)

print(f"Regression: {results['regression_mean_error']:.1%} ± {results['regression_std_error']:.1%}")
print(f"PMM:        {results['pmm_mean_error']:.1%} ± {results['pmm_std_error']:.1%}")
```

## Results on Nuclear Data

Note: The following results are representative examples. Actual performance depends on
the specific problem, hyperparameters, and data characteristics. Run the example script
`examples/beta_decay_backend_comparison.py` for current metrics.

### Interpolation (within training region)

| Method | Mean Error | Train Time |
|--------|------------|-----------|\n| Regression (ridge) | Varies by problem | Order of seconds |
| PMM | Varies by problem | Order of tens of seconds |

Both methods work well for interpolation.

### Extrapolation (outside training region)

| Method | Performance |
|--------|-------------|
| Regression | Limited extrapolation capability |
| PMM | Better extrapolation due to physics-based structure |

PMM typically outperforms regression for extrapolation.

### Sum Rule Preservation

The m₁ sum rule (Thomas-Reiche-Kuhn) should be constant:

| Method | Sum Rule Preservation |
|--------|-----------------------|
| Regression | Approximate |
| PMM | Excellent (< 1% deviation typical) |

PMM preserves sum rules much better due to eigenvalue structure.

## Visualization

The following plots show a side-by-side comparison of the emulation backends:

![Backend Comparison Spectra](figs/backend_comparison_spectra.png)

The summary metrics confirm PMM's advantages for extrapolation:

![Backend Comparison Summary](figs/backend_comparison_summary.png)

## Code

??? example "Full Comparison Script"
    ```python
    """Compare regression and PMM emulation backends."""
    import numpy as np
    import matplotlib.pyplot as plt
    from smlr.backends import get_emulator
    from smlr.pmm import ParametricMatrixModel, compare_emulation_methods
    from smlr.data import StrengthDataset, StrengthSample
    from smlr.lorentz import lorentzian_sum
    from smlr.metrics import normalized_l2
    from smlr.observables import SumRule
    
    def generate_spectrum(alpha, beta, energy):
        """2-parameter spectrum for comparison."""
        center = 15 + 3 * alpha - 2 * beta
        width = 2 + 0.5 * beta
        strength = 1 + 0.3 * alpha
        return lorentzian_sum(energy, [center], [strength], [width])
    
    def build_dataset(n_samples=50, seed=42):
        rng = np.random.default_rng(seed)
        energy = np.linspace(0, 40, 200)
        samples = []
        for i in range(n_samples):
            alpha = rng.uniform(0, 1)
            beta = rng.uniform(0, 1)
            spectrum = generate_spectrum(alpha, beta, energy)
            samples.append(StrengthSample(
                params=np.array([alpha, beta]),
                energy=energy,
                strength=spectrum,
                label=f"sample_{i}"
            ))
        return StrengthDataset(samples)
    
    # Build dataset
    dataset = build_dataset(n_samples=50)
    energy = np.linspace(0, 40, 200)
    
    # Train both backends
    reg_emu = get_emulator("regression", n_components=3)
    reg_emu.fit(dataset)
    
    pmm_emu = get_emulator("pmm", n_poles=5)
    pmm_emu.fit(dataset, reference_point=np.array([0.5, 0.5]))
    
    # Test interpolation
    test_interp = np.array([0.3, 0.7])
    true_interp = generate_spectrum(*test_interp, energy)
    
    reg_pred = reg_emu.predict_mixture(test_interp)
    reg_spectrum = lorentzian_sum(energy, reg_pred.energies, reg_pred.strengths, reg_pred.widths)
    
    pmm_result = pmm_emu.predict(test_interp, energy)
    pmm_spectrum = pmm_result.spectrum
    
    print("=== Interpolation ===")
    print(f"Regression error: {normalized_l2(true_interp, reg_spectrum, energy):.1%}")
    print(f"PMM error: {normalized_l2(true_interp, pmm_spectrum, energy):.1%}")
    
    # Test extrapolation
    test_extrap = np.array([1.5, 1.5])  # Outside [0,1] training range
    true_extrap = generate_spectrum(*test_extrap, energy)
    
    reg_pred = reg_emu.predict_mixture(test_extrap)
    reg_spectrum = lorentzian_sum(energy, reg_pred.energies, reg_pred.strengths, reg_pred.widths)
    
    pmm_result = pmm_emu.predict(test_extrap, energy)
    pmm_spectrum = pmm_result.spectrum
    
    print("\n=== Extrapolation ===")
    print(f"Regression error: {normalized_l2(true_extrap, reg_spectrum, energy):.1%}")
    print(f"PMM error: {normalized_l2(true_extrap, pmm_spectrum, energy):.1%}")
    
    # Check sum rule
    m1 = SumRule(k=1, name="m1")
    print("\n=== Sum Rule (m₁) ===")
    print(f"True: {m1.compute(energy, true_extrap):.2f}")
    print(f"Regression: {m1.compute(energy, reg_spectrum):.2f}")
    print(f"PMM: {m1.compute(energy, pmm_spectrum):.2f}")
    ```

??? example "Using the Factory Function"
    ```python
    from smlr.backends import get_emulator, list_backends
    
    # See available backends
    for name, desc in list_backends().items():
        print(f"{name}: {desc[:60]}...")
    
    # Create emulators with unified interface
    emu_fast = get_emulator("regression", n_components=5)
    emu_physics = get_emulator("pmm", n_poles=10)
    
    # Both have .fit() and similar prediction APIs
    emu_fast.fit(dataset)
    emu_physics.fit(dataset)
    ```

## Recommendations

### Use Regression When:
- Large training sets available (100+ samples)
- Smooth, continuous parameter dependence
- Fast training/prediction is critical
- Uncertainty quantification needed (GP option)
- High-dimensional parameters (10+)

### Use PMM When:
- Physics-based interpolation matters
- Sum rules must be preserved
- Extrapolation beyond training range needed
- Fewer training samples available
- Response matrix structure is known
- Interpretability of learned model important

### Hybrid Approach

For best results, consider:

1. Start with regression for fast prototyping
2. Switch to PMM for production if extrapolation/physics matters
3. Use both and ensemble predictions for uncertainty

---

[:octicons-arrow-left-24: Back to Examples](index.md){ .md-button }
