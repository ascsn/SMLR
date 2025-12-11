# Materials Spectroscopy Emulation

This example applies SMLR to **optical and X-ray absorption spectroscopy** 
in materials science, demonstrating emulation of electronic excitation spectra.

## Applications

- **X-ray absorption spectroscopy (XAS)**: XANES, EXAFS edge analysis
- **Optical spectroscopy**: UV-Vis absorption, photoluminescence
- **Semiconductor physics**: Band gap engineering, quantum dots
- **Catalysis research**: Active site characterization

## The Physics

Electronic excitation spectra in solids show Lorentzian-like peaks due to:

- **Core-level excitations**: Discrete atomic transitions (XANES)
- **Excitonic peaks**: Bound electron-hole pairs below band gap
- **Interband transitions**: Band-to-band absorption with lifetime broadening

The absorption coefficient:

$$\alpha(E) = \sum_n \frac{f_n \Gamma_n / 2\pi}{(E - E_n)^2 + \Gamma_n^2/4}$$

where $f_n$ are oscillator strengths and $E_n$ are transition energies.

## Example Setup

```python
from smlr.backends import get_emulator
from smlr.observables import SumRule

# 5 parameters: band gap, spin-orbit, strain, temperature, doping
dataset = build_materials_dataset(n_samples=80)

# Create emulator with 5 components
emu = get_emulator("regression", n_components=5, regression_method="polynomial")
emu.fit(dataset)
```

## Physical Parameters

| Parameter | Range | Physical Meaning |
|-----------|-------|------------------|
| Band gap | 0.5 - 3.0 eV | Fundamental gap |
| Spin-orbit | 0.1 - 0.8 eV | SO coupling strength |
| Strain | -0.02 - 0.02 | Biaxial strain |
| Temperature | 100 - 500 K | Thermal broadening |
| Doping | 0 - 0.1 | Carrier concentration |

## Results

### Method Comparison

We compare different regression methods for XANES emulation:

![XANES Methods Comparison](figs/xanes_methods.png)

*Comparison of linear, ridge, and polynomial regression methods. Polynomial regression provides the best fit for the nonlinear parameter dependencies in XANES spectra.*

### Spectrum Reconstruction

The best-performing emulator accurately reproduces XANES features:

![XANES Comparison](figs/xanes_comparison.png)

*Emulator predictions (dashed) vs reference spectra (solid) for test samples. The absorption edge position, pre-edge features, and EXAFS oscillations are all well-captured.*

### Error Analysis

Quantitative error assessment across different methods:

![XANES Error Comparison](figs/xanes_error_comparison.png)

*Normalized L² error distribution for different regression methods. Polynomial regression achieves ~1.2% error.*

### Sum Rule Verification

The Thomas-Reiche-Kuhn sum rule is preserved:

$$m_0 = \int \alpha(E) dE \propto N_{\text{electrons}}$$

### Accuracy Metrics

The example script (`examples/materials_spectroscopy.py`) compares multiple regression methods
and reports performance based on the specific configuration used.
With default settings (50 training samples, 10 test samples for XANES with 6-dimensional parameter space):

| Method | Mean L² Error |
|--------|---------------|
| Linear | 1.23% |
| Ridge | 1.23% |
| Polynomial | 1.21% |

Best method: polynomial (1.21%)

All three methods achieve excellent accuracy (~1-2%) for this problem.

## Code

??? example "Spectrum Generator"
    ```python
    import numpy as np
    from smlr.lorentz import lorentzian_sum
    
    def generate_materials_spectrum(params, energy):
        """Generate synthetic XAS/optical absorption spectrum."""
        band_gap, spin_orbit, strain, temperature, doping = params
        
        # Strain affects band gap linearly
        effective_gap = band_gap * (1 - 5 * strain)
        
        # Temperature broadening
        thermal_width = 0.026 * temperature / 300  # ~kT at 300K
        
        # Define transitions
        n_transitions = 5
        positions = []
        strengths = []
        
        # Excitonic peak (below gap)
        exciton_binding = 0.1 * band_gap
        positions.append(effective_gap - exciton_binding)
        strengths.append(0.3 * (1 - doping * 5))  # Bleaches with doping
        
        # Band edge
        positions.append(effective_gap)
        strengths.append(1.0)
        
        # Spin-orbit split peak
        positions.append(effective_gap + spin_orbit)
        strengths.append(0.5)
        
        # Higher transitions
        positions.append(effective_gap + 2 * spin_orbit)
        strengths.append(0.2)
        positions.append(effective_gap + 3 * spin_orbit + 0.5)
        strengths.append(0.1)
        
        positions = np.array(positions)
        strengths = np.array(strengths)
        
        # Width increases with energy (lifetime broadening)
        base_width = 0.1 + thermal_width
        widths = base_width * (1 + 0.2 * np.arange(n_transitions))
        
        return lorentzian_sum(energy, positions, strengths, widths)
    ```

??? example "Full Training Script"
    ```python
    """Materials spectroscopy emulation."""
    import numpy as np
    from smlr.backends import get_emulator
    from smlr.data import StrengthDataset, StrengthSample
    from smlr.observables import SumRule
    
    def build_dataset(n_samples=80, seed=42):
        rng = np.random.default_rng(seed)
        energy = np.linspace(0.5, 5.0, 250)
        samples = []
        for i in range(n_samples):
            params = np.array([
                rng.uniform(1.0, 2.5),      # band_gap
                rng.uniform(0.2, 0.6),      # spin_orbit
                rng.uniform(-0.02, 0.02),   # strain
                rng.uniform(150, 400),      # temperature
                rng.uniform(0, 0.05),       # doping
            ])
            spectrum = generate_materials_spectrum(params, energy)
            samples.append(StrengthSample(params=params, energy=energy,
                                          strength=spectrum, label=f"xas_{i}"))
        return StrengthDataset(samples)
    
    # Train emulator
    dataset = build_dataset(n_samples=80)
    emu = get_emulator("regression", n_components=5, regression_method="polynomial", 
                       poly_degree=2)
    emu.fit(dataset)
    
    # Check sum rule
    m0 = SumRule(k=0, name="m0")
    energy = np.linspace(0.5, 5.0, 250)
    
    test_params = np.array([1.5, 0.4, 0.0, 300, 0.02])
    true_spectrum = generate_materials_spectrum(test_params, energy)
    pred = emu.predict_mixture(test_params)
    pred_spectrum = lorentzian_sum(energy, pred.energies, pred.strengths, pred.widths)
    
    print(f"True m0: {m0.compute(energy, true_spectrum):.3f}")
    print(f"Pred m0: {m0.compute(energy, pred_spectrum):.3f}")
    ```

## Key Insights

1. Polynomial features help: Nonlinear parameter dependence (strain, temperature) benefits from polynomial regression.

2. Physical constraints: Sum rules provide important validation checks.

3. Temperature effects: Thermal broadening is naturally captured by Lorentzian widths.

4. 5D tractable: 80 samples suffice for 5-parameter materials physics.

---

[:octicons-arrow-left-24: Back to Examples](index.md){ .md-button }
