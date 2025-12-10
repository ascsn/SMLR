# Examples Gallery

This section showcases SMLR's capabilities through worked examples with pre-computed outputs.
You don't need to run anything—just browse to understand what SMLR can do!

## Physics Applications

### [Nuclear Response](nuclear_response.md)
High-dimensional emulation of nuclear QRPA calculations with 5-15 parameters.
Demonstrates sum rule preservation and scaling to complex nuclear models.

### [Acoustic Resonance](acoustic_resonance.md)
Room acoustics and instrument modeling using Lorentzian decomposition.
Shows how SMLR handles harmonic series and Q-factor variation.

### [Materials Spectroscopy](materials_spectroscopy.md)
X-ray absorption and optical spectroscopy emulation.
Covers band gaps, excitonic peaks, and temperature-dependent broadening.

## Methodology Comparisons

### [Backend Comparison](backend_comparison.md)
Side-by-side comparison of PMM vs regression-based emulation.
Helps you choose the right approach for your problem.

### [β-Decay Comparison](beta_decay_comparison.md)
Comprehensive comparison using realistic nuclear β-decay data.
Demonstrates interpolation, extrapolation, and sum rule analysis.

## Quick Reference Table

| Example | Domain | Parameters | Key Features |
|---------|--------|------------|--------------|
| [Nuclear Response](nuclear_response.md) | Nuclear physics | 5-15 | High-dimensional, sum rules |
| [Acoustic Resonance](acoustic_resonance.md) | Acoustics | 4 | Harmonic series, Q-factors |
| [Materials Spectroscopy](materials_spectroscopy.md) | Materials science | 5 | Band gaps, excitonic peaks |
| [Backend Comparison](backend_comparison.md) | Methods | 2 | PMM vs Regression |
| [β-Decay Comparison](beta_decay_comparison.md) | Nuclear | 2 | Extrapolation analysis |

---

Note: All examples are available as Python scripts in the `examples/` folder:

```bash
cd examples/
python high_dim_emulation.py --help
python acoustic_resonance.py --n-samples 100
```
