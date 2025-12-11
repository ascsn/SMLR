# Examples Gallery

This section showcases SMLR's capabilities through worked examples.

## Executable Notebooks

These Jupyter notebooks are executed at documentation build time, showing real outputs
from the current codebase. Download them to run interactively in your own environment.

### [Acoustic Resonance](../notebooks/acoustic_resonance.ipynb)
Room acoustics and instrument modeling using Lorentzian decomposition.
Shows how SMLR handles harmonic series and Q-factor variation.

### [Materials Spectroscopy](../notebooks/materials_spectroscopy.ipynb)
X-ray absorption and optical spectroscopy emulation.
Covers band gaps, excitonic peaks, and temperature-dependent broadening.

### [High-Dimensional Emulation](../notebooks/high_dim_emulation.ipynb)
Demonstrates handling 4+ input parameters with polynomial regression.
Shows how SMLR scales to complex multi-parameter problems.

## Reference Examples

These markdown-based examples show pre-computed outputs with detailed explanations.

### [Nuclear Response](nuclear_response.md)
High-dimensional emulation of nuclear QRPA calculations with 5-15 parameters.
Demonstrates sum rule preservation and scaling to complex nuclear models.

### [Backend Comparison](backend_comparison.md)
Side-by-side comparison of PMM vs regression-based emulation.
Helps you choose the right approach for your problem.

### [β-Decay Comparison](beta_decay_comparison.md)
Comprehensive comparison using realistic nuclear β-decay data.
Demonstrates interpolation, extrapolation, and sum rule analysis.

## Quick Reference Table

| Example | Type | Parameters | Key Features |
|---------|------|------------|--------------|
| [Acoustic Resonance](../notebooks/acoustic_resonance.ipynb) | 📓 Notebook | 2 | Harmonic series, Q-factors |
| [Materials Spectroscopy](../notebooks/materials_spectroscopy.ipynb) | 📓 Notebook | 2 | Band gaps, excitonic peaks |
| [High-Dimensional](../notebooks/high_dim_emulation.ipynb) | 📓 Notebook | 4 | Polynomial regression |
| [Nuclear Response](nuclear_response.md) | 📄 Reference | 5-15 | Sum rules, QRPA |
| [Backend Comparison](backend_comparison.md) | 📄 Reference | 2 | PMM vs Regression |
| [β-Decay](beta_decay_comparison.md) | 📄 Reference | 2 | Extrapolation analysis |

---

**Running locally:**

```bash
# Run notebooks interactively
jupyter notebook docs/notebooks/

# Python scripts
cd examples/
python high_dim_emulation.py --help
python acoustic_resonance.py --n-samples 100
```
