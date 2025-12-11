# Examples Gallery

This section showcases SMLR's capabilities through **realistic applications** and **pedagogical examples**.

## 📊 Realistic Applications

These examples use physics-based models or real experimental data, demonstrating 
production-ready SMLR workflows.

### [Materials Spectroscopy (XANES)](../notebooks/materials_spectroscopy.ipynb) 🌟
X-ray absorption near-edge structure emulation for transition metal oxides.
**Realistic**: Based on actual XANES physics with orbital effects and crystal field splitting.

### [Structural Vibration Analysis](../notebooks/structural_vibration.ipynb) 🌟
Frequency response function (FRF) emulation for mechanical design.
**Realistic**: Uses Euler-Bernoulli beam theory with physical parameter dependencies.

### [Backend Comparison](../notebooks/backend_comparison.ipynb)
Quantitative comparison of PMM vs regression emulators with metrics.
**Realistic**: Tests on multiple performance criteria with statistical analysis.

## 🎓 Pedagogical Examples

Simplified tutorials focusing on core concepts with synthetic data.

### [Acoustic Resonance (Toy)](../notebooks/acoustic_resonance.ipynb)
Introduction to basic workflow with simple Lorentzian peaks.
**Purpose**: Learning SMLR API and understanding pole-based emulation.

### [High-Dimensional Emulation](../notebooks/high_dim_emulation.ipynb)
Demonstrates scaling to 4+ parameters with polynomial regression.
**Purpose**: Understanding parameter space complexity and regression methods.

## 📚 Reference Examples

These markdown-based examples show pre-computed outputs with detailed physics context.

### [Nuclear Response](nuclear_response.md)
High-dimensional emulation of nuclear QRPA calculations (5-15 parameters).
**Application**: Giant resonances in atomic nuclei with sum rule preservation.

### [β-Decay Comparison](beta_decay_comparison.md)
Comprehensive backend comparison using nuclear β-decay half-life calculations.
**Application**: Rare isotope physics with extrapolation validation.

---

## Quick Reference Table

| Example | Type | Domain | Parameters | Realistic? |
|---------|------|--------|------------|------------|
| [Materials Spectroscopy](../notebooks/materials_spectroscopy.ipynb) | 📓 Notebook | Physics | 2 | ✅ XANES |
| [Structural Vibration](../notebooks/structural_vibration.ipynb) | 📓 Notebook | Engineering | 3 | ✅ FEA replacement |
| [Backend Comparison](../notebooks/backend_comparison.ipynb) | 📓 Notebook | Meta-analysis | 2 | ✅ Benchmarking |
| [Acoustic Resonance](../notebooks/acoustic_resonance.ipynb) | 📓 Notebook | Tutorial | 2 | 🎓 Pedagogical |
| [High-Dimensional](../notebooks/high_dim_emulation.ipynb) | 📓 Notebook | Tutorial | 4 | 🎓 Pedagogical |
| [Nuclear Response](nuclear_response.md) | 📄 Reference | Nuclear | 5-15 | ✅ QRPA |
| [β-Decay](beta_decay_comparison.md) | 📄 Reference | Nuclear | 2 | ✅ Experimental data |

---

**Running locally:**

```bash
# Execute notebooks interactively
jupyter notebook docs/notebooks/

# Or render with MkDocs
mkdocs serve
```
