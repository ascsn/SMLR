"""SMLR: surrogate models for linear response.

This package offers utilities to fit Lorentzian mixtures to strength functions and learn
emulators that map theory parameters to predicted spectra.

Key Features:
- General-purpose emulation for arbitrary parameter dimensions (2, 5, 10, 15+)
- Multiple emulation backends: regression-based or Parametric Matrix Models (PMM)
- Multiple optimization backends (scipy, TensorFlow, JAX)
- Physics-informed training with observable constraints
- Flexible regression methods (linear, ridge, polynomial, Gaussian process)
"""

from . import data, lorentz, emulator, metrics, plotting, observables, optimization, pmm, backends

__all__ = [
    "data",
    "lorentz", 
    "emulator",
    "pmm",
    "backends",
    "metrics",
    "plotting",
    "observables",
    "optimization",
]
__version__ = "0.2.0"
