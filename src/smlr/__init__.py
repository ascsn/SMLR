"""SMLR: surrogate models for linear response.

This package offers utilities to fit Lorentzian mixtures to strength functions and learn
emulators that map theory parameters to predicted spectra.

Key Features:
- General-purpose emulation for arbitrary parameter dimensions (2, 5, 10, 15+)
- Multiple emulation backends: regression-based or Parametric Matrix Models (PMM)
- Unified Surrogate interface for consistent API across backends
- Multiple optimization backends (scipy, TensorFlow, JAX)
- Physics-informed training with observable constraints
- Flexible regression methods (linear, ridge, polynomial, Gaussian process)

Quick Start
-----------
>>> from smlr import Surrogate
>>> from smlr.data import StrengthDataset
>>>
>>> # Load or create your dataset
>>> dataset = StrengthDataset.from_folder("metadata.csv", param_columns=["alpha", "beta"])
>>>
>>> # Create and fit a surrogate model
>>> model = Surrogate(backend="pmm", n_poles=10)
>>> model.fit(dataset)
>>>
>>> # Predict at new parameter points
>>> result = model.predict(new_params, energy_grid)
>>> print(result.spectrum)  # The predicted strength function
"""

from . import data, lorentz, emulator, metrics, plotting, observables, optimization, pmm, backends, base

# Import key classes at package level for convenience
from .base import Surrogate, BaseEmulator, EmulatorResult
from .data import StrengthDataset, StrengthSample
from .emulator import StrengthEmulator
from .pmm import ParametricMatrixModel
from .backends import get_emulator, list_backends

__all__ = [
    # Primary user-facing API
    "Surrogate",
    "EmulatorResult",
    "StrengthDataset",
    "StrengthSample",
    
    # Backend classes (for advanced users)
    "StrengthEmulator",
    "ParametricMatrixModel",
    "BaseEmulator",
    
    # Factory functions
    "get_emulator",
    "list_backends",
    
    # Submodules
    "data",
    "lorentz", 
    "emulator",
    "pmm",
    "backends",
    "base",
    "metrics",
    "plotting",
    "observables",
    "optimization",
]
__version__ = "0.3.0"  # Bumped for major API update
