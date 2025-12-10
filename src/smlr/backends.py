"""Factory functions for creating emulators with different backends.

This module provides a unified interface for selecting between different
emulation approaches:

1. **Regression-based** (`StrengthEmulator`): Fits Lorentzian mixtures to each
   training spectrum, then learns a mapping from parameters to pole properties.
   Fast training, works well for smooth parameter dependence.

2. **Parametric Matrix Model** (`ParametricMatrixModel`): Learns a parameterized
   response matrix M(p) = D + Σᵢ (pᵢ - p₀ᵢ)·Sᵢ whose eigenstructure reproduces
   training spectra. Better physics preservation and extrapolation.

Usage
-----
>>> from smlr.backends import get_emulator
>>> 
>>> # Regression-based (default, fast)
>>> emu = get_emulator("regression", n_components=5)
>>> emu.fit(dataset)
>>> 
>>> # Parametric Matrix Model (physics-based)
>>> emu = get_emulator("pmm", n_poles=10)
>>> emu.fit(dataset, reference_point=center)
"""
from __future__ import annotations

from typing import Any, Literal, Optional, Union

from .emulator import StrengthEmulator
from .pmm import ParametricMatrixModel


EmulatorType = Union[StrengthEmulator, ParametricMatrixModel]
BackendName = Literal["regression", "lorentzian", "pmm", "matrix"]


def get_emulator(
    backend: BackendName = "regression",
    n_components: Optional[int] = None,
    n_poles: Optional[int] = None,
    **kwargs,
) -> EmulatorType:
    """Create an emulator with the specified backend.
    
    Parameters
    ----------
    backend : {"regression", "lorentzian", "pmm", "matrix"}
        Emulation method to use:
        - "regression" or "lorentzian": Regression-based emulator that fits
          Lorentzian mixtures and learns parameter-to-pole mappings.
        - "pmm" or "matrix": Parametric Matrix Model that learns a reduced-order
          response matrix with parameter-dependent eigenstructure.
    n_components : int, optional
        Number of Lorentzian components (for regression backend).
    n_poles : int, optional
        Matrix dimension / number of poles (for PMM backend).
        If not provided, uses n_components.
    **kwargs
        Additional arguments passed to the emulator constructor.
        
    Returns
    -------
    StrengthEmulator or ParametricMatrixModel
        Configured emulator ready for fitting.
        
    Examples
    --------
    >>> from smlr.backends import get_emulator
    >>> 
    >>> # Fast regression-based emulation
    >>> emu = get_emulator("regression", n_components=4, regression_method="ridge")
    >>> emu.fit(dataset)
    >>> 
    >>> # Physics-based PMM emulation
    >>> emu = get_emulator("pmm", n_poles=10, max_iterations=5000)
    >>> emu.fit(dataset, reference_point=np.array([0.5, 0.5]))
    
    Notes
    -----
    **When to use each backend:**
    
    Regression-based:
    - Large datasets (100+ samples)
    - Smooth parameter dependence
    - Fast training/prediction needed
    - Uncertainty quantification (with GP option)
    
    Parametric Matrix Model:
    - Physics-based interpolation important
    - Sum rule preservation needed
    - Better extrapolation behavior
    - Fewer training samples available
    - Response matrix structure is known/assumed
    """
    backend = backend.lower()
    
    if backend in ("regression", "lorentzian"):
        if n_components is None:
            n_components = n_poles or 4
        return StrengthEmulator(n_components=n_components, **kwargs)
    
    elif backend in ("pmm", "matrix"):
        if n_poles is None:
            n_poles = n_components or 10
        return ParametricMatrixModel(n_poles=n_poles, **kwargs)
    
    else:
        raise ValueError(
            f"Unknown backend '{backend}'. "
            f"Choose from: 'regression', 'lorentzian', 'pmm', 'matrix'"
        )


def list_backends() -> dict:
    """List available emulation backends with descriptions.
    
    Returns
    -------
    dict
        Backend names mapped to descriptions.
    """
    return {
        "regression": (
            "Regression-based emulation. Fits Lorentzian mixtures to training "
            "spectra and learns parameter-to-pole mappings via linear/polynomial/GP "
            "regression. Fast training, good for smooth parameter dependence."
        ),
        "lorentzian": "Alias for 'regression'.",
        "pmm": (
            "Parametric Matrix Model. Learns a reduced-order response matrix "
            "M(p) = D + Σᵢ (pᵢ-p₀ᵢ)·Sᵢ whose eigenstructure reproduces training "
            "spectra. Better physics preservation and extrapolation."
        ),
        "matrix": "Alias for 'pmm'.",
    }
