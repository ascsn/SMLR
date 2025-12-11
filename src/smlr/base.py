"""Base classes and protocols for SMLR emulators.

This module defines the abstract interface that all emulation backends must implement,
ensuring a consistent API across different approaches (regression, PMM, future backends).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple, Union, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .data import StrengthDataset

Array = np.ndarray


@dataclass
class EmulatorResult:
    """Unified result container for emulator predictions.
    
    Provides consistent access to prediction outputs regardless of backend.
    
    Attributes
    ----------
    spectrum : array
        Predicted strength function on the energy grid.
    energy : array
        Energy grid used for evaluation.
    poles : array
        Pole energies (peak positions).
    strengths : array  
        Pole strengths (peak heights/areas).
    widths : array
        Pole widths. Shape is (n_poles,) or (1,) for global width.
    metadata : dict
        Backend-specific additional information.
    """
    spectrum: Array
    energy: Array
    poles: Array
    strengths: Array
    widths: Array
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def n_poles(self) -> int:
        """Number of poles in the prediction."""
        return len(self.poles)
    
    def sum_rule(self, k: int = 0) -> float:
        """Compute the k-th energy-weighted moment (sum rule).
        
        m_k = ∫ E^k S(E) dE
        """
        return float(np.trapezoid(self.energy**k * self.spectrum, self.energy))


class BaseEmulator(ABC):
    """Abstract base class for all SMLR emulators.
    
    All emulation backends (regression-based, PMM, future methods) must implement
    this interface to ensure consistent usage patterns.
    
    Subclasses must implement:
    - fit(dataset, **kwargs) -> self
    - predict(params, energy, **kwargs) -> EmulatorResult
    
    Optional overrides:
    - predict_batch(params_batch, energy, **kwargs) -> List[EmulatorResult]
    - score(dataset, metric) -> float
    """
    
    # Core attributes that all backends should set after fitting
    param_dim: Optional[int] = None
    n_samples_seen: int = 0
    _is_fitted: bool = False
    
    @abstractmethod
    def fit(
        self,
        dataset: "StrengthDataset",
        **kwargs,
    ) -> "BaseEmulator":
        """Fit the emulator to training data.
        
        Parameters
        ----------
        dataset : StrengthDataset
            Training dataset with strength function samples.
        **kwargs
            Backend-specific fitting options.
            
        Returns
        -------
        self
            Fitted emulator for method chaining.
        """
        pass
    
    @abstractmethod
    def predict(
        self,
        params: Array,
        energy: Optional[Array] = None,
        **kwargs,
    ) -> EmulatorResult:
        """Predict strength function at a new parameter point.
        
        Parameters
        ----------
        params : array of shape (param_dim,)
            Parameter vector at which to predict.
        energy : array, optional
            Energy grid for spectrum evaluation. If None, uses a default grid.
        **kwargs
            Backend-specific prediction options.
            
        Returns
        -------
        EmulatorResult
            Prediction with spectrum, poles, and metadata.
        """
        pass
    
    def predict_batch(
        self,
        params_batch: Array,
        energy: Optional[Array] = None,
        **kwargs,
    ) -> List[EmulatorResult]:
        """Predict for multiple parameter points.
        
        Default implementation calls predict() in a loop. Backends may override
        for more efficient batch processing.
        
        Parameters
        ----------
        params_batch : array of shape (n_points, param_dim)
            Parameter vectors.
        energy : array, optional
            Shared energy grid for all predictions.
            
        Returns
        -------
        list of EmulatorResult
            Predictions for each parameter point.
        """
        params_batch = np.atleast_2d(params_batch)
        return [self.predict(p, energy, **kwargs) for p in params_batch]
    
    def score(
        self,
        dataset: "StrengthDataset",
        metric: Literal["l2", "mse", "mae"] = "l2",
    ) -> float:
        """Evaluate emulator accuracy on a test dataset.
        
        Parameters
        ----------
        dataset : StrengthDataset
            Test dataset.
        metric : {"l2", "mse", "mae"}
            Error metric to use.
            - "l2": Normalized L2 error (default, recommended)
            - "mse": Mean squared error
            - "mae": Mean absolute error
            
        Returns
        -------
        float
            Mean error across all samples.
        """
        from .metrics import normalized_l2
        
        self._check_fitted()
        errors = []
        
        for sample in dataset.samples:
            result = self.predict(sample.params, sample.energy)
            
            if metric == "l2":
                err = normalized_l2(result.spectrum, sample.strength, sample.energy)
            elif metric == "mse":
                err = float(np.mean((result.spectrum - sample.strength) ** 2))
            elif metric == "mae":
                err = float(np.mean(np.abs(result.spectrum - sample.strength)))
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            errors.append(err)
        
        return float(np.mean(errors))
    
    def _check_fitted(self) -> None:
        """Raise error if emulator is not fitted."""
        if not self._is_fitted:
            raise RuntimeError(
                f"{self.__class__.__name__} is not fitted. Call fit() first."
            )
    
    @property
    def is_fitted(self) -> bool:
        """Whether the emulator has been fitted."""
        return self._is_fitted
    
    def get_info(self) -> Dict[str, Any]:
        """Return information about the emulator configuration and state.
        
        Returns
        -------
        dict
            Emulator metadata including type, parameters, and training info.
        """
        return {
            "backend": self.__class__.__name__,
            "is_fitted": self._is_fitted,
            "param_dim": self.param_dim,
            "n_samples_seen": self.n_samples_seen,
        }


class Surrogate:
    """High-level surrogate model for strength function emulation.
    
    This is the recommended user-facing class. It wraps any emulation backend
    and provides a consistent, easy-to-use interface.
    
    Parameters
    ----------
    backend : str or BaseEmulator
        Emulation method to use:
        - "regression" or "lorentzian": Fast regression-based approach
        - "pmm" or "matrix": Physics-based Parametric Matrix Model
        - BaseEmulator instance: Use a pre-configured backend
    n_poles : int
        Number of poles/components to use.
    **kwargs
        Additional arguments passed to the backend constructor.
        
    Examples
    --------
    >>> from smlr import Surrogate
    >>> 
    >>> # Simple usage with automatic backend selection
    >>> model = Surrogate(backend="pmm", n_poles=10)
    >>> model.fit(dataset)
    >>> result = model.predict(new_params, energy_grid)
    >>> 
    >>> # Access prediction components
    >>> plt.plot(result.energy, result.spectrum)
    >>> print(f"Poles at: {result.poles}")
    """
    
    def __init__(
        self,
        backend: Union[str, BaseEmulator] = "regression",
        n_poles: Optional[int] = None,
        n_components: Optional[int] = None,
        **kwargs,
    ):
        self._backend_name = backend if isinstance(backend, str) else backend.__class__.__name__
        
        # Handle n_poles/n_components unification
        # Allow either n_poles or n_components, prefer explicit n_poles
        effective_n = n_poles if n_poles is not None else n_components
        if effective_n is None:
            effective_n = 5  # default
        self.n_poles = effective_n
        self._kwargs = kwargs
        
        # Create the backend
        if isinstance(backend, BaseEmulator):
            self._emulator = backend
        else:
            self._emulator = self._create_backend(backend, self.n_poles, **kwargs)
    
    @property
    def backend_name(self) -> str:
        """Return the backend name."""
        return self._backend_name
    
    def _create_backend(
        self,
        backend: str,
        n_poles: int,
        **kwargs,
    ) -> BaseEmulator:
        """Create an emulator backend from string identifier."""
        # Import here to avoid circular imports
        from .emulator import StrengthEmulator
        from .pmm import ParametricMatrixModel
        
        backend = backend.lower()
        
        if backend in ("regression", "lorentzian"):
            return StrengthEmulator(n_components=n_poles, **kwargs)
        elif backend in ("pmm", "matrix"):
            return ParametricMatrixModel(n_poles=n_poles, **kwargs)
        else:
            raise ValueError(
                f"Unknown backend '{backend}'. "
                f"Choose from: 'regression', 'lorentzian', 'pmm', 'matrix'"
            )
    
    def fit(
        self,
        dataset: "StrengthDataset",
        **kwargs,
    ) -> "Surrogate":
        """Fit the surrogate model to training data.
        
        Parameters
        ----------
        dataset : StrengthDataset
            Training dataset.
        **kwargs
            Backend-specific fitting options.
            
        Returns
        -------
        self
            Fitted model for method chaining.
        """
        self._emulator.fit(dataset, **kwargs)
        return self
    
    def predict(
        self,
        params: Array,
        energy: Optional[Array] = None,
        **kwargs,
    ) -> EmulatorResult:
        """Predict strength function at a parameter point.
        
        Parameters
        ----------
        params : array
            Parameter vector.
        energy : array, optional
            Energy grid for evaluation.
            
        Returns
        -------
        EmulatorResult
            Unified result with spectrum, poles, and metadata.
        """
        return self._emulator.predict(params, energy, **kwargs)
    
    def predict_batch(
        self,
        params_batch: Array,
        energy: Optional[Array] = None,
        **kwargs,
    ) -> List[EmulatorResult]:
        """Predict for multiple parameter points."""
        return self._emulator.predict_batch(params_batch, energy, **kwargs)
    
    def score(
        self,
        dataset: "StrengthDataset",
        metric: Literal["l2", "mse", "mae"] = "l2",
    ) -> float:
        """Evaluate model accuracy on a test dataset."""
        return self._emulator.score(dataset, metric)
    
    @property
    def backend(self) -> BaseEmulator:
        """Access the underlying emulator backend."""
        return self._emulator
    
    @property
    def is_fitted(self) -> bool:
        """Whether the model has been fitted."""
        return self._emulator.is_fitted
    
    @property
    def param_dim(self) -> Optional[int]:
        """Parameter dimension (set after fitting)."""
        return self._emulator.param_dim
    
    def get_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        info = self._emulator.get_info()
        info["surrogate_backend"] = self._backend_name
        info["n_poles"] = self.n_poles
        return info
    
    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted else "not fitted"
        return f"Surrogate(backend='{self._backend_name}', n_poles={self.n_poles}, {status})"
