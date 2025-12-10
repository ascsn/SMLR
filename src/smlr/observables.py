"""Observable functions for physics-informed emulator training.

This module provides a framework for computing physics observables from strength
functions. These observables can be used as additional loss terms during emulator
training to ensure physical constraints are satisfied.

Supported observables include:
- Energy-weighted sum rules (m_k for any k)
- Dipole polarizability (alpha_D)
- Beta decay half-lives
- Thomas-Reiche-Kuhn sum rule
- Custom user-defined observables

The design supports arbitrary parameter dimensions, not just 2D parameter spaces.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import integrate

Array = np.ndarray


@dataclass
class ObservableResult:
    """Container for observable computation results."""
    name: str
    value: float
    target: Optional[float] = None
    error: Optional[float] = None
    relative_error: Optional[float] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.target is not None and self.error is None:
            self.error = abs(self.value - self.target)
            if abs(self.target) > 1e-12:
                self.relative_error = self.error / abs(self.target)


class Observable(abc.ABC):
    """Base class for physics observables.
    
    Observables compute scalar quantities from strength function data.
    They can optionally be used in optimization with target values.
    
    Parameters
    ----------
    name : str
        Human-readable name for the observable.
    weight : float
        Weight in combined loss function (default 1.0).
    target : float, optional
        Target value for optimization. If None, no loss is computed.
    normalize_loss : bool
        If True, normalize loss by target magnitude (default True).
    """
    
    def __init__(
        self,
        name: str,
        weight: float = 1.0,
        target: Optional[float] = None,
        normalize_loss: bool = True,
    ):
        self.name = name
        self.weight = weight
        self.target = target
        self.normalize_loss = normalize_loss
    
    @abc.abstractmethod
    def compute(self, energy: Array, strength: Array, **kwargs) -> float:
        """Compute the observable value.
        
        Parameters
        ----------
        energy : Array
            Energy grid (1D).
        strength : Array
            Strength function values (1D).
        **kwargs
            Additional parameters (e.g., from Lorentzian decomposition).
            
        Returns
        -------
        float
            Observable value.
        """
        pass
    
    def compute_with_result(self, energy: Array, strength: Array, **kwargs) -> ObservableResult:
        """Compute observable and return full result object."""
        value = self.compute(energy, strength, **kwargs)
        return ObservableResult(
            name=self.name,
            value=value,
            target=self.target,
        )
    
    def loss(self, energy: Array, strength: Array, **kwargs) -> float:
        """Compute loss term for optimization.
        
        Returns weighted squared error if target is set, else 0.
        """
        if self.target is None:
            return 0.0
        value = self.compute(energy, strength, **kwargs)
        error = value - self.target
        if self.normalize_loss and abs(self.target) > 1e-12:
            error = error / abs(self.target)
        return self.weight * error ** 2


class SumRule(Observable):
    """Energy-weighted sum rule m_k = ∫ E^k S(E) dE.
    
    Parameters
    ----------
    k : int
        Power of energy weighting.
    energy_min : float, optional
        Lower integration limit. If None, use full range.
    energy_max : float, optional
        Upper integration limit. If None, use full range.
    **kwargs
        Additional arguments passed to Observable base class.
        
    Examples
    --------
    >>> m0 = SumRule(k=0, name="m0")  # Non-energy-weighted sum
    >>> m1 = SumRule(k=1, name="TRK", target=875.0)  # Thomas-Reiche-Kuhn
    >>> m_minus1 = SumRule(k=-1, name="m_{-1}")  # For polarizability
    """
    
    def __init__(
        self,
        k: int = 0,
        name: Optional[str] = None,
        energy_min: Optional[float] = None,
        energy_max: Optional[float] = None,
        **kwargs
    ):
        name = name or f"m_{k}"
        super().__init__(name=name, **kwargs)
        self.k = k
        self.energy_min = energy_min
        self.energy_max = energy_max
    
    def compute(self, energy: Array, strength: Array, **kwargs) -> float:
        energy = np.asarray(energy, dtype=float)
        strength = np.asarray(strength, dtype=float)
        
        # Apply energy range if specified
        mask = np.ones(len(energy), dtype=bool)
        if self.energy_min is not None:
            mask &= energy >= self.energy_min
        if self.energy_max is not None:
            mask &= energy <= self.energy_max
        
        e_masked = energy[mask]
        s_masked = strength[mask]
        
        if len(e_masked) < 2:
            return 0.0
        
        # Handle k < 0 carefully (avoid division by zero)
        if self.k < 0:
            # Mask out zero or negative energies for negative powers
            valid = e_masked > 1e-10
            e_valid = e_masked[valid]
            s_valid = s_masked[valid]
            if len(e_valid) < 2:
                return 0.0
            integrand = s_valid * (e_valid ** self.k)
        else:
            integrand = s_masked * (e_masked ** self.k)
            e_valid = e_masked
        
        return float(np.trapezoid(integrand, e_valid))


class DipolePolarizability(Observable):
    """Dipole polarizability alpha_D from inverse energy-weighted sum rule.
    
    alpha_D = (8π α_c ħc / 9) ∫ S(E)/E dE
    
    where α_c ≈ 1/137 is the fine-structure constant.
    
    Parameters
    ----------
    hbar_c : float
        ħc in MeV·fm (default 197.33).
    alpha_c : float
        Fine-structure constant (default 1/137).
    energy_min : float
        Minimum energy for integration (to avoid singularity at E=0).
    **kwargs
        Additional arguments passed to Observable base class.
    """
    
    def __init__(
        self,
        hbar_c: float = 197.33,
        alpha_c: float = 1 / 137,
        energy_min: float = 1.0,
        name: str = "alpha_D",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.hbar_c = hbar_c
        self.alpha_c = alpha_c
        self.energy_min = energy_min
        self.prefactor = 8.0 * np.pi * alpha_c * hbar_c / 9.0
    
    def compute(self, energy: Array, strength: Array, **kwargs) -> float:
        energy = np.asarray(energy, dtype=float)
        strength = np.asarray(strength, dtype=float)
        
        mask = energy > self.energy_min
        e_masked = energy[mask]
        s_masked = strength[mask]
        
        if len(e_masked) < 2:
            return 0.0
        
        integrand = s_masked / e_masked
        integral = np.trapezoid(integrand, e_masked)
        return float(self.prefactor * integral)
    
    def compute_from_poles(
        self,
        pole_energies: Array,
        pole_strengths: Array,
        energy_min: Optional[float] = None,
    ) -> float:
        """Compute alpha_D directly from pole decomposition.
        
        This is more efficient than integrating the full spectrum.
        
        Parameters
        ----------
        pole_energies : Array
            Pole positions (resonance energies).
        pole_strengths : Array
            Pole strengths (integrated areas).
        energy_min : float, optional
            Minimum energy threshold (default: self.energy_min).
            
        Returns
        -------
        float
            Dipole polarizability value.
        """
        e_min = energy_min if energy_min is not None else self.energy_min
        pole_energies = np.asarray(pole_energies, dtype=float)
        pole_strengths = np.asarray(pole_strengths, dtype=float)
        
        mask = pole_energies > e_min
        return float(self.prefactor * np.sum(pole_strengths[mask] / pole_energies[mask]))


class BetaDecayHalfLife(Observable):
    """Beta decay half-life from Gamow-Teller strength function.
    
    t_{1/2} = ln(2) / λ
    
    where λ = Σ_n f(Z, E_n) |M_n|^2 g_A^2 / K
    
    Parameters
    ----------
    Z : int
        Atomic number of parent nucleus.
    A : int
        Mass number.
    Q_value : float
        Q-value (endpoint energy) in MeV.
    g_A : float
        Axial-vector coupling constant (default 1.27).
    kappa : float
        Constant K in seconds (default 6147).
    **kwargs
        Additional arguments passed to Observable base class.
    """
    
    def __init__(
        self,
        Z: int,
        A: int,
        Q_value: float,
        g_A: float = 1.27,
        kappa: float = 6147.0,
        name: str = "t_half",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.Z = Z
        self.A = A
        self.Q_value = Q_value
        self.g_A = g_A
        self.kappa = kappa
        self.m_e = 0.511  # electron mass in MeV
        self.alpha_c = 1 / 137
        self.delta_np = 1.293  # neutron-proton mass difference in MeV
    
    def fermi_function(self, W: float) -> float:
        """Fermi function F(Z, W) for electron energy W (in units of m_e)."""
        from scipy.special import gamma as gamma_fn
        
        R = 1.2 * self.A ** (1/3)  # nuclear radius in fm
        compton = 197.33 / self.m_e  # Compton wavelength in fm
        R = R / compton
        
        gamma_1 = np.sqrt(1 - (self.alpha_c * (self.Z + 1)) ** 2)
        if W <= 1:
            return 0.0
        p = np.sqrt(W ** 2 - 1)
        y = self.alpha_c * (self.Z + 1) * W / p
        
        gamma_part = np.abs(gamma_fn(gamma_1 + 1j * y)) ** 2 / gamma_fn(2 * gamma_1 + 1) ** 2
        part_1 = 4 * (2 * p * R) ** (-2 * (1 - gamma_1))
        part_2 = np.exp(np.pi * y)
        L_0 = 0.5 * (1 + gamma_1)
        
        return part_1 * part_2 * gamma_part * L_0
    
    def phase_space_integrand(self, W_0: float, W: float) -> float:
        """Phase space integrand p W (W_0 - W)^2 F(Z, W)."""
        if W <= 1 or W >= W_0:
            return 0.0
        p = np.sqrt(W ** 2 - 1)
        return p * W * (W_0 - W) ** 2 * self.fermi_function(W)
    
    def phase_space_factor(self, W_0: float) -> float:
        """Compute phase space factor f(Z, W_0)."""
        if W_0 <= 1:
            return 0.0
        result, _ = integrate.quad(
            lambda W: self.phase_space_integrand(W_0, W),
            1.0, W_0, limit=100
        )
        return result
    
    def compute(self, energy: Array, strength: Array, **kwargs) -> float:
        """Compute half-life from continuous spectrum."""
        energy = np.asarray(energy, dtype=float)
        strength = np.asarray(strength, dtype=float)
        
        # Only contributions below Q-value contribute
        mask = energy < self.Q_value
        e_masked = energy[mask]
        s_masked = strength[mask]
        
        if len(e_masked) < 2:
            return float('inf')
        
        # For each energy bin, compute the contribution
        total_rate = 0.0
        for i in range(len(e_masked)):
            E_n = e_masked[i]
            W_0 = (self.delta_np - E_n) / self.m_e
            if W_0 > 1:
                f = self.phase_space_factor(W_0)
                total_rate += s_masked[i] * f
        
        # Apply coupling and constants
        total_rate *= self.g_A ** 2 * np.log(2) / self.kappa
        
        if total_rate <= 0:
            return float('inf')
        
        return float(np.log(2) / total_rate)
    
    def compute_from_poles(
        self,
        pole_energies: Array,
        pole_strengths: Array,
    ) -> float:
        """Compute half-life directly from pole decomposition.
        
        More efficient than integrating full spectrum.
        """
        pole_energies = np.asarray(pole_energies, dtype=float)
        pole_strengths = np.asarray(pole_strengths, dtype=float)
        
        total_rate = 0.0
        for E_n, B_n in zip(pole_energies, pole_strengths):
            if E_n < self.Q_value:
                W_0 = (self.delta_np - E_n) / self.m_e
                if W_0 > 1:
                    f = self.phase_space_factor(W_0)
                    total_rate += B_n * f
        
        total_rate *= self.g_A ** 2 * np.log(2) / self.kappa
        
        if total_rate <= 0:
            return float('inf')
        
        return float(np.log(2) / total_rate)


class CustomObservable(Observable):
    """User-defined observable from a custom function.
    
    Parameters
    ----------
    compute_fn : Callable
        Function (energy, strength, **kwargs) -> float.
    name : str
        Name for the observable.
    **kwargs
        Additional arguments passed to Observable base class.
        
    Examples
    --------
    >>> def my_obs(energy, strength, **kw):
    ...     return np.sum(strength * np.exp(-energy**2))
    >>> obs = CustomObservable(my_obs, name="gaussian_weighted", target=10.0)
    """
    
    def __init__(
        self,
        compute_fn: Callable[[Array, Array], float],
        name: str = "custom",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self._compute_fn = compute_fn
    
    def compute(self, energy: Array, strength: Array, **kwargs) -> float:
        return float(self._compute_fn(energy, strength, **kwargs))


@dataclass
class ObservableSet:
    """Collection of observables for multi-objective optimization.
    
    Parameters
    ----------
    observables : list of Observable
        List of observables to compute.
        
    Examples
    --------
    >>> obs_set = ObservableSet([
    ...     SumRule(k=1, target=875.0, weight=1.0),
    ...     DipolePolarizability(target=18.5, weight=0.5),
    ... ])
    >>> results = obs_set.compute_all(energy, strength)
    >>> total_loss = obs_set.total_loss(energy, strength)
    """
    observables: List[Observable]
    
    def compute_all(self, energy: Array, strength: Array, **kwargs) -> List[ObservableResult]:
        """Compute all observables."""
        return [obs.compute_with_result(energy, strength, **kwargs) for obs in self.observables]
    
    def total_loss(self, energy: Array, strength: Array, **kwargs) -> float:
        """Compute weighted sum of all observable losses."""
        return sum(obs.loss(energy, strength, **kwargs) for obs in self.observables)
    
    def to_dict(self, energy: Array, strength: Array, **kwargs) -> Dict[str, float]:
        """Return dictionary of observable values."""
        return {obs.name: obs.compute(energy, strength, **kwargs) for obs in self.observables}
    
    def add(self, observable: Observable) -> "ObservableSet":
        """Add an observable to the set."""
        self.observables.append(observable)
        return self
    
    def __iter__(self):
        return iter(self.observables)
    
    def __len__(self):
        return len(self.observables)


# Convenience factory functions
def create_trk_sum_rule(target: float = 875.0, weight: float = 1.0) -> SumRule:
    """Create a Thomas-Reiche-Kuhn (m_1) sum rule observable."""
    return SumRule(k=1, name="TRK", target=target, weight=weight)


def create_polarizability(target: float, weight: float = 1.0, **kwargs) -> DipolePolarizability:
    """Create a dipole polarizability observable."""
    return DipolePolarizability(target=target, weight=weight, **kwargs)


def create_half_life(
    Z: int, A: int, Q_value: float,
    target: Optional[float] = None,
    weight: float = 1.0,
    use_log: bool = True,
    **kwargs
) -> BetaDecayHalfLife:
    """Create a beta decay half-life observable.
    
    Parameters
    ----------
    use_log : bool
        If True, use log-scale for loss computation (recommended for half-lives
        which can span many orders of magnitude).
    """
    obs = BetaDecayHalfLife(Z=Z, A=A, Q_value=Q_value, target=target, weight=weight, **kwargs)
    if use_log and target is not None:
        # Override loss to use log-scale
        original_loss = obs.loss
        def log_loss(energy, strength, **kw):
            if obs.target is None or obs.target <= 0:
                return 0.0
            value = obs.compute(energy, strength, **kw)
            if value <= 0:
                return float('inf')
            error = np.log(value) - np.log(obs.target)
            return obs.weight * error ** 2
        obs.loss = log_loss
    return obs
