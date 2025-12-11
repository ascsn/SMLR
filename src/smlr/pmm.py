"""Parametric Matrix Model (PMM) for strength function emulation.

This module implements a reduced-order model based on parametric matrix models,
which is fundamentally different from the regression-based approach in emulator.py.

**Key Innovation:**

Instead of learning a mapping `parameters -> Lorentzian poles` via regression,
the PMM learns a parametric Hamiltonian/response matrix:

    M(p) = D + Σᵢ (pᵢ - p₀ᵢ) · Sᵢ

where:
- D is a diagonal matrix (base eigenvalues)
- Sᵢ are symmetric perturbation matrices
- p is the parameter vector, p₀ is a reference point

The strength function is then:

    S(E; p) = Σₙ |⟨n(p)|v₀|0⟩|² · L(E; Eₙ(p), Γ)

where Eₙ(p) are eigenvalues of M(p) and the transition strengths come from
projecting an external field vector v₀ onto the eigenvectors.

**When to use PMM vs Regression:**

- PMM: Better for physics-based interpolation, sum rule preservation,
  extrapolation when the linear response matrix structure is known/assumed.
- Regression: Simpler, faster training, works well with smooth parameter dependence.


"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize, least_squares

from .base import BaseEmulator, EmulatorResult
from .data import StrengthDataset
from .lorentz import lorentzian_sum

Array = np.ndarray


@dataclass
class PMMConfig:
    """Configuration for Parametric Matrix Model.
    
    Parameters
    ----------
    n_poles : int
        Size of the response matrix (number of poles).
    retain : float
        Fraction of eigenvalues to retain (centered). Default 0.9 means
        keep the middle 90% of eigenvalues, discarding extreme ones.
        This is crucial for good performance - the middle eigenvalues
        typically capture the physical response.
    width_mode : str
        How to handle widths: "global", "per_component", or "parametric".
    optimizer : str
        Optimization backend: "scipy", "tensorflow", "jax".
    max_iterations : int
        Maximum optimization iterations.
    tolerance : float
        Convergence tolerance.
    regularization : float
        L2 regularization on matrix elements.
    energy_window : tuple of float, optional
        (min, max) energy window for masking eigenvalues.
    verbose : bool
        Print progress during training.
    """
    n_poles: int = 10
    retain: float = 0.9  # Critical: fraction of eigenvalues to keep (centered)
    width_mode: Literal["global", "per_component", "parametric"] = "parametric"
    optimizer: str = "scipy"
    max_iterations: int = 5000
    tolerance: float = 1e-8
    regularization: float = 1e-6
    energy_window: Optional[Tuple[float, float]] = None
    verbose: bool = False


@dataclass
class PMMResult:
    """Container for PMM prediction results.
    
    Note: This class is deprecated. Use EmulatorResult from smlr.base instead.
    Kept for backward compatibility.
    
    Attributes
    ----------
    eigenvalues : array
        Pole energies from diagonalization.
    strengths : array
        Transition strengths (squared projections).
    width : float or array
        Pole width(s).
    spectrum : array
        Reconstructed strength function on energy grid.
    energy : array
        Energy grid used.
    """
    eigenvalues: Array
    strengths: Array
    width: Union[float, Array]
    spectrum: Array
    energy: Array


class ParametricMatrixModel(BaseEmulator):
    """Parametric Matrix Model emulator for strength functions.
    
    This emulator learns a reduced-order response matrix whose eigenvalue
    structure reproduces the training spectra. Unlike regression-based
    approaches, PMM maintains the algebraic structure of linear response
    theory, leading to better extrapolation and sum rule preservation.
    
    **Key Innovation - Centered Eigenvalue Retention:**
    
    The PMM keeps only the middle `retain` fraction of eigenvalues when
    computing the spectrum. This is critical because:
    - Extreme eigenvalues are numerical artifacts
    - Physical response is dominated by central eigenmodes
    - This matches the approach in nuclear physics QRPA calculations
    
    Parameters
    ----------
    config : PMMConfig, optional
        Configuration object. If not provided, uses defaults.
    n_poles : int
        Number of poles (matrix dimension). Overrides config if provided.
    retain : float
        Fraction of eigenvalues to retain (centered). Default 0.9.
    **kwargs
        Additional configuration options passed to PMMConfig.
        
    Attributes
    ----------
    param_dim : int
        Dimension of the parameter space (set after fitting).
    reference_point : array
        Central parameter point p₀ for the expansion.
    D : array
        Diagonal matrix (n_poles,).
    S : list of arrays
        Perturbation matrices, one per parameter dimension.
    v0 : array
        External field vector (n_poles,).
    eta : float or array
        Global or per-component width.
    
    Examples
    --------
    >>> from smlr import Surrogate
    >>> 
    >>> # Recommended: use the unified Surrogate interface
    >>> model = Surrogate(backend="pmm", n_poles=10)
    >>> model.fit(dataset)
    >>> result = model.predict(params, energy)
    >>> 
    >>> # Or use ParametricMatrixModel directly
    >>> from smlr.pmm import ParametricMatrixModel
    >>> pmm = ParametricMatrixModel(n_poles=10, retain=0.9)
    >>> pmm.fit(dataset, reference_point=np.array([0.5, 0.5]))
    >>> result = pmm.predict(np.array([0.7, 0.3]), energy_grid)
    """
    
    def __init__(
        self,
        config: Optional[PMMConfig] = None,
        n_poles: Optional[int] = None,
        retain: Optional[float] = None,
        **kwargs,
    ):
        if config is not None:
            self.config = config
        else:
            if n_poles is not None:
                kwargs['n_poles'] = n_poles
            if retain is not None:
                kwargs['retain'] = retain
            self.config = PMMConfig(**kwargs)
        
        self.n_poles = self.config.n_poles
        self.retain = self.config.retain
        self.param_dim: Optional[int] = None
        self.reference_point: Optional[Array] = None
        
        # BaseEmulator interface
        self.n_samples_seen: int = 0
        self._is_fitted: bool = False
        self._default_energy: Optional[Array] = None
        
        # Learned parameters
        self.D: Optional[Array] = None  # Diagonal (n_poles,)
        self.S: Optional[List[Array]] = None  # List of (n_poles, n_poles) matrices
        self.v0: Optional[Array] = None  # External field (n_poles,)
        self.eta: Optional[Union[float, Array]] = None  # Base width
        
        # Parametric width coefficients: width = sqrt(eta² + (x1 + x2*p1 + x3*p2 + ...)²)
        # Stored as [eta_base, x1, x2, x3, ...]
        self.width_coeffs: Optional[Array] = None
        
        # Training history
        self._training_samples: List[Tuple[Array, Array, Array]] = []
        self._cost_history: List[float] = []
    
    def _build_matrix(self, params: Array) -> Array:
        """Construct the response matrix M(p) = D + Σᵢ (pᵢ - p₀ᵢ) · Sᵢ."""
        if self.D is None or self.S is None or self.reference_point is None:
            raise RuntimeError("Model not initialized. Call fit() first.")
        
        params = np.asarray(params, dtype=np.float64)
        delta_p = params - self.reference_point
        
        M = np.diag(self.D)
        for i, dp in enumerate(delta_p):
            if i < len(self.S):
                M = M + dp * self.S[i]
        
        return M
    
    def _compute_spectrum(
        self,
        params: Array,
        energy: Array,
        return_poles: bool = False,
    ) -> Union[Array, Tuple[Array, Array, Array]]:
        """Compute strength function from eigenvalue decomposition.
        
        Uses centered eigenvalue retention: only the middle `retain` fraction
        of eigenvalues are used. This is critical for good performance.
        """
        M = self._build_matrix(params)
        eigenvalues, eigenvectors = eigh(M)
        
        # --- Centered eigenvalue retention ---
        n = len(eigenvalues)
        k_keep = int(round(self.retain * n))
        k_keep = max(1, min(k_keep, n))  # Clamp to valid range
        left = (n - k_keep) // 2
        right = left + k_keep
        
        eigenvalues = eigenvalues[left:right]
        eigenvectors = eigenvectors[:, left:right]
        
        # Transition strengths: |<n|v0>|²
        projections = eigenvectors.T @ self.v0
        strengths = projections ** 2
        
        # Apply energy window mask if specified
        if self.config.energy_window is not None:
            e_min, e_max = self.config.energy_window
            mask = (eigenvalues >= e_min) & (eigenvalues <= e_max)
            strengths = strengths * mask.astype(float)
        
        # --- Parametric width: sqrt(eta² + (x1 + x2*p1 + x3*p2 + ...)²) ---
        if self.config.width_mode == "parametric" and self.width_coeffs is not None:
            params_arr = np.asarray(params, dtype=np.float64)
            eta_base = self.width_coeffs[0]
            x_coeffs = self.width_coeffs[1:]  # [x1, x2, x3, ...]
            
            # Linear combination: x1 + x2*p1 + x3*p2 + ...
            linear_term = x_coeffs[0]  # x1 (constant term)
            if len(x_coeffs) > 1 and len(params_arr) > 0:
                # x2*p1 + x3*p2 + ...
                n_terms = min(len(x_coeffs) - 1, len(params_arr))
                linear_term += np.dot(x_coeffs[1:1+n_terms], params_arr[:n_terms])
            
            width = np.sqrt(eta_base**2 + linear_term**2)
            width = max(width, 0.1)  # Ensure positive
        else:
            width = self.eta
        
        # Reconstruct spectrum
        spectrum = lorentzian_sum(energy, eigenvalues, strengths, width)
        
        if return_poles:
            return spectrum, eigenvalues, strengths
        return spectrum
    
    def _init_parameters(
        self,
        dataset: StrengthDataset,
        reference_point: Optional[Array] = None,
    ) -> Array:
        """Initialize PMM parameters from training data.
        
        Uses the central point (or mean parameters) to initialize D and v0.
        Applies centered placement of fitted Lorentzians following the
        approach in helper.py's encode_initial_guess.
        """
        n = self.n_poles
        self.param_dim = dataset.param_dim
        
        # Set reference point
        if reference_point is not None:
            self.reference_point = np.asarray(reference_point, dtype=np.float64)
        else:
            # Use mean of training parameters
            all_params = np.array([s.params for s in dataset.samples])
            self.reference_point = np.mean(all_params, axis=0)
        
        # Find sample closest to reference point
        all_params = np.array([s.params for s in dataset.samples])
        dists = np.linalg.norm(all_params - self.reference_point, axis=1)
        closest_idx = np.argmin(dists)
        ref_sample = dataset.samples[closest_idx]
        
        # Fit Lorentzians to reference spectrum for initialization
        from .lorentz import fit_lorentzian_mixture
        
        # How many eigenvalues we actually keep
        k_keep = int(round(self.retain * n))
        k_keep = max(1, min(k_keep, n))
        left = (n - k_keep) // 2
        right = left + k_keep
        
        try:
            # Fit more poles than we keep to allow selection
            mix = fit_lorentzian_mixture(
                ref_sample.energy, ref_sample.strength, 
                min(n, k_keep + 4),  # Fit slightly more than k_keep
                width_mode="global"
            )
            E_fit = np.array(mix.energies)
            B_fit = np.array(mix.strengths)
            order = np.argsort(E_fit)
            E_fit, B_fit = E_fit[order], B_fit[order]
            
            # Select centered portion of fitted poles
            m = len(E_fit)
            start = max(0, (m - k_keep) // 2)
            end = min(m, start + k_keep)
            E_sel = E_fit[start:end]
            B_sel = B_fit[start:end]
            
            self.eta = float(np.mean(mix.widths))
        except Exception:
            # Fallback: distribute poles across energy range
            e_min, e_max = ref_sample.energy.min(), ref_sample.energy.max()
            E_sel = np.linspace(e_min + 1, e_max - 1, k_keep)
            B_sel = np.ones(k_keep)
            self.eta = 1.0
        
        # --- Build full D with +/- 2 stepping on ends (like helper.py) ---
        D_full = np.empty(n, dtype=np.float64)
        D_full[left:right] = E_sel if len(E_sel) == k_keep else np.linspace(E_sel[0], E_sel[-1], k_keep)
        
        step = 2.0
        # Left side: decreasing by step
        cur = D_full[left] if left < n else 0
        for i in range(left - 1, -1, -1):
            cur -= step
            D_full[i] = cur
        # Right side: increasing by step
        cur = D_full[right - 1] if right > 0 else 0
        for i in range(right, n):
            cur += step
            D_full[i] = cur
        
        self.D = D_full.astype(np.float64)
        
        # --- Build v0: sqrt(B) in kept block, zeros elsewhere ---
        v0_full = np.zeros(n, dtype=np.float64)
        B_fill = B_sel if len(B_sel) == k_keep else np.ones(k_keep)
        v0_full[left:right] = np.sqrt(np.maximum(B_fill, 1e-10))
        self.v0 = v0_full.astype(np.float64)
        
        # Initialize S matrices as small random symmetric matrices
        rng = np.random.default_rng(42)
        self.S = []
        for _ in range(self.param_dim):
            A = rng.normal(0, 0.1, (n, n)).astype(np.float64)
            S_i = (A + A.T) / 2  # Ensure symmetry
            self.S.append(S_i)
        
        # Parametric width coefficients: [eta_base, x1, x2, x3, ...]
        # width = sqrt(eta² + (x1 + x2*p1 + x3*p2)²)
        # Initialize: eta_base = fitted eta, x1 = 0.1, x2 = x3 = 0
        if self.config.width_mode == "parametric":
            n_coeffs = 1 + self.param_dim + 1  # eta_base + x1 + (x2, x3, ...)
            self.width_coeffs = np.zeros(n_coeffs, dtype=np.float64)
            self.width_coeffs[0] = self.eta  # eta_base
            self.width_coeffs[1] = 0.1  # x1 (constant term)
            # x2, x3, ... start at 0
        
        # Pack all parameters into a single vector
        return self._pack_parameters()
    
    def _pack_parameters(self) -> Array:
        """Pack all learnable parameters into a flat vector."""
        params = []
        
        # Width parameters: eta (or width_coeffs for parametric)
        if self.config.width_mode == "parametric" and self.width_coeffs is not None:
            params.append(self.width_coeffs)
        else:
            params.append(np.array([self.eta]))
        
        # v0
        params.append(self.v0)
        
        # D diagonal
        params.append(self.D)
        
        # S matrices (upper triangular)
        for S_i in self.S:
            upper_tri = S_i[np.triu_indices(self.n_poles)]
            params.append(upper_tri)
        
        return np.concatenate(params)
    
    def _unpack_parameters(self, packed: Array) -> None:
        """Unpack flat vector into model parameters."""
        n = self.n_poles
        idx = 0
        
        # Width parameters
        if self.config.width_mode == "parametric":
            # [eta_base, x1, x2, x3, ...]
            n_width = 1 + self.param_dim + 1  # eta_base + x1 + param coeffs
            self.width_coeffs = packed[idx:idx + n_width].copy()
            self.eta = abs(self.width_coeffs[0])  # Ensure positive
            idx += n_width
        else:
            self.eta = abs(float(packed[idx]))  # Ensure positive
            idx += 1
        
        # v0
        self.v0 = packed[idx:idx + n].copy()
        idx += n
        
        # D diagonal
        self.D = packed[idx:idx + n].copy()
        idx += n
        
        # S matrices
        num_upper = n * (n + 1) // 2
        self.S = []
        for _ in range(self.param_dim):
            upper_tri = packed[idx:idx + num_upper]
            S_i = np.zeros((n, n), dtype=np.float64)
            S_i[np.triu_indices(n)] = upper_tri
            S_i = S_i + S_i.T - np.diag(np.diag(S_i))  # Symmetrize
            self.S.append(S_i)
            idx += num_upper
    
    def _cost_function(
        self,
        packed: Array,
        training_data: List[Tuple[Array, Array, Array]],
    ) -> float:
        """Compute total cost over all training samples."""
        self._unpack_parameters(packed)
        
        total_cost = 0.0
        eps = 1e-16
        
        for params, energy, strength in training_data:
            try:
                pred = self._compute_spectrum(params, energy)
                
                # Normalized L2 loss: ||pred - true||² / ||true||²
                diff = pred - strength
                numer = np.trapezoid(diff ** 2, energy)
                denom = np.trapezoid(strength ** 2, energy) + eps
                total_cost += numer / denom
            except Exception:
                total_cost += 1e10  # Penalize failed computations
        
        # L2 regularization
        if self.config.regularization > 0:
            total_cost += self.config.regularization * np.sum(packed ** 2)
        
        return float(total_cost)
    
    def _cost_gradient(
        self,
        packed: Array,
        training_data: List[Tuple[Array, Array, Array]],
    ) -> Array:
        """Compute gradient via finite differences (for scipy)."""
        eps = 1e-7
        grad = np.zeros_like(packed)
        f0 = self._cost_function(packed, training_data)
        
        for i in range(len(packed)):
            packed_plus = packed.copy()
            packed_plus[i] += eps
            f_plus = self._cost_function(packed_plus, training_data)
            grad[i] = (f_plus - f0) / eps
        
        return grad
    
    def fit(
        self,
        dataset: StrengthDataset,
        reference_point: Optional[Array] = None,
        sample_indices: Optional[Sequence[int]] = None,
        observable_weight: float = 0.0,
        observable_targets: Optional[Dict[str, Array]] = None,
    ) -> "ParametricMatrixModel":
        """Fit the PMM to a dataset of strength functions.
        
        Parameters
        ----------
        dataset : StrengthDataset
            Training dataset.
        reference_point : array, optional
            Central parameter point p₀ for the expansion.
            If not provided, uses the mean of training parameters.
        sample_indices : sequence of int, optional
            Indices of samples to use for training (for subsampling).
        observable_weight : float
            Weight for observable-based loss (e.g., half-life).
        observable_targets : dict, optional
            Target values for observables keyed by sample index.
            
        Returns
        -------
        self
            Fitted PMM.
        """
        # Prepare training data
        if sample_indices is not None:
            samples = [dataset.samples[i] for i in sample_indices]
        else:
            samples = dataset.samples
        
        training_data = [
            (s.params, s.energy, s.strength) for s in samples
        ]
        self._training_samples = training_data
        
        # Initialize parameters
        x0 = self._init_parameters(dataset, reference_point)
        
        if self.config.verbose:
            print(f"PMM Training: {len(training_data)} samples, "
                  f"{self.param_dim}D parameters, {self.n_poles} poles")
            print(f"Total parameters: {len(x0)}")
        
        # Optimize
        self._cost_history = []
        self._iteration_count = [0]  # Use list to allow modification in closure
        
        if self.config.verbose:
            print(f"\nStarting optimization...")
            print(f"  Initial cost: {self._cost_function(x0, training_data):.6f}")
        
        def callback(xk):
            self._iteration_count[0] += 1
            cost = self._cost_function(xk, training_data)
            self._cost_history.append(cost)
            
            if self.config.verbose:
                # Print every 50 iterations or on first few
                if self._iteration_count[0] <= 5 or self._iteration_count[0] % 50 == 0:
                    print(f"  Iteration {self._iteration_count[0]:4d}: cost = {cost:.6e}")
        
        result = minimize(
            lambda x: self._cost_function(x, training_data),
            x0,
            method='L-BFGS-B',
            options={
                'maxiter': self.config.max_iterations,
                'gtol': self.config.tolerance,
            },
            callback=callback,
        )
        
        # Unpack final parameters
        self._unpack_parameters(result.x)
        
        # Store default energy grid and mark as fitted (BaseEmulator interface)
        self._default_energy = samples[0].energy.copy()
        self.n_samples_seen = len(samples)
        self._is_fitted = True
        
        if self.config.verbose:
            print(f"\n{'='*60}")
            print(f"Optimization Results:")
            print(f"  Status: {result.message}")
            print(f"  Total iterations: {result.nit}")
            print(f"  Function evaluations: {result.nfev}")
            print(f"  Initial cost: {self._cost_history[0]:.6e}")
            print(f"  Final cost: {result.fun:.6e}")
            print(f"  Cost reduction: {(1 - result.fun/self._cost_history[0])*100:.2f}%")
            print(f"  Optimized parameters:")
            print(f"    - Width (η): {self.eta:.4f}")
            print(f"    - D eigenvalues range: [{self.D.min():.2f}, {self.D.max():.2f}]")
            print(f"    - v0 norm: {np.linalg.norm(self.v0):.4f}")
            print(f"    - S matrices: {len(self.S)} × ({self.n_poles}×{self.n_poles})")
            print(f"{'='*60}\n")
        
        return self
    
    def predict(
        self,
        params: Array,
        energy: Optional[Array] = None,
        **kwargs,
    ) -> EmulatorResult:
        """Predict strength function at a new parameter point.
        
        Implements the BaseEmulator interface.
        
        Parameters
        ----------
        params : array
            Parameter vector.
        energy : array, optional
            Energy grid. If not provided, uses the grid from training data
            or a default grid based on eigenvalue range.
        **kwargs
            Ignored (for interface compatibility).
            
        Returns
        -------
        EmulatorResult
            Unified result with spectrum, poles, and metadata.
        """
        self._check_fitted()
        
        params = np.asarray(params, dtype=np.float64)
        
        if energy is None:
            if self._default_energy is not None:
                energy = self._default_energy
            else:
                # Fallback: default grid based on eigenvalue range
                e_min = self.D.min() - 5
                e_max = self.D.max() + 5
                energy = np.linspace(e_min, e_max, 200)
        
        energy = np.asarray(energy, dtype=float)
        spectrum, eigenvalues, strengths = self._compute_spectrum(
            params, energy, return_poles=True
        )
        
        # Get width(s)
        width = self.eta
        if np.isscalar(width):
            widths = np.full(len(eigenvalues), width)
        else:
            widths = np.asarray(width)
            if len(widths) != len(eigenvalues):
                widths = np.full(len(eigenvalues), float(np.mean(widths)))
        
        return EmulatorResult(
            spectrum=spectrum,
            energy=energy,
            poles=eigenvalues,
            strengths=strengths,
            widths=widths,
            metadata={
                "backend": "pmm",
                "reference_point": self.reference_point.tolist() if self.reference_point is not None else None,
                "n_poles": self.n_poles,
                "retain": self.retain,
            },
        )
    
    def predict_legacy(
        self,
        params: Array,
        energy: Optional[Array] = None,
    ) -> PMMResult:
        """Legacy predict method returning PMMResult.
        
        Deprecated: Use predict() which returns EmulatorResult.
        
        Parameters
        ----------
        params : array
            Parameter vector.
        energy : array, optional
            Energy grid.
            
        Returns
        -------
        PMMResult
            Legacy result object.
        """
        result = self.predict(params, energy)
        return PMMResult(
            eigenvalues=result.poles,
            strengths=result.strengths,
            width=self.eta,
            spectrum=result.spectrum,
            energy=result.energy,
        )
    
    def predict_batch(
        self,
        params_batch: Array,
        energy: Optional[Array] = None,
        **kwargs,
    ) -> List[EmulatorResult]:
        """Predict for multiple parameter points.
        
        Parameters
        ----------
        params_batch : array of shape (n_samples, param_dim)
            Parameter vectors.
        energy : array, optional
            Shared energy grid.
        **kwargs
            Ignored (for interface compatibility).
            
        Returns
        -------
        list of EmulatorResult
            Predictions for each parameter point.
        """
        params_batch = np.atleast_2d(params_batch)
        return [self.predict(p, energy, **kwargs) for p in params_batch]
    
    def score(
        self,
        dataset: StrengthDataset,
        metric: str = "l2",
    ) -> float:
        """Evaluate emulator accuracy on a dataset.
        
        Parameters
        ----------
        dataset : StrengthDataset
            Test dataset.
        metric : {"l2", "mse", "mae"}
            Error metric to use.
            
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
    
    def get_eigenvalues(self, params: Array) -> Tuple[Array, Array]:
        """Get eigenvalues and strengths without computing full spectrum.
        
        Parameters
        ----------
        params : array
            Parameter vector.
            
        Returns
        -------
        eigenvalues : array
            Pole energies.
        strengths : array
            Transition strengths.
        """
        M = self._build_matrix(params)
        eigenvalues, eigenvectors = eigh(M)
        projections = eigenvectors.T @ self.v0
        strengths = projections ** 2
        return eigenvalues, strengths
    
    def model_diagnostics(self) -> Dict[str, Any]:
        """Return diagnostic information about the fitted model.
        
        Returns
        -------
        dict
            Model diagnostics including matrix properties and training info.
        """
        if self.D is None:
            return {"fitted": False}
        
        # Compute condition numbers of S matrices
        S_norms = [np.linalg.norm(S, 'fro') for S in self.S]
        S_ranks = [np.linalg.matrix_rank(S) for S in self.S]
        
        return {
            "fitted": True,
            "param_dim": self.param_dim,
            "n_poles": self.n_poles,
            "n_training_samples": len(self._training_samples),
            "reference_point": self.reference_point.tolist(),
            "D_range": (float(self.D.min()), float(self.D.max())),
            "v0_norm": float(np.linalg.norm(self.v0)),
            "eta": float(self.eta) if np.isscalar(self.eta) else self.eta.tolist(),
            "S_frobenius_norms": S_norms,
            "S_ranks": S_ranks,
            "cost_history_length": len(self._cost_history),
            "final_cost": self._cost_history[-1] if self._cost_history else None,
        }
    
    def get_info(self) -> Dict[str, Any]:
        """Return information about the fitted emulator.
        
        Implements the BaseEmulator interface.
        """
        info = {
            "backend": "ParametricMatrixModel",
            "is_fitted": self._is_fitted,
            "n_poles": self.n_poles,
            "param_dim": self.param_dim,
            "n_samples_seen": self.n_samples_seen,
            "retain": self.retain,
            "width_mode": self.config.width_mode,
        }
        if self._is_fitted:
            info.update({
                "reference_point": self.reference_point.tolist() if self.reference_point is not None else None,
                "final_cost": self._cost_history[-1] if self._cost_history else None,
            })
        return info


def compare_emulation_methods(
    dataset: StrengthDataset,
    n_components: int = 4,
    test_fraction: float = 0.2,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compare regression-based and PMM emulation on a dataset.
    
    Parameters
    ----------
    dataset : StrengthDataset
        Dataset to use for comparison.
    n_components : int
        Number of poles/components to use.
    test_fraction : float
        Fraction of data to hold out for testing.
    seed : int
        Random seed for train/test split.
        
    Returns
    -------
    dict
        Comparison results with errors and timing for each method.
    """
    from .emulator import StrengthEmulator
    from .metrics import normalized_l2
    import time
    
    rng = np.random.default_rng(seed)
    n_samples = len(dataset)
    n_test = int(n_samples * test_fraction)
    indices = rng.permutation(n_samples)
    train_idx = indices[n_test:]
    test_idx = indices[:n_test]
    
    results = {}
    
    # Regression-based emulator
    t0 = time.time()
    reg_emu = StrengthEmulator(n_components=n_components, regression_method="ridge")
    train_ds = StrengthDataset([dataset.samples[i] for i in train_idx])
    reg_emu.fit(train_ds)
    results['regression_train_time'] = time.time() - t0
    
    # PMM emulator
    t0 = time.time()
    pmm = ParametricMatrixModel(n_poles=n_components, verbose=False)
    pmm.fit(train_ds)
    results['pmm_train_time'] = time.time() - t0
    
    # Evaluate on test set
    reg_errors = []
    pmm_errors = []
    
    for i in test_idx:
        sample = dataset.samples[i]
        
        # Regression prediction
        reg_mix = reg_emu.predict_mixture(sample.params)
        reg_pred = lorentzian_sum(
            sample.energy, reg_mix.energies, reg_mix.strengths, reg_mix.widths
        )
        reg_errors.append(normalized_l2(sample.strength, reg_pred, sample.energy))
        
        # PMM prediction
        pmm_result = pmm.predict(sample.params, sample.energy)
        pmm_errors.append(normalized_l2(sample.strength, pmm_result.spectrum, sample.energy))
    
    results['regression_mean_error'] = float(np.mean(reg_errors))
    results['regression_std_error'] = float(np.std(reg_errors))
    results['pmm_mean_error'] = float(np.mean(pmm_errors))
    results['pmm_std_error'] = float(np.std(pmm_errors))
    results['n_test'] = len(test_idx)
    results['n_train'] = len(train_idx)
    
    return results
