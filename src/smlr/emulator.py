from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline

from .data import StrengthDataset, StrengthSample
from .lorentz import LorentzianMixture, fit_lorentzian_mixture

Array = np.ndarray


@dataclass
class FittedMixture:
    """Container for a single fitted Lorentzian mixture."""
    params: Array
    mixture: LorentzianMixture
    energy: Array
    strength: Array


@dataclass 
class EmulatorConfig:
    """Configuration for emulator training and prediction.
    
    Parameters
    ----------
    n_components : int
        Number of Lorentzian components in the mixture.
    width_mode : str
        How to handle widths: "global" (single width) or "per_component".
    regression_method : str
        Regression model: "linear", "ridge", "polynomial", or "gp".
    poly_degree : int
        Polynomial degree (only used if regression_method="polynomial").
    gp_kernel : str
        GP kernel type: "rbf", "matern32", "matern52" (only for regression_method="gp").
    alpha : float
        Regularization strength for ridge regression.
    normalize_strengths : bool
        Whether to apply log1p transform to strengths.
    random_state : int
        Random seed for reproducibility.
    """
    n_components: int = 4
    width_mode: Literal["global", "per_component"] = "global"
    regression_method: Literal["linear", "ridge", "polynomial", "gp"] = "linear"
    poly_degree: int = 2
    gp_kernel: Literal["rbf", "matern32", "matern52"] = "matern52"
    alpha: float = 1e-3
    normalize_strengths: bool = True
    random_state: int = 0


class StrengthEmulator:
    """Learn a mapping from theory parameters to Lorentzian mixture parameters.
    
    This emulator supports arbitrary parameter dimensions (not limited to 2D).
    It learns to predict pole energies, strengths, and widths as functions of
    the input physics parameters.
    
    Parameters
    ----------
    n_components : int
        Number of Lorentzian components (poles) in the mixture model.
    width_mode : {"global", "per_component"}
        How to model widths. "global" uses a single width for all poles.
        "per_component" learns a separate width per pole.
    regression_method : {"linear", "ridge", "polynomial", "gp"}
        Which regression model to use for the parameter -> pole mapping.
        - "linear": Simple linear regression (fast, may underfit).
        - "ridge": L2-regularized linear regression (recommended default).
        - "polynomial": Polynomial features + ridge (captures nonlinearity).
        - "gp": Gaussian Process regression (best for uncertainty, expensive).
    poly_degree : int
        Polynomial degree when using regression_method="polynomial".
    alpha : float
        Regularization strength for ridge/polynomial regression.
    random_state : int
        Random seed for reproducibility.
        
    Attributes
    ----------
    param_dim : int
        Dimension of the parameter space (set after fitting).
    n_samples_seen : int
        Number of training samples used.
        
    Examples
    --------
    >>> from smlr.data import StrengthDataset
    >>> from smlr.emulator import StrengthEmulator
    >>> 
    >>> # Works with any parameter dimension (2, 5, 10, 15, ...)
    >>> ds = StrengthDataset.from_folder("metadata.csv", param_columns=["p1", "p2", "p3", "p4", "p5"])
    >>> emu = StrengthEmulator(n_components=4, regression_method="ridge")
    >>> emu.fit(ds)
    >>> 
    >>> # Predict at new 5D parameter point
    >>> mixture = emu.predict_mixture(np.array([0.1, 0.2, 0.3, 0.4, 0.5]))
    """

    def __init__(
        self,
        n_components: int,
        *,
        width_mode: Literal["global", "per_component"] = "global",
        regression_method: Literal["linear", "ridge", "polynomial", "gp"] = "linear",
        poly_degree: int = 2,
        alpha: float = 1e-3,
        random_state: int = 0,
    ) -> None:
        if width_mode not in {"global", "per_component"}:
            raise ValueError("width_mode must be 'global' or 'per_component'")
        if regression_method not in {"linear", "ridge", "polynomial", "gp"}:
            raise ValueError("regression_method must be 'linear', 'ridge', 'polynomial', or 'gp'")
            
        self.n_components = int(n_components)
        self.width_mode = width_mode
        self.regression_method = regression_method
        self.poly_degree = poly_degree
        self.alpha = alpha
        self.random_state = random_state
        self._normalize_strengths = True
        self.widths_size = 1 if width_mode == "global" else self.n_components

        self.param_scaler = StandardScaler()
        self.energy_scaler = StandardScaler()
        self.strength_scaler = StandardScaler()
        self.width_scaler = StandardScaler()

        self.energy_reg: Optional[MultiOutputRegressor] = None
        self.strength_reg: Optional[MultiOutputRegressor] = None
        self.width_reg: Optional[LinearRegression | MultiOutputRegressor] = None
        
        # Track parameter dimension and samples
        self.param_dim: Optional[int] = None
        self.n_samples_seen: int = 0
        
        # Store fitted mixtures for diagnostics
        self._fitted_mixtures: List[FittedMixture] = []

    def _create_regressor(self) -> Any:
        """Create a single regression estimator based on configuration."""
        if self.regression_method == "linear":
            return LinearRegression()
        elif self.regression_method == "ridge":
            return Ridge(alpha=self.alpha, random_state=self.random_state)
        elif self.regression_method == "polynomial":
            return Pipeline([
                ("poly", PolynomialFeatures(degree=self.poly_degree, include_bias=False)),
                ("ridge", Ridge(alpha=self.alpha, random_state=self.random_state)),
            ])
        elif self.regression_method == "gp":
            kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
            return GaussianProcessRegressor(
                kernel=kernel,
                alpha=1e-6,
                random_state=self.random_state,
                n_restarts_optimizer=3,
            )
        else:
            raise ValueError(f"Unknown regression method: {self.regression_method}")

    # ---------------------- fitting pipeline ----------------------
    def _encode_mixture(self, mixture: LorentzianMixture) -> Tuple[Array, Array, Array]:
        energies = np.asarray(mixture.energies, dtype=float)
        strengths = np.asarray(mixture.strengths, dtype=float)
        widths = np.asarray(mixture.widths, dtype=float)
        order = np.argsort(energies)
        energies = energies[order]
        strengths = strengths[order]
        widths = widths if widths.shape == strengths.shape else np.broadcast_to(widths, strengths.shape)
        widths = widths[order]
        return energies, strengths, widths

    def fit(
        self,
        dataset: StrengthDataset,
        *,
        mixtures: Optional[Sequence[LorentzianMixture]] = None,
        fitter: Callable[..., LorentzianMixture] = fit_lorentzian_mixture,
        fit_kwargs: Optional[dict] = None,
        normalize_strengths: bool = True,
        verbose: bool = False,
    ) -> "StrengthEmulator":
        """Fit the emulator on a dataset of strength functions.

        The fitting process consists of:
        1. For each sample, fit a Lorentzian mixture (or use provided mixtures).
        2. Extract pole parameters (energies, strengths, widths) from each mixture.
        3. Train regression models to predict poles from input parameters.
        
        Parameters
        ----------
        dataset : StrengthDataset
            Dataset containing strength function samples with parameters.
            Works with any parameter dimension (2D, 5D, 10D, etc.).
        mixtures : sequence of LorentzianMixture, optional
            Pre-fitted mixtures. If provided, must align with dataset order.
            Useful for caching expensive fits.
        fitter : callable
            Function to fit Lorentzian mixtures if not provided.
        fit_kwargs : dict, optional
            Additional arguments passed to the fitter.
        normalize_strengths : bool
            Whether to apply log1p transform to strengths (helps with wide ranges).
        verbose : bool
            Whether to print progress.
            
        Returns
        -------
        self
            Fitted emulator.
        """

        fit_kwargs = fit_kwargs or {}
        self.param_dim = dataset.param_dim
        self.n_samples_seen = len(dataset)

        fitted: list[FittedMixture] = []
        if mixtures is not None:
            if len(mixtures) != len(dataset):
                raise ValueError("mixtures length must match dataset")
            for sample, mix in zip(dataset.samples, mixtures):
                fitted.append(FittedMixture(sample.params, mix, sample.energy, sample.strength))
        else:
            for i, sample in enumerate(dataset.samples):
                if verbose and (i + 1) % 10 == 0:
                    print(f"Fitting mixture {i + 1}/{len(dataset)}")
                mix = fitter(
                    sample.energy,
                    sample.strength,
                    self.n_components,
                    width_mode=self.width_mode,
                    **fit_kwargs,
                )
                fitted.append(FittedMixture(sample.params, mix, sample.energy, sample.strength))
        
        self._fitted_mixtures = fitted

        params = np.vstack([f.params for f in fitted])
        energy_mat = []
        strength_mat = []
        width_vec = []
        for f in fitted:
            e, s, w = self._encode_mixture(f.mixture)
            energy_mat.append(e)
            strength_mat.append(np.log1p(s) if normalize_strengths else s)
            width_vec.append(w if self.width_mode == "per_component" else [w.mean()])

        X = self.param_scaler.fit_transform(params)
        E_targets = self.energy_scaler.fit_transform(np.vstack(energy_mat))
        S_targets = self.strength_scaler.fit_transform(np.vstack(strength_mat))
        W_targets = np.vstack(width_vec)
        W_targets = self.width_scaler.fit_transform(W_targets)

        # Create regressors using configured method
        self.energy_reg = MultiOutputRegressor(self._create_regressor())
        self.strength_reg = MultiOutputRegressor(self._create_regressor())
        self.energy_reg.fit(X, E_targets)
        self.strength_reg.fit(X, S_targets)

        if self.width_mode == "global":
            self.width_reg = self._create_regressor()
            self.width_reg.fit(X, W_targets.ravel())
        else:
            self.width_reg = MultiOutputRegressor(self._create_regressor())
            self.width_reg.fit(X, W_targets)
        self._normalize_strengths = normalize_strengths
        
        if verbose:
            print(f"Emulator fitted: {self.n_samples_seen} samples, {self.param_dim}D parameter space")
        
        return self

    # ---------------------- prediction -----------------------------
    def _check_fitted(self):
        if self.energy_reg is None or self.strength_reg is None or self.width_reg is None:
            raise RuntimeError("Emulator not fitted. Call `fit` first.")

    def predict_mixture(self, params: Array) -> LorentzianMixture:
        """Predict Lorentzian mixture for given parameter values.
        
        Parameters
        ----------
        params : array-like
            Parameter vector of shape (param_dim,). Must match the dimension
            used during training.
            
        Returns
        -------
        LorentzianMixture
            Predicted pole decomposition.
        """
        self._check_fitted()
        params = np.atleast_1d(np.asarray(params, dtype=float))
        if self.param_dim is not None and len(params) != self.param_dim:
            raise ValueError(
                f"Expected {self.param_dim}D parameter vector, got {len(params)}D"
            )
        X = self.param_scaler.transform(params.reshape(1, -1))
        e_pred = self.energy_reg.predict(X)
        s_pred = self.strength_reg.predict(X)
        e_pred = self.energy_scaler.inverse_transform(e_pred)[0]
        s_pred = self.strength_scaler.inverse_transform(s_pred)[0]
        if self._normalize_strengths:
            s_pred = np.expm1(s_pred)
        w_pred = self.width_reg.predict(X)
        if self.width_mode == "global":
            w_pred = np.asarray(w_pred).reshape(1, -1)
            w_pred = self.width_scaler.inverse_transform(w_pred)[0]
            widths = np.full(self.n_components, np.maximum(w_pred[0], 1e-6))
        else:
            w_pred = np.asarray(w_pred).reshape(1, -1)
            w_pred = self.width_scaler.inverse_transform(w_pred)[0]
            widths = np.maximum(w_pred[: self.n_components], 1e-6)
        strengths = np.maximum(s_pred, 0.0)
        energies = e_pred
        order = np.argsort(energies)
        return LorentzianMixture(energies[order], strengths[order], widths[order])

    def predict_spectrum(self, params: Array, energy_grid: Array) -> Array:
        """Predict strength function on an energy grid.
        
        Parameters
        ----------
        params : array-like
            Parameter vector of shape (param_dim,).
        energy_grid : array-like
            Energy values at which to evaluate the spectrum.
            
        Returns
        -------
        array
            Predicted strength function values.
        """
        mixture = self.predict_mixture(params)
        return mixture.evaluate(energy_grid)

    def predict(
        self, params: Array, energy_grid: Optional[Array] = None
    ) -> Tuple[LorentzianMixture, Optional[Array]]:
        """Predict both mixture and spectrum.
        
        Parameters
        ----------
        params : array-like
            Parameter vector of shape (param_dim,).
        energy_grid : array-like, optional
            If provided, also evaluate the spectrum on this grid.
            
        Returns
        -------
        mixture : LorentzianMixture
            Predicted pole decomposition.
        spectrum : array or None
            Evaluated spectrum if energy_grid provided.
        """
        mixture = self.predict_mixture(params)
        if energy_grid is None:
            return mixture, None
        return mixture, mixture.evaluate(energy_grid)
    
    def predict_batch(
        self, params_batch: Array, energy_grid: Optional[Array] = None
    ) -> Tuple[List[LorentzianMixture], Optional[Array]]:
        """Predict for multiple parameter points at once.
        
        Parameters
        ----------
        params_batch : array-like
            Parameter matrix of shape (n_points, param_dim).
        energy_grid : array-like, optional
            If provided, evaluate spectra on this grid.
            
        Returns
        -------
        mixtures : list of LorentzianMixture
            Predicted pole decompositions.
        spectra : array or None
            Shape (n_points, len(energy_grid)) if energy_grid provided.
        """
        params_batch = np.atleast_2d(params_batch)
        mixtures = [self.predict_mixture(p) for p in params_batch]
        if energy_grid is None:
            return mixtures, None
        spectra = np.array([m.evaluate(energy_grid) for m in mixtures])
        return mixtures, spectra
    
    def score(self, dataset: StrengthDataset, metric: str = "l2") -> float:
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
        
        errors = []
        for sample in dataset.samples:
            pred = self.predict_spectrum(sample.params, sample.energy)
            if metric == "l2":
                err = normalized_l2(pred, sample.strength, sample.energy)
            elif metric == "mse":
                err = np.mean((pred - sample.strength) ** 2)
            elif metric == "mae":
                err = np.mean(np.abs(pred - sample.strength))
            else:
                raise ValueError(f"Unknown metric: {metric}")
            errors.append(err)
        return float(np.mean(errors))
    
    def get_training_info(self) -> Dict[str, Any]:
        """Return information about the fitted emulator."""
        return {
            "n_components": self.n_components,
            "param_dim": self.param_dim,
            "n_samples_seen": self.n_samples_seen,
            "width_mode": self.width_mode,
            "regression_method": self.regression_method,
            "normalize_strengths": self._normalize_strengths,
        }


def _softplus_single(x: Array) -> Array:
    x = np.asarray(x, dtype=float)
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)
