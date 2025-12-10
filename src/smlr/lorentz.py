from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np
from scipy.optimize import least_squares

Array = np.ndarray


@dataclass
class LorentzianMixture:
    energies: Array
    strengths: Array
    widths: Array

    def evaluate(self, energy_grid: Array) -> Array:
        """Evaluate the mixture on the provided energy grid."""
        energy_grid = np.asarray(energy_grid, dtype=float)
        energies = np.asarray(self.energies, dtype=float)
        strengths = np.asarray(self.strengths, dtype=float)
        widths = np.asarray(self.widths, dtype=float)
        widths = np.broadcast_to(widths, strengths.shape)
        num = strengths * (widths / (2 * np.pi))
        den = (energy_grid[:, None] - energies[None, :]) ** 2 + (widths[None, :] ** 2) / 4
        return np.sum(num / den, axis=1)

    def as_tuple(self) -> Tuple[Array, Array, Array]:
        return self.energies, self.strengths, self.widths


# --- internal helpers -------------------------------------------------------

def _softplus(x: Array) -> Array:
    x = np.asarray(x, dtype=float)
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def _inv_softplus(y: Array) -> Array:
    y = np.maximum(np.asarray(y, dtype=float), 1e-12)
    return np.log(np.expm1(y))


def lorentzian_sum(energy: Array, centers: Array, strengths: Array, widths: Array) -> Array:
    energy = np.asarray(energy, dtype=float)
    centers = np.asarray(centers, dtype=float)
    strengths = np.asarray(strengths, dtype=float)
    widths = np.asarray(widths, dtype=float)
    widths = np.broadcast_to(widths, strengths.shape)
    num = strengths * (widths / (2 * np.pi))
    den = (energy[:, None] - centers[None, :]) ** 2 + (widths[None, :] ** 2) / 4
    return np.sum(num / den, axis=1)


# --- fitting ---------------------------------------------------------------

def fit_lorentzian_mixture(
    energy: Array,
    strength: Array,
    n_components: int,
    *,
    width_mode: Literal["global", "per_component"] = "global",
    eta_init: float = 0.5,
    min_spacing: float = 0.1,
    l2: float = 1e-3,
) -> LorentzianMixture:
    """Fit a Lorentzian mixture with soft constraints using SciPy.

    Parameters
    ----------
    energy, strength : 1D arrays
        Sampled spectrum.
    n_components : int
        Number of Lorentzian poles.
    width_mode : "global" | "per_component"
        Whether to fit a single width shared by all components or one per component.
    eta_init : float
        Initial width guess (MeV).
    min_spacing : float
        Enforce increasing pole energies with this minimum spacing.
    l2 : float
        Small L2 penalty to regularize the latent variables.
    """

    energy = np.asarray(energy, dtype=float)
    strength = np.asarray(strength, dtype=float)
    if energy.ndim != 1 or strength.ndim != 1:
        raise ValueError("energy and strength must be 1D arrays")
    if len(energy) != len(strength):
        raise ValueError("energy and strength must match lengths")
    if n_components < 1:
        raise ValueError("n_components must be >= 1")

    wmin, wmax = float(np.min(energy)), float(np.max(energy))
    widths_size = 1 if width_mode == "global" else n_components

    # rough seed using evenly spaced centers and NNLS-like strengths
    centers0 = np.linspace(wmin + 1e-3, wmax - 1e-3, n_components)
    strengths0 = np.maximum(strength.max(), 1e-6) * np.ones(n_components)
    widths0 = eta_init * np.ones(widths_size)

    zE = _inv_softplus(np.r_[centers0[0] - wmin, np.diff(centers0) - min_spacing])
    zB = _inv_softplus(np.sqrt(np.maximum(strengths0, 1e-12)))
    zW = _inv_softplus(widths0)
    z0 = np.concatenate([zE, zB, zW])

    def unpack(z: Array):
        zE, zB, zW = z[:n_components], z[n_components : 2 * n_components], z[2 * n_components :]
        e0 = wmin + _softplus(zE[0])
        gaps = _softplus(zE[1:]) + min_spacing
        centers = np.concatenate([[e0], e0 + np.cumsum(gaps)])
        strengths_np = _softplus(zB) ** 2
        widths_np = _softplus(zW)
        widths_full = np.broadcast_to(widths_np, (n_components,))
        return centers, strengths_np, widths_full

    def residuals(z: Array) -> Array:
        centers, strengths_np, widths_np = unpack(z)
        model = lorentzian_sum(energy, centers, strengths_np, widths_np)
        res = model - strength
        if l2 > 0:
            res = np.concatenate([res, np.sqrt(l2) * z])
        return res

    result = least_squares(residuals, z0, method="trf", max_nfev=5000)
    centers, strengths_np, widths_np = unpack(result.x)
    return LorentzianMixture(centers, strengths_np, widths_np)
