from __future__ import annotations

import numpy as np


def absolute_error(predicted, true) -> np.ndarray:
    """Elementwise absolute error."""

    predicted = np.asarray(predicted, dtype=float)
    true = np.asarray(true, dtype=float)
    return np.abs(predicted - true)


def relative_error(predicted, true, *, eps: float = 1e-12) -> np.ndarray:
    """Elementwise absolute relative error."""

    predicted = np.asarray(predicted, dtype=float)
    true = np.asarray(true, dtype=float)
    return np.abs(predicted - true) / np.maximum(np.abs(true), eps)


def summarize_errors(errors) -> dict[str, float]:
    """Return mean/median/max summaries for an error array."""

    errors = np.asarray(errors, dtype=float).reshape(-1)
    if errors.size == 0:
        raise ValueError("Cannot summarize an empty error array.")
    return {
        "mean": float(np.mean(errors)),
        "median": float(np.median(errors)),
        "max": float(np.max(errors)),
    }


def observable_error_summary(predicted, true, *, eps: float = 1e-12, prefix: str = "") -> dict[str, float]:
    """Summarize absolute and relative observable errors."""

    abs_summary = summarize_errors(absolute_error(predicted, true))
    rel_summary = summarize_errors(relative_error(predicted, true, eps=eps))
    prefix = f"{prefix}_" if prefix else ""
    return {
        f"{prefix}mean_absolute_error": abs_summary["mean"],
        f"{prefix}median_absolute_error": abs_summary["median"],
        f"{prefix}max_absolute_error": abs_summary["max"],
        f"{prefix}mean_relative_error": rel_summary["mean"],
        f"{prefix}median_relative_error": rel_summary["median"],
        f"{prefix}max_relative_error": rel_summary["max"],
    }


def integrated_strength_error(predicted, true, energy, *, relative: bool = True, eps: float = 1e-12) -> np.ndarray:
    """Per-spectrum integrated squared strength-function error.

    ``predicted`` and ``true`` may be shaped ``(n_samples, n_energy)`` or
    ``(n_energy,)``. When ``relative=True``, the result is normalized by the
    integrated squared true spectrum for each sample.
    """

    predicted = np.atleast_2d(np.asarray(predicted, dtype=float))
    true = np.atleast_2d(np.asarray(true, dtype=float))
    energy = np.asarray(energy, dtype=float).reshape(-1)
    if predicted.shape != true.shape:
        raise ValueError(f"predicted and true must have the same shape, got {predicted.shape} and {true.shape}.")
    if predicted.shape[1] != energy.size:
        raise ValueError(f"energy has length {energy.size}, expected {predicted.shape[1]}.")

    numer = np.trapezoid((predicted - true) ** 2, energy, axis=1)
    if not relative:
        return numer
    denom = np.maximum(np.trapezoid(true**2, energy, axis=1), eps)
    return numer / denom


def weighted_spectral_loss(predicted, true, energy, weights=None, *, eps: float = 1e-12) -> float:
    """Weighted mean of relative integrated strength-function errors."""

    errors = integrated_strength_error(predicted, true, energy, relative=True, eps=eps)
    if weights is None:
        return float(np.mean(errors))
    weights = np.asarray(weights, dtype=float).reshape(-1)
    if weights.size != errors.size:
        raise ValueError(f"weights has length {weights.size}, expected {errors.size}.")
    total = float(np.sum(weights))
    if total <= 0.0:
        raise ValueError("weights must sum to a positive value.")
    return float(np.sum(weights * errors) / total)


def rmse(predicted, true, axis=None) -> np.ndarray:
    """Root-mean-square error."""

    predicted = np.asarray(predicted, dtype=float)
    true = np.asarray(true, dtype=float)
    return np.sqrt(np.mean((predicted - true) ** 2, axis=axis))
