from __future__ import annotations

import numpy as np
from scipy import integrate

Array = np.ndarray


def normalized_l2(y_pred: Array, y_true: Array, x: Array) -> float:
    """Compute ||y_pred - y_true||_2 / ||y_true||_2 using trapezoidal integration."""
    y_pred = np.asarray(y_pred, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    x = np.asarray(x, dtype=float)
    if y_pred.shape != y_true.shape:
        raise ValueError("y_pred and y_true must share the same shape")
    num = integrate.trapezoid((y_pred - y_true) ** 2, x)
    den = integrate.trapezoid(y_true**2, x)
    return float(num / (den + 1e-16))


def mean_absolute_relative_error(y_pred: Array, y_true: Array) -> float:
    y_pred = np.asarray(y_pred, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    return float(np.mean(np.abs(y_pred - y_true) / (np.abs(y_true) + 1e-16)))
