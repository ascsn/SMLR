from __future__ import annotations

from typing import Optional

import numpy as np
import tensorflow as tf
from scipy.optimize import least_squares, nnls

from .numerics import give_me_lorentzian


def _softplus_np(x, dtype):
    x = np.asarray(x, dtype=dtype)
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def _inv_softplus_np(y, dtype):
    y = np.maximum(np.asarray(y, dtype=dtype), np.asarray(1e-12, dtype=dtype))
    return np.log(np.expm1(y))


def _unpack_poles_and_strengths_tf(z, n, wmin, min_spacing, dtype):
    z = tf.convert_to_tensor(z, dtype=dtype)
    zE, zB = z[:n], z[n:]
    e0 = wmin + tf.nn.softplus(zE[0])
    gaps = tf.nn.softplus(zE[1:]) + min_spacing
    E = tf.concat([e0[None], e0 + tf.cumsum(gaps)], axis=0)
    B = tf.nn.softplus(zB) ** 2
    return E, B


def _pack_poles_and_strengths_np(E0, B0, wmin, min_spacing, dtype, gap_floor):
    zE = np.empty_like(E0, dtype=dtype)
    zE[0] = _inv_softplus_np(E0[0] - wmin, dtype)
    gaps = np.diff(E0)
    zE[1:] = _inv_softplus_np(np.maximum(gaps - min_spacing, gap_floor), dtype)
    zB = _inv_softplus_np(np.sqrt(np.maximum(B0, np.asarray(1e-12, dtype=dtype))), dtype)
    return np.concatenate([zE, zB])


def fit_strength_with_tf_lorentzian(
    omega,
    y,
    n,
    eta,
    grid_M: Optional[int] = None,
    min_spacing: float = 0.2,
    l2: float = 0.0,
    *,
    np_dtype=np.float32,
    tf_dtype=tf.float32,
    gap_floor: float = 1e-12,
):
    """Fit a spectrum as a nonnegative sum of Lorentzians."""
    omega_np = np.asarray(omega, np_dtype)
    y_np = np.asarray(y, np_dtype)
    wmin, wmax = float(omega_np.min()), float(omega_np.max())
    if grid_M is None:
        grid_M = len(omega_np)

    E_grid = np.linspace(wmin + 1e-6, wmax - 1e-6, grid_M, dtype=np_dtype)
    A = 1.0 / ((omega_np[:, None] - E_grid[None, :]) ** 2 + (eta ** 2) / 4.0) * (eta / (2 * np.pi))
    coeff, _ = nnls(A, y_np)
    coeff = coeff.astype(np_dtype)
    idx = np.argsort(coeff)[-int(n):]
    order = np.argsort(E_grid[idx])
    E0 = np.sort(E_grid[idx]).astype(np_dtype)
    B0 = coeff[idx][order].astype(np_dtype)

    for k in range(1, int(n)):
        if E0[k] - E0[k - 1] < min_spacing:
            E0[k] = E0[k - 1] + min_spacing
    z0 = _pack_poles_and_strengths_np(E0, B0, wmin, min_spacing, np_dtype, gap_floor)

    def residuals(z):
        E_tf, B_tf = _unpack_poles_and_strengths_tf(
            z,
            int(n),
            tf.constant(wmin, tf_dtype),
            tf.constant(min_spacing, tf_dtype),
            tf_dtype,
        )
        yhat_tf = give_me_lorentzian(omega_np, E_tf, B_tf, tf.constant(eta, tf_dtype), dtype=tf_dtype)
        r = yhat_tf.numpy() - y_np
        if l2 > 0:
            r = np.concatenate([r, np.sqrt(l2) * np.asarray(z, dtype=np_dtype)])
        return r

    res = least_squares(residuals, z0, method="trf", max_nfev=5000, xtol=1e-10, ftol=1e-10, gtol=1e-10)
    E_tf, B_tf = _unpack_poles_and_strengths_tf(
        res.x,
        int(n),
        tf.constant(wmin, tf_dtype),
        tf.constant(min_spacing, tf_dtype),
        tf_dtype,
    )
    yhat_tf = give_me_lorentzian(omega_np, E_tf, B_tf, tf.constant(eta, tf_dtype), dtype=tf_dtype)
    return E_tf.numpy(), B_tf.numpy(), yhat_tf.numpy()

