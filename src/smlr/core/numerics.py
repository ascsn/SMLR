from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import tensorflow as tf

from .retention import RetainedModePolicy, centered_keep_indices

def _infer_float_dtype(*values, default=tf.float32):
    for value in values:
        dtype = getattr(value, "dtype", None)
        if dtype is None:
            continue
        try:
            dtype = tf.as_dtype(dtype)
        except TypeError:
            continue
        if dtype.is_floating:
            return dtype
    return default


@tf.function
def give_me_lorentzian(energy, poles, strength, width, dtype: Optional[tf.DType] = None):
    """Evaluate a sum of Lorentzian strength functions."""
    dtype = dtype or _infer_float_dtype(energy, poles, strength, width)
    energy = tf.convert_to_tensor(energy, dtype=dtype)
    poles = tf.convert_to_tensor(poles, dtype=dtype)
    strength = tf.convert_to_tensor(strength, dtype=dtype)
    width = tf.convert_to_tensor(width, dtype=dtype)

    energy_expanded = tf.expand_dims(energy, axis=-1)
    numerator = strength * (width / tf.cast(2.0 * np.pi, dtype))
    denominator = tf.square(energy_expanded - poles) + tf.square(width) / tf.cast(4.0, dtype)
    return tf.reduce_sum(numerator / denominator, axis=-1)


@tf.function
def give_me_lorentzian_batched(omega, poles_batch, strengths_batch, half_width_batch, dtype: Optional[tf.DType] = None):
    """Evaluate batched Lorentzian sums using half widths."""
    dtype = dtype or _infer_float_dtype(omega, poles_batch, strengths_batch, half_width_batch)
    omega = tf.convert_to_tensor(omega, dtype=dtype)
    poles_batch = tf.convert_to_tensor(poles_batch, dtype=dtype)
    strengths_batch = tf.convert_to_tensor(strengths_batch, dtype=dtype)
    half_width_batch = tf.reshape(tf.convert_to_tensor(half_width_batch, dtype=dtype), (-1, 1, 1))

    omega_exp = tf.expand_dims(omega, axis=1)
    omega_exp = tf.tile(omega_exp, [tf.shape(poles_batch)[0], 1, 1])
    poles_exp = tf.expand_dims(poles_batch, axis=-1)
    strengths_exp = tf.expand_dims(strengths_batch, axis=-1)

    numerator = strengths_exp * half_width_batch / tf.cast(np.pi, dtype)
    denominator = tf.square(omega_exp - poles_exp) + tf.square(half_width_batch)
    return tf.reduce_sum(numerator / denominator, axis=1)


def centered_spectrum_initialization(E, B, n: int, retain: float, *, dtype=np.float32, step: float = 2.0):
    """Build full diagonal and v0 arrays from a centered retained spectrum."""
    left, right, k_keep = centered_keep_indices(n, retain)
    E = np.asarray(E, dtype=dtype).reshape(-1)
    B = np.asarray(B, dtype=dtype).reshape(-1)
    order = np.argsort(E)
    E, B = E[order], B[order]
    if len(E) < k_keep:
        raise ValueError(f"E,B need at least k_keep={k_keep} entries (got {len(E)}).")

    start = (len(E) - k_keep) // 2
    E_sel = E[start:start + k_keep]
    B_sel = B[start:start + k_keep]

    D_full = np.empty(int(n), dtype=dtype)
    D_full[left:right] = E_sel
    cur = E_sel[0]
    for i in range(left - 1, -1, -1):
        cur -= step
        D_full[i] = cur
    cur = E_sel[-1]
    for i in range(right, int(n)):
        cur += step
        D_full[i] = cur

    v0_full = np.zeros(int(n), dtype=dtype)
    v0_full[left:right] = np.sqrt(np.maximum(B_sel, 0.0)).astype(dtype)
    return D_full, v0_full, (left, right, k_keep)
