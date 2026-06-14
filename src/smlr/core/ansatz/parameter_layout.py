from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import tensorflow as tf


@dataclass
class PackedLayout:
    eta_size: int
    v0_slice: slice
    v_linear_slice: slice
    d_diag_slice: slice
    basis_slice: slice
    width_bias_slice: slice
    width_linear_slice: slice
    feature_param_slice: slice
    n_upper: int
    n_basis: int
    total_size: int


def get_packed_layout(config) -> PackedLayout:
    n = int(config.n)
    p = int(config.n_params)
    n_upper = n * (n + 1) // 2
    n_basis = _n_basis_from_config(config)

    idx = 0
    eta_size = 1
    idx += eta_size
    v0_slice = slice(idx, idx + n)
    idx += n
    v_linear_size = p * n if config.use_vector_terms else 0
    v_linear_slice = slice(idx, idx + v_linear_size)
    idx += v_linear_size
    d_diag_slice = slice(idx, idx + n)
    idx += n
    basis_slice = slice(idx, idx + n_basis * n_upper)
    idx += n_basis * n_upper
    width_bias_slice = slice(idx, idx + 1)
    idx += 1
    width_linear_size = p if config.width_model == "affine" else 0
    width_linear_slice = slice(idx, idx + width_linear_size)
    idx += width_linear_size
    feature_param_size = p if config.ansatz == "linear_exp" else 0
    if config.ansatz == "paper_dipole":
        feature_param_size = 1
    feature_param_slice = slice(idx, idx + feature_param_size)
    idx += feature_param_size

    return PackedLayout(
        eta_size=eta_size,
        v0_slice=v0_slice,
        v_linear_slice=v_linear_slice,
        d_diag_slice=d_diag_slice,
        basis_slice=basis_slice,
        width_bias_slice=width_bias_slice,
        width_linear_slice=width_linear_slice,
        feature_param_slice=feature_param_slice,
        n_upper=n_upper,
        n_basis=n_basis,
        total_size=idx,
    )


def sym_from_upper(flat_upper: tf.Tensor, n: int, dtype=tf.float32) -> tf.Tensor:
    flat_upper = tf.cast(flat_upper, dtype)
    upper_idx = np.triu_indices(int(n))
    indices = tf.constant(np.column_stack(upper_idx), dtype=tf.int32)
    mat = tf.tensor_scatter_nd_update(tf.zeros((int(n), int(n)), dtype=dtype), indices, flat_upper)
    return mat + tf.transpose(mat) - tf.linalg.diag(tf.linalg.diag_part(mat))