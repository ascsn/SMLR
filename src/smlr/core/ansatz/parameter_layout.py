from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import tensorflow as tf

'''
Contents: PackedLayout, layout construction, unpacking, sym_from_upper.

Current layout is too monolithic.
It assumes one packed vector format for all emulators.
Beta EM1 already uses a different layout, so either the
layout API must support multiple named layouts,
or paper beta unpacking should remain domain-specific until generalized.

Also, consider renaming as initialization.py?
Or consider a separate file for initialization.
(random vector initialization,
encoding a central fitted spectrum into D and v0,
deterministic seed-aware initialization helpers)

config is duck-typed.
class MatrixModelConfig?

dtype is hard-coded around tf.float32. Beta code often uses float64.
Either dtype should be config-driven,
or domain trainers should cast consistently before entering core.

'''

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
    '''
    Need to remove domain-specific logic from this function,
    e.g. the ansatz types and width model types.
    Instead, we should just have a function that takes in the necessary parameters
    (n, p, use_vector_terms, width_model, ansatz) and returns the layout.
    '''

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


def unpack_trainable_parameters(params: tf.Tensor, config) -> Dict[str, tf.Tensor]:    
    params = tf.convert_to_tensor(params, dtype=tf.float32)
    layout = get_packed_layout(config)
    n, p = int(config.n), int(config.n_params)

    eta0 = params[0]
    v0 = params[layout.v0_slice]
    if config.use_vector_terms and layout.v_linear_slice.stop > layout.v_linear_slice.start:
        v_linear = tf.reshape(params[layout.v_linear_slice], (p, n))
    else:
        v_linear = tf.zeros((p, n), dtype=tf.float32)

    d_diag = params[layout.d_diag_slice]
    D = tf.linalg.diag(d_diag)
    basis_flat = params[layout.basis_slice]
    basis_mats = tf.reshape(basis_flat, (layout.n_basis, layout.n_upper))
    basis_mats = tf.map_fn(lambda x: sym_from_upper(x, n, dtype=tf.float32), basis_mats, fn_output_signature=tf.float32)

    width_bias = params[layout.width_bias_slice][0]
    if config.width_model == "affine" and layout.width_linear_slice.stop > layout.width_linear_slice.start:
        width_linear = params[layout.width_linear_slice]
    else:
        width_linear = tf.zeros((p,), dtype=tf.float32)

    if config.ansatz == "linear_exp":
        feature_params = tf.nn.softplus(params[layout.feature_param_slice])
    elif config.ansatz == "paper_dipole":
        feature_params = params[layout.feature_param_slice]
    else:
        feature_params = tf.zeros((0,), dtype=tf.float32)

    return {
        "eta0": eta0,
        "v0": v0,
        "v_linear": v_linear,
        "D": D,
        "d_diag": d_diag,
        "basis_mats": basis_mats,
        "width_bias": width_bias,
        "width_linear": width_linear,
        "feature_params": feature_params,
        "layout": layout,
    }


def make_random_initial_guess(config, seed: Optional[int] = None, fold: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    layout = get_packed_layout(config)
    vec = rng.normal(scale=0.05, size=layout.total_size).astype(np.float32)
    vec[0] = np.float32(fold)
    vec[layout.width_bias_slice] = np.array([0.0], dtype=np.float32)
    if config.ansatz == "linear_exp":
        vec[layout.feature_param_slice] = np.full(
            layout.feature_param_slice.stop - layout.feature_param_slice.start,
            0.2,
            dtype=np.float32,
        )
    elif config.ansatz == "paper_dipole":
        vec[layout.feature_param_slice] = np.array([0.2], dtype=np.float32)
    return vec