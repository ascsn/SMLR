from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import tensorflow as tf

'''
Reusable ansatz core:
1. parameter vector layout
2. packed-to-structured conversion
3. symmetric matrix reconstruction
4. generic feature maps
5. matrix batch construction
6. vector batch construction
7. width batch construction
8. eigensolver-based prediction
9. deterministic initialization primitives
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


def _n_basis_from_config(config) -> int:
    p = int(config.n_params)
    if config.ansatz == "linear":
        return p
    if config.ansatz == "linear_exp":
        return 2 * p
    if config.ansatz == "quadratic":
        return p + p * (p + 1) // 2
    if config.ansatz == "paper_dipole":
        if p != 2:
            raise ValueError("paper_dipole ansatz requires exactly two parameters: alpha and beta.")
        return 3
    raise ValueError(f"Unknown ansatz {config.ansatz!r}.")


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


def compute_ansatz_features(param_shifts: tf.Tensor, config, feature_params: Optional[tf.Tensor] = None) -> tf.Tensor:
    dx = tf.convert_to_tensor(param_shifts, dtype=tf.float32)
    if dx.shape.rank == 1:
        dx = dx[None, :]
    p = int(config.n_params)

    if config.ansatz == "linear":
        return dx
    if config.ansatz == "linear_exp":
        if feature_params is None:
            feature_params = tf.ones((p,), dtype=tf.float32)
        decay = tf.reshape(feature_params, (1, p))
        return tf.concat([dx, dx * tf.exp(-decay * tf.abs(dx))], axis=1)
    if config.ansatz == "quadratic":
        feats = [dx]
        quad_terms = []
        for i in range(p):
            for j in range(i, p):
                quad_terms.append(dx[:, i] * dx[:, j])
        if quad_terms:
            feats.append(tf.stack(quad_terms, axis=1))
        return tf.concat(feats, axis=1)
    if config.ansatz == "paper_dipole":
        if p != 2:
            raise ValueError("paper_dipole ansatz requires exactly two parameters: alpha and beta.")
        if feature_params is None:
            feature_params = tf.ones((1,), dtype=tf.float32)
        alpha_shift = dx[:, 0]
        beta_shift = dx[:, 1]
        x1 = tf.reshape(feature_params, (-1,))[0]
        return tf.stack([alpha_shift, beta_shift, beta_shift * tf.exp(-alpha_shift * x1)], axis=1)
    raise ValueError(f"Unknown ansatz {config.ansatz!r}.")


def compute_trainable_fwhm(unpacked: Dict[str, tf.Tensor], dx: tf.Tensor, config) -> tf.Tensor:
    if config.width_model == "constant":
        return tf.fill((tf.shape(dx)[0],), tf.abs(unpacked["eta0"]))
    width_affine = unpacked["width_bias"] + tf.einsum("bp,p->b", dx, unpacked["width_linear"])
    return tf.sqrt(tf.square(unpacked["eta0"]) + tf.square(width_affine))


def build_model_matrices_and_vectors(
    params: tf.Tensor,
    config,
    param_values: tf.Tensor,
    central_point: tf.Tensor,
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    unpacked = unpack_trainable_parameters(params, config)
    param_values = tf.cast(param_values, tf.float32)
    dx = param_values - tf.cast(central_point[None, :], tf.float32)
    features = compute_ansatz_features(dx, config, unpacked["feature_params"])
    M_batch = unpacked["D"][None, :, :] + tf.einsum("bf,fij->bij", features, unpacked["basis_mats"])
    v_batch = unpacked["v0"][None, :] + tf.einsum("bp,pn->bn", dx, unpacked["v_linear"])
    fwhm_batch = compute_trainable_fwhm(unpacked, dx, config)
    return M_batch, v_batch, fwhm_batch, features


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

