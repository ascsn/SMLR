from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import tensorflow as tf


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


def compute_trainable_fwhm(unpacked: Dict[str, tf.Tensor], dx: tf.Tensor, config) -> tf.Tensor:
    if config.width_model == "constant":
        return tf.fill((tf.shape(dx)[0],), tf.abs(unpacked["eta0"]))
    width_affine = unpacked["width_bias"] + tf.einsum("bp,p->b", dx, unpacked["width_linear"])
    return tf.sqrt(tf.square(unpacked["eta0"]) + tf.square(width_affine))