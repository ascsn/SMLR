from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import tensorflow as tf


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