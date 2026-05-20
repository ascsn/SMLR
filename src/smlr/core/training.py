from __future__ import annotations

import random as rn

import numpy as np
import tensorflow as tf


def set_all_seeds(seed: int):
    rn.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def moving_average(arr, k):
    if len(arr) < k:
        return None
    return np.convolve(arr, np.ones(k) / k, mode="valid")


def make_optimizer(learning_rate: float):
    try:
        return tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
    except ImportError:
        return tf.keras.optimizers.Adam(learning_rate=learning_rate)

