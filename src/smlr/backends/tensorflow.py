from __future__ import annotations

from smlr.backends.base import BackendCapabilities, BackendUnavailableError


def _tf():
    try:
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover - depends on optional env
        raise BackendUnavailableError("TensorFlow backend requires `tensorflow`.") from exc
    return tf


capabilities = BackendCapabilities(name="tensorflow", autodiff=True, jit=True)


def tensor(value, dtype=None):
    tf = _tf()
    return tf.convert_to_tensor(value, dtype=dtype)


def eigh(matrix):
    return _tf().linalg.eigh(matrix)


def matvec(matrix, vector):
    return _tf().linalg.matvec(matrix, vector)


def reduce_sum(value, axis=None):
    return _tf().reduce_sum(value, axis=axis)


def square(value):
    return _tf().square(value)


def sqrt(value):
    return _tf().sqrt(value)


def to_numpy(value):
    return value.numpy()
