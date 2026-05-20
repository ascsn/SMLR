from __future__ import annotations

from smlr.backends.base import BackendCapabilities, BackendUnavailableError


def _jax_numpy():
    try:
        import jax.numpy as jnp
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise BackendUnavailableError("JAX backend requires `jax`. Install with the `jax` extra.") from exc
    return jnp


capabilities = BackendCapabilities(name="jax", autodiff=True, jit=True)


def tensor(value, dtype=None):
    jnp = _jax_numpy()
    return jnp.asarray(value, dtype=dtype)


def eigh(matrix):
    return _jax_numpy().linalg.eigh(matrix)


def matvec(matrix, vector):
    return _jax_numpy().matmul(matrix, vector)


def reduce_sum(value, axis=None):
    return _jax_numpy().sum(value, axis=axis)


def square(value):
    return _jax_numpy().square(value)


def sqrt(value):
    return _jax_numpy().sqrt(value)


def to_numpy(value):
    import numpy as np

    return np.asarray(value)
