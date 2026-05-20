from __future__ import annotations

from smlr.backends.base import BackendCapabilities, BackendUnavailableError


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise BackendUnavailableError("PyTorch backend requires `torch`. Install with the `torch` extra.") from exc
    return torch


capabilities = BackendCapabilities(name="pytorch", autodiff=True, jit=True)


def tensor(value, dtype=None):
    torch = _torch()
    return torch.as_tensor(value, dtype=dtype)


def eigh(matrix):
    return _torch().linalg.eigh(matrix)


def matvec(matrix, vector):
    return _torch().matvec(matrix, vector)


def reduce_sum(value, axis=None):
    return _torch().sum(value, dim=axis)


def square(value):
    return _torch().square(value)


def sqrt(value):
    return _torch().sqrt(value)


def to_numpy(value):
    return value.detach().cpu().numpy()
