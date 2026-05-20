from __future__ import annotations

from dataclasses import dataclass


class BackendUnavailableError(ImportError):
    """Raised when an optional backend dependency is not installed."""


@dataclass(frozen=True)
class BackendCapabilities:
    name: str
    autodiff: bool
    jit: bool
    gpu: bool | None = None
