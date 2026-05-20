"""Backend adapters for optimizer and tensor-library experiments."""

from .base import BackendCapabilities, BackendUnavailableError

__all__ = ["BackendCapabilities", "BackendUnavailableError"]
