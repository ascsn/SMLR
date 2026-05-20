from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


def centered_keep_indices(n: int, retain: float) -> Tuple[int, int, int]:
    """Return (left, right, k_keep) for centered mode retention."""

    n = int(n)
    k_keep = int(round(float(retain) * n))
    k_keep = max(1, min(k_keep, n))
    left = (n - k_keep) // 2
    right = left + k_keep
    return left, right, k_keep


@dataclass(frozen=True)
class RetainedModePolicy:
    """Policy for selecting emulator eigenmodes after diagonalization."""

    kind: str = "centered"
    retain: float = 1.0

    def indices(self, n: int) -> Tuple[int, int, int]:
        n = int(n)
        if self.kind == "all":
            return 0, n, n
        if self.kind != "centered":
            raise ValueError(f"Unknown retained-mode policy kind {self.kind!r}.")
        return centered_keep_indices(n, self.retain)

    def apply(self, values, axis: int = -1):
        left, right, _ = self.indices(np.shape(values)[axis])
        slicer = [slice(None)] * np.ndim(values)
        slicer[axis] = slice(left, right)
        return values[tuple(slicer)]

    def to_dict(self) -> dict[str, float | str]:
        return {"kind": self.kind, "retain": float(self.retain)}

    @classmethod
    def from_dict(cls, data):
        return cls(kind=data.get("kind", "centered"), retain=float(data.get("retain", 1.0)))
