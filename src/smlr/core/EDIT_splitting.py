from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class ParameterSplit:
    """Boolean train/test masks for a parameter grid."""

    train_mask: np.ndarray
    test_mask: np.ndarray

    @property
    def train_indices(self) -> np.ndarray:
        return np.flatnonzero(self.train_mask)

    @property
    def test_indices(self) -> np.ndarray:
        return np.flatnonzero(self.test_mask)


def parse_filter_ranges(value: str | Mapping[str, Sequence[float]] | None) -> dict[str, tuple[float, float]]:
    """Parse parameter-box filters of the form ``{"p1": [lo, hi]}``."""

    if value is None:
        return {}
    parsed = json.loads(value) if isinstance(value, str) else dict(value)
    out: dict[str, tuple[float, float]] = {}
    for key, bounds in parsed.items():
        if len(bounds) != 2:
            raise ValueError(f"Filter for {key!r} must have two entries [min, max].")
        lo, hi = float(bounds[0]), float(bounds[1])
        if lo > hi:
            raise ValueError(f"Filter for {key!r} must satisfy min <= max, got [{lo}, {hi}].")
        out[str(key)] = (lo, hi)
    return out


def split_by_parameter_ranges(
    param_values,
    param_names: Sequence[str],
    filter_ranges: str | Mapping[str, Sequence[float]] | None,
) -> ParameterSplit:
    """Split a parameter grid into train/test masks using inclusive ranges.

    Points inside all supplied parameter ranges are training points. Points
    outside at least one range are test points. With no ranges, all points are
    assigned to training and the test mask is empty.
    """

    values = np.asarray(param_values, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"param_values must be a 2D array, got shape {values.shape}.")
    if values.shape[1] != len(param_names):
        raise ValueError(
            f"param_values has {values.shape[1]} columns but param_names has {len(param_names)} entries."
        )

    ranges = parse_filter_ranges(filter_ranges)
    train_mask = np.ones(values.shape[0], dtype=bool)
    if not ranges:
        return ParameterSplit(train_mask=train_mask, test_mask=np.zeros(values.shape[0], dtype=bool))

    name_to_idx = {name: idx for idx, name in enumerate(param_names)}
    for name, (lo, hi) in ranges.items():
        if name not in name_to_idx:
            raise ValueError(f"Unknown filter key {name!r}. Available names: {list(param_names)}")
        col = name_to_idx[name]
        train_mask &= (values[:, col] >= lo) & (values[:, col] <= hi)
    return ParameterSplit(train_mask=train_mask, test_mask=~train_mask)


def choose_central_parameter_point(param_values):
    """Choose the sampled point nearest the center of the parameter bounding box."""

    values = np.asarray(param_values, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("param_values must be a non-empty 2D array.")
    center = 0.5 * (np.min(values, axis=0) + np.max(values, axis=0))
    idx = int(np.argmin(np.sum((values - center[None, :]) ** 2, axis=1)))
    return values[idx].copy(), idx
