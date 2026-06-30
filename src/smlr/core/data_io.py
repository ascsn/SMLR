#!/usr/bin/env python3

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import tensorflow as tf

STRENGTH_FILENAME_RE = re.compile(r"^strength_(?P<params>.+)\.out$")


@dataclass(frozen=True)
class DatasetPoint:
    """One strength-function file and its parameter coordinates."""

    params: np.ndarray
    path: Path


@dataclass(frozen=True)
class StrengthDataset:
    """Discovered strength-function files with a stable parameter table."""

    points: tuple[DatasetPoint, ...]
    param_values: np.ndarray

    @property
    def ndim(self) -> int:
        return int(self.param_values.shape[1])

    @property
    def paths(self) -> tuple[Path, ...]:
        return tuple(point.path for point in self.points)


def parse_strength_filename(path: str | Path) -> np.ndarray:
    """Parse ``strength_<p1>_..._<pD>.out`` into a 1D float array."""

    name = Path(path).name
    match = STRENGTH_FILENAME_RE.match(name)
    if match is None:
        raise ValueError(f"Not a strength filename: {name!r}")

    raw_params = match.group("params").split("_")
    if not raw_params:
        raise ValueError(f"No parameter values found in strength filename: {name!r}")

    try:
        values = np.asarray([float(value) for value in raw_params], dtype=float)
    except ValueError as exc:
        raise ValueError(f"Could not parse parameter values from strength filename: {name!r}") from exc

    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"Expected at least one parameter value in strength filename: {name!r}")
    return values


def discover_dataset(data_dir: str | Path, *, pattern: str = "strength_*.out") -> StrengthDataset:
    """Discover strength files under ``data_dir`` in deterministic parameter order."""

    root = Path(data_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Strength data directory does not exist: {root}")

    points = [
        DatasetPoint(params=parse_strength_filename(path), path=path)
        for path in sorted(root.glob(pattern))
        if path.is_file()
    ]
    if not points:
        raise ValueError(f"No strength files matching {pattern!r} found in {root}")

    ndim = points[0].params.size
    bad = [point.path.name for point in points if point.params.size != ndim]
    if bad:
        raise ValueError(
            f"Inconsistent parameter dimension in {root}; expected {ndim}, "
            f"but {bad[0]!r} has {parse_strength_filename(bad[0]).size}."
        )

    points = sorted(points, key=lambda point: tuple(float(value) for value in point.params))
    param_values = np.vstack([point.params for point in points])
    return StrengthDataset(points=tuple(points), param_values=param_values)


def load_strengths(points: StrengthDataset | Iterable[DatasetPoint]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load strength files as ``(omega, strengths, params)``.

    The returned ``strengths`` array has shape ``(n_points, n_grid)`` and
    ``params`` has shape ``(n_points, n_dim)``. All files must contain at least
    two numeric columns and share the same first-column energy grid.
    """

    if isinstance(points, StrengthDataset):
        dataset_points = points.points
    else:
        dataset_points = tuple(points)

    if not dataset_points:
        raise ValueError("Cannot load strengths from an empty dataset.")

    omega_ref: np.ndarray | None = None
    strengths: list[np.ndarray] = []
    params: list[np.ndarray] = []
    
    for point in dataset_points:
        data = np.loadtxt(point.path, comments="#")
        if data.ndim == 1:
            data = data[None, :]
        if data.ndim != 2 or data.shape[1] < 2:
            raise ValueError(f"Expected at least two numeric columns in {point.path}")

        omega = np.asarray(data[:, 0], dtype=float)
        strength = np.asarray(data[:, 1], dtype=float)
        if omega_ref is None:
            omega_ref = omega
        elif omega.shape != omega_ref.shape or not np.allclose(omega, omega_ref, rtol=1e-10, atol=1e-12):
            raise ValueError(f"Energy grid in {point.path} does not match the first strength file.")
        
        strengths.append(strength)
        params.append(np.asarray(point.params, dtype=float))

    assert omega_ref is not None
    return omega_ref, np.vstack(strengths), np.vstack(params)