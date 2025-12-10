from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

Array = np.ndarray


@dataclass
class StrengthSample:
    """Single strength function tied to one parameter vector."""

    params: Array
    energy: Array
    strength: Array
    label: Optional[str] = None

    def normalized(self) -> "StrengthSample":
        """Return a copy with strength normalized to unit integral (trapezoidal)."""
        energy = np.asarray(self.energy, dtype=float)
        strength = np.asarray(self.strength, dtype=float)
        area = np.trapz(strength, energy)
        if area <= 0:
            return StrengthSample(self.params.copy(), energy.copy(), strength.copy(), self.label)
        return StrengthSample(self.params.copy(), energy.copy(), strength / area, self.label)


class StrengthDataset:
    """In-memory collection of strength samples.

    Parameters are stored as a 2D array of shape (n_samples, n_params); energy grids may differ
    across samples but will be interpolated when converting to matrices.
    """

    def __init__(self, samples: Sequence[StrengthSample]):
        if not samples:
            raise ValueError("StrengthDataset requires at least one sample")
        self.samples: List[StrengthSample] = list(samples)
        dims = {len(np.atleast_1d(s.params)) for s in self.samples}
        if len(dims) != 1:
            raise ValueError("All samples must share the same parameter dimension")
        self.param_dim = dims.pop()

    def __len__(self) -> int:
        return len(self.samples)

    def parameters(self) -> Array:
        """Return stacked parameter matrix (n_samples, n_params)."""
        return np.vstack([np.asarray(s.params, dtype=float) for s in self.samples])

    def energy_grids(self) -> List[Array]:
        return [np.asarray(s.energy, dtype=float) for s in self.samples]

    def strength_arrays(self) -> List[Array]:
        return [np.asarray(s.strength, dtype=float) for s in self.samples]

    def to_matrix(
        self,
        energy_grid: Optional[Array] = None,
        *,
        normalize: bool = False,
    ) -> Tuple[Array, Array, Array]:
        """Return (params, energy_grid, strengths_matrix).

        If `energy_grid` is provided, strengths are linearly interpolated. If omitted, a common grid
        is inferred if all samples share identical energy points; otherwise a ValueError is raised.
        """

        if energy_grid is None:
            grids = self.energy_grids()
            first = grids[0]
            if not all(np.array_equal(first, g) for g in grids[1:]):
                raise ValueError("Energy grids differ; provide `energy_grid` for interpolation")
            energy_grid = first
        energy_grid = np.asarray(energy_grid, dtype=float)

        strengths = []
        for sample in self.samples:
            y = np.asarray(sample.strength, dtype=float)
            x = np.asarray(sample.energy, dtype=float)
            if not np.array_equal(x, energy_grid):
                y = np.interp(energy_grid, x, y)
            if normalize:
                area = np.trapz(y, energy_grid)
                if area > 0:
                    y = y / area
            strengths.append(y)

        params = self.parameters()
        return params, energy_grid, np.vstack(strengths)

    def train_val_test_split(
        self, train: float = 0.7, val: float = 0.15, *, seed: int = 0
    ) -> Tuple["StrengthDataset", Optional["StrengthDataset"], Optional["StrengthDataset"]]:
        """Split the dataset with reproducible shuffling.

        Returns (train_ds, val_ds_or_None, test_ds_or_None). Validation or test splits that would
        be empty are returned as ``None`` instead of failing construction.
        """
        if not 0 < train < 1 or not 0 <= val < 1 or train + val >= 1:
            raise ValueError("train and val must be in (0,1) and train+val<1")
        rng = np.random.default_rng(seed)
        idx = np.arange(len(self.samples))
        rng.shuffle(idx)
        n_train = max(1, int(len(idx) * train)) if len(idx) > 0 else 0
        n_val = int(len(idx) * val)
        train_idx = idx[:n_train]
        val_idx = idx[n_train : n_train + n_val]
        test_idx = idx[n_train + n_val :]

        train_ds = StrengthDataset([self.samples[i] for i in train_idx])
        val_ds = StrengthDataset([self.samples[i] for i in val_idx]) if len(val_idx) else None
        test_ds = StrengthDataset([self.samples[i] for i in test_idx]) if len(test_idx) else None
        return train_ds, val_ds, test_ds

    @classmethod
    def from_folder(
        cls,
        metadata_csv: Path | str,
        *,
        spectrum_column: str = "spectrum",
        label_column: Optional[str] = None,
        param_columns: Optional[Sequence[str]] = None,
        root: Optional[Path | str] = None,
        sep: str = ",",
    ) -> "StrengthDataset":
        """Load spectra using a metadata CSV.

        Expected CSV columns: parameter columns (auto-detected if `param_columns` is None), a
        `spectrum` column with relative paths to files containing two columns (energy, strength),
        and an optional label column.
        """

        root_path = Path(root) if root is not None else Path(metadata_csv).parent
        df = pd.read_csv(metadata_csv, sep=sep)
        if spectrum_column not in df.columns:
            raise ValueError(f"Missing required column '{spectrum_column}' in metadata")

        if param_columns is None:
            param_columns = [c for c in df.columns if c not in {spectrum_column, label_column}]
        for col in param_columns:
            if col not in df.columns:
                raise ValueError(f"Parameter column '{col}' not found in metadata")

        samples: List[StrengthSample] = []
        for _, row in df.iterrows():
            spectrum_path = root_path / str(row[spectrum_column])
            if not spectrum_path.exists():
                raise FileNotFoundError(f"Spectrum file not found: {spectrum_path}")
            spec_df = pd.read_csv(
                spectrum_path,
                sep=None,
                engine="python",
                names=["energy", "strength"],
                header=None,
                comment="#",
            )
            params = row[param_columns].to_numpy(dtype=float)
            label = str(row[label_column]) if label_column and pd.notna(row[label_column]) else None
            samples.append(
                StrengthSample(
                    params=params,
                    energy=spec_df["energy"].to_numpy(dtype=float),
                    strength=spec_df["strength"].to_numpy(dtype=float),
                    label=label,
                )
            )

        return cls(samples)

    @classmethod
    def from_arrays(
        cls,
        params: Array,
        energy: Array,
        strengths: Array,
        labels: Optional[Sequence[str]] = None,
        *,
        normalize: bool = False,
    ) -> "StrengthDataset":
        """Convenience constructor for already-gridded data."""
        params = np.asarray(params, dtype=float)
        energy = np.asarray(energy, dtype=float)
        strengths = np.asarray(strengths, dtype=float)
        if params.shape[0] != strengths.shape[0]:
            raise ValueError("params and strengths must share leading dimension")
        samples: List[StrengthSample] = []
        for i in range(params.shape[0]):
            y = strengths[i]
            if normalize:
                area = np.trapz(y, energy)
                if area > 0:
                    y = y / area
            label = labels[i] if labels is not None else None
            samples.append(StrengthSample(params[i], energy, y, label))
        return cls(samples)
