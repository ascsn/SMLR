#!/usr/bin/env python3
"""Plot strength spectra from an n-dimensional SMLR dataset.

Expected dataset layout::

    data/<field>/<object>_<dim>/total_strength*/strength_<p1>_..._<pn>.out

Examples::

    python scripts/plot_strengths.py --field nuclear --dataset 48Ca_4d --strength-subdir total_strength_K1
    python scripts/plot_strengths.py --dataset-dir data/nuclear/160Yb_2d --index 3
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


STRENGTH_RE = re.compile(r"^strength_(?P<params>.+)\.out$")


@dataclass(frozen=True)
class StrengthSpectrum:
    path: Path
    params: tuple[float, ...]
    energy: np.ndarray
    strength: np.ndarray

    @property
    def label(self) -> str:
        return "(" + ", ".join(f"{value:.5g}" for value in self.params) + ")"


@dataclass(frozen=True)
class StrengthDataset:
    root: Path
    strength_dir: Path
    spectra: tuple[StrengthSpectrum, ...]
    params_table: np.ndarray | None = None

    @property
    def ndim(self) -> int:
        return len(self.spectra[0].params) if self.spectra else 0

    @property
    def name(self) -> str:
        return self.root.name


def parse_strength_filename(path: Path) -> tuple[float, ...]:
    match = STRENGTH_RE.match(path.name)
    if match is None:
        raise ValueError(f"Not a strength filename: {path.name}")
    pieces = match.group("params").split("_")
    try:
        return tuple(float(piece) for piece in pieces)
    except ValueError as exc:
        raise ValueError(f"Could not parse parameters from {path.name}") from exc


def load_strength_file(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, comments="#")
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"Expected at least two numeric columns in {path}")
    return data[:, 0], data[:, 1]


def resolve_dataset_root(
    *,
    data_root: Path,
    field: str,
    dataset: str,
    dataset_dir: Path | None,
) -> Path:
    return dataset_dir if dataset_dir is not None else data_root / field / dataset


def find_strength_dir(dataset_root: Path, strength_subdir: str | None = None) -> Path:
    if strength_subdir is not None:
        strength_dir = dataset_root / strength_subdir
        if not strength_dir.is_dir():
            raise FileNotFoundError(f"Strength directory does not exist: {strength_dir}")
        return strength_dir

    candidates = sorted(path for path in dataset_root.iterdir() if path.is_dir() and path.name.startswith("total_strength"))
    if not candidates:
        raise FileNotFoundError(f"No total_strength* directory found under {dataset_root}")
    if len(candidates) > 1:
        names = ", ".join(path.name for path in candidates)
        raise ValueError(
            f"Multiple strength directories found under {dataset_root}: {names}. "
            "Pass --strength-subdir to choose one."
        )
    return candidates[0]


def load_params_table(dataset_root: Path) -> np.ndarray | None:
    params_path = dataset_root / "params.txt"
    if not params_path.exists():
        return None
    table = np.loadtxt(params_path, comments="#")
    if table.ndim == 1:
        table = table[None, :]
    return table


def load_strength_dataset(
    *,
    data_root: Path = Path("data"),
    field: str = "nuclear",
    dataset: str | None = None,
    dataset_dir: Path | None = None,
    strength_subdir: str | None = None,
) -> StrengthDataset:
    """Load all strength spectra for an n-dimensional dataset."""

    if dataset is None and dataset_dir is None:
        raise ValueError("Provide either --dataset or --dataset-dir.")
    root = resolve_dataset_root(data_root=data_root, field=field, dataset=dataset or "", dataset_dir=dataset_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")

    strength_dir = find_strength_dir(root, strength_subdir)
    spectra: list[StrengthSpectrum] = []
    for path in sorted(strength_dir.glob("strength_*.out")):
        params = parse_strength_filename(path)
        energy, strength = load_strength_file(path)
        spectra.append(StrengthSpectrum(path=path, params=params, energy=energy, strength=strength))

    if not spectra:
        raise ValueError(f"No strength_*.out files found in {strength_dir}")

    ndim = len(spectra[0].params)
    bad = [s.path.name for s in spectra if len(s.params) != ndim]
    if bad:
        raise ValueError(f"Inconsistent parameter dimension in {strength_dir}; first bad file: {bad[0]}")

    return StrengthDataset(
        root=root,
        strength_dir=strength_dir,
        spectra=tuple(spectra),
        params_table=load_params_table(root),
    )


def _parameter_colors(spectra: Sequence[StrengthSpectrum], color_param: int):
    params = np.asarray([s.params for s in spectra], dtype=float)
    if params.shape[1] == 0:
        return ["#2457a6"] * len(spectra), None, None
    idx = max(0, min(int(color_param), params.shape[1] - 1))
    values = params[:, idx]
    if np.allclose(values, values[0]):
        return ["#2457a6"] * len(spectra), None, None
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(float(values.min()), float(values.max()))
    return [cmap(norm(value)) for value in values], norm, cmap


def _select_spectrum(dataset: StrengthDataset, *, index: int | None = None, params: Sequence[float] | None = None) -> StrengthSpectrum:
    if params is not None:
        target = np.asarray(params, dtype=float)
        for spectrum in dataset.spectra:
            if np.allclose(np.asarray(spectrum.params), target):
                return spectrum
        raise ValueError(f"No spectrum found for params={tuple(target)}")

    idx = 0 if index is None else int(index) - 1
    if idx < 0 or idx >= len(dataset.spectra):
        raise ValueError(f"Index must be between 1 and {len(dataset.spectra)}, got {index}.")
    return dataset.spectra[idx]


def plot_individual_spectrum(
    dataset: StrengthDataset,
    out_dir: Path,
    *,
    index: int | None = None,
    params: Sequence[float] | None = None,
    xlim: tuple[float, float] | None = None,
    title: str | None = None,
) -> Path:
    """Plot one strength spectrum selected by 1-based index or parameter values."""

    spectrum = _select_spectrum(dataset, index=index, params=params)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = "_".join(f"{value:g}" for value in spectrum.params).replace("-", "m").replace(".", "p")
    out = out_dir / f"{dataset.name}_{dataset.strength_dir.name}_strength_{stem}.png"

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(spectrum.energy, spectrum.strength, color="#2457a6", lw=1.8)
    ax.set_title(title or f"{dataset.name} {dataset.strength_dir.name}: p={spectrum.label}")
    ax.set_xlabel("Energy")
    ax.set_ylabel("Strength")
    if xlim is not None:
        ax.set_xlim(*xlim)
    ymax = float(np.nanmax(spectrum.strength))
    ax.set_ylim(0.0, ymax * 1.08 if ymax > 0 else 1.0)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_all_spectra_overlay(
    dataset: StrengthDataset,
    out_dir: Path,
    *,
    color_param: int = 0,
    xlim: tuple[float, float] | None = None,
    alpha: float = 0.55,
    title: str | None = None,
) -> Path:
    """Overlay all strength spectra in a dataset."""

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{dataset.name}_{dataset.strength_dir.name}_strength_overlay.png"
    fig, ax = plt.subplots(figsize=(10, 6))

    colors, norm, cmap = _parameter_colors(dataset.spectra, color_param)
    ymax = 0.0
    for spectrum, color in zip(dataset.spectra, colors):
        ymax = max(ymax, float(np.nanmax(spectrum.strength)))
        ax.plot(spectrum.energy, spectrum.strength, lw=1.0, alpha=alpha, color=color)

    if norm is not None and cmap is not None:
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        cbar = fig.colorbar(sm, ax=ax, pad=0.015)
        cbar.set_label(f"p{color_param + 1}")

    ax.set_title(title or f"{dataset.name} {dataset.strength_dir.name}: {len(dataset.spectra)} spectra")
    ax.set_xlabel("Energy")
    ax.set_ylabel("Strength")
    if xlim is not None:
        ax.set_xlim(*xlim)
    ax.set_ylim(0.0, ymax * 1.08 if ymax > 0 else 1.0)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def parse_params(value: str | None) -> tuple[float, ...] | None:
    if value is None:
        return None
    return tuple(float(piece.strip()) for piece in value.split(",") if piece.strip())


def parse_xlim(values: Sequence[float] | None) -> tuple[float, float] | None:
    if values is None:
        return None
    if len(values) != 2:
        raise ValueError("--xlim requires two values: LOW HIGH")
    return float(values[0]), float(values[1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--field", default="nuclear")
    parser.add_argument("--dataset", default=None, help="Dataset directory name, e.g. 48Ca_4d or 160Yb_2d.")
    parser.add_argument("--dataset-dir", type=Path, default=None, help="Explicit dataset directory path.")
    parser.add_argument("--strength-subdir", default=None, help="Strength subdirectory, e.g. total_strength_K1.")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--mode", choices=["overlay", "individual", "both"], default="overlay")
    parser.add_argument("--index", type=int, default=1, help="1-based spectrum index for individual mode.")
    parser.add_argument("--params", default=None, help="Comma-separated parameter values for individual mode.")
    parser.add_argument("--color-param", type=int, default=1, help="1-based parameter index used for overlay colors.")
    parser.add_argument("--xlim", type=float, nargs=2, default=None)
    args = parser.parse_args()

    dataset = load_strength_dataset(
        data_root=args.data_root,
        field=args.field,
        dataset=args.dataset,
        dataset_dir=args.dataset_dir,
        strength_subdir=args.strength_subdir,
    )
    out_dir = args.out_dir or (dataset.root / "plots")
    xlim = parse_xlim(args.xlim)
    params = parse_params(args.params)

    outputs: list[Path] = []
    if args.mode in {"overlay", "both"}:
        outputs.append(
            plot_all_spectra_overlay(
                dataset,
                out_dir,
                color_param=args.color_param - 1,
                xlim=xlim,
            )
        )
    if args.mode in {"individual", "both"}:
        outputs.append(
            plot_individual_spectrum(
                dataset,
                out_dir,
                index=args.index,
                params=params,
                xlim=xlim,
            )
        )

    print(f"Loaded {len(dataset.spectra)} spectra from {dataset.strength_dir} ({dataset.ndim}D parameters).")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
