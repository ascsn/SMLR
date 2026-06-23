#!/usr/bin/env python3
"""Plot nearest parameter slices from train/validation/test split files."""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_RUN = Path("runs_em1/48Ca_4d_K0_n30_affinew_localcluster_split")
SPLIT_ORDER = ("train", "validation", "test")
COLORS = {"train": "tab:blue", "validation": "tab:orange", "test": "tab:green", "all": "tab:blue"}
MARKERS = {"train": "o", "validation": "^", "test": "s", "all": "o"}


def load_array(path: Path) -> np.ndarray:
    first_data_line = ""
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            first_data_line = stripped
            break
    delimiter = "," if "," in first_data_line else None
    values = np.loadtxt(path, delimiter=delimiter)
    if values.ndim == 1:
        values = values[None, :]
    if path.name == "params.txt" and values.shape[1] > 1:
        first = values[:, 0]
        one_based = np.arange(1, values.shape[0] + 1, dtype=float)
        zero_based = np.arange(values.shape[0], dtype=float)
        if np.allclose(first, one_based) or np.allclose(first, zero_based):
            values = values[:, 1:]
    return values


def load_split_directory(
    run_dir: Path,
    *,
    split_names: tuple[str, ...] = SPLIT_ORDER,
    params_filename: str = "params.txt",
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    arrays = {}
    for name in split_names:
        path = run_dir / f"{name}_set.txt"
        if path.is_file():
            arrays[name] = load_array(path)

    if not arrays:
        params_path = run_dir / params_filename
        if not params_path.is_file():
            raise FileNotFoundError(
                f"{run_dir} must contain split files like train_set.txt or {params_filename}."
            )
        arrays["all"] = load_array(params_path)

    n_cols = {arr.shape[1] for arr in arrays.values()}
    if len(n_cols) != 1:
        raise ValueError(f"All loaded arrays must have the same number of columns, got {sorted(n_cols)}.")

    points = np.vstack([arrays[name] for name in arrays])
    labels = split_labels({name: len(arrays[name]) for name in arrays})
    return arrays, points, labels


def load_dataset(
    source: Path,
    *,
    split_names: tuple[str, ...] = SPLIT_ORDER,
    params_filename: str = "params.txt",
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    if source.is_file():
        arrays = {"all": load_array(source)}
        return arrays, arrays["all"], split_labels({"all": len(arrays["all"])})
    return load_split_directory(source, split_names=split_names, params_filename=params_filename)


def normalized_slice_distance(points: np.ndarray, held_cols: list[int], held_values: np.ndarray) -> np.ndarray:
    if not held_cols:
        return np.zeros(points.shape[0], dtype=float)
    ranges = np.ptp(points[:, held_cols], axis=0)
    ranges = np.where(ranges > 0.0, ranges, 1.0)
    scaled = (points[:, held_cols] - held_values[None, :]) / ranges[None, :]
    return np.linalg.norm(scaled, axis=1)


def split_labels(lengths: dict[str, int]) -> np.ndarray:
    labels = []
    for name, count in lengths.items():
        labels.extend([name] * count)
    return np.asarray(labels)


def default_param_names(n_params: int) -> list[str]:
    return [f"parameter {i + 1}" for i in range(n_params)]


def short_param_names(n_params: int) -> list[str]:
    return [f"p{i + 1}" for i in range(n_params)]


def held_title(held_cols: list[int], held_values: np.ndarray, short_names: list[str]) -> str:
    if not held_cols:
        return "no held parameters"
    return "holding " + ", ".join(
        f"{short_names[col]}≈{value:.4g}" for col, value in zip(held_cols, held_values)
    )


def filename_for_slice(kind: str, varied_cols: list[int], held_cols: list[int], short_names: list[str]) -> str:
    varied = "_".join(short_names[col] for col in varied_cols)
    held = "_".join(short_names[col] for col in held_cols) if held_cols else "none"
    return f"{kind}_nearest_slice_varied_{varied}_held_{held}.png"


def plot_2d_slice(
    points: np.ndarray,
    labels: np.ndarray,
    out_path: Path,
    n_slice: int,
    varied_cols: list[int],
    held_cols: list[int],
    param_names: list[str],
    short_names: list[str],
    title_prefix: str,
) -> None:
    x_col, y_col = varied_cols
    held_values = np.median(points[:, held_cols], axis=0) if held_cols else np.asarray([])
    dist = normalized_slice_distance(points, held_cols, held_values)
    chosen = np.argsort(dist)[:n_slice]

    fig, ax = plt.subplots(figsize=(7, 5.6))
    for name in list(SPLIT_ORDER) + ["all"]:
        mask = labels[chosen] == name
        if not np.any(mask):
            continue
        selected = chosen[mask]
        ax.scatter(
            points[selected, x_col],
            points[selected, y_col],
            s=72,
            c=COLORS[name],
            marker=MARKERS[name],
            edgecolor="black",
            linewidth=0.5,
            label=f"{name} ({len(selected)})",
        )

    ax.set_xlabel(param_names[x_col])
    ax.set_ylabel(param_names[y_col])
    ax.set_title(
        f"{title_prefix}: 2D nearest slice\n"
        f"{held_title(held_cols, held_values, short_names)}"
    )
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_3d_slice(
    points: np.ndarray,
    labels: np.ndarray,
    out_path: Path,
    n_slice: int,
    varied_cols: list[int],
    held_cols: list[int],
    param_names: list[str],
    short_names: list[str],
    title_prefix: str,
) -> None:
    xyz_cols = varied_cols
    held_values = np.median(points[:, held_cols], axis=0) if held_cols else np.asarray([])
    dist = normalized_slice_distance(points, held_cols, held_values)
    chosen = np.argsort(dist)[:n_slice]

    fig = plt.figure(figsize=(7.5, 6.2))
    ax = fig.add_subplot(111, projection="3d")
    for name in list(SPLIT_ORDER) + ["all"]:
        mask = labels[chosen] == name
        if not np.any(mask):
            continue
        selected = chosen[mask]
        ax.scatter(
            points[selected, xyz_cols[0]],
            points[selected, xyz_cols[1]],
            points[selected, xyz_cols[2]],
            s=56,
            c=COLORS[name],
            marker=MARKERS[name],
            edgecolor="black",
            linewidth=0.4,
            depthshade=True,
            label=f"{name} ({len(selected)})",
        )

    ax.set_xlabel(param_names[xyz_cols[0]])
    ax.set_ylabel(param_names[xyz_cols[1]])
    ax.set_zlabel(param_names[xyz_cols[2]])
    ax.set_title(
        f"{title_prefix}: 3D nearest slice\n"
        f"{held_title(held_cols, held_values, short_names)}"
    )
    ax.view_init(elev=24, azim=-52)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def output_all_permutation_plots(
    points: np.ndarray,
    labels: np.ndarray,
    out_dir: Path,
    n_2d: int,
    n_3d: int,
    param_names: list[str] | None = None,
    title_prefix: str = "Parameter split",
) -> list[Path]:
    n_params = points.shape[1]
    param_names = param_names or default_param_names(n_params)
    short_names = short_param_names(n_params)
    out_dir.mkdir(parents=True, exist_ok=True)

    written = []
    all_cols = set(range(n_params))
    if n_params >= 2:
        for varied in combinations(range(n_params), 2):
            varied_cols = list(varied)
            held_cols = sorted(all_cols.difference(varied_cols))
            out_path = out_dir / filename_for_slice("2d", varied_cols, held_cols, short_names)
            plot_2d_slice(
                points,
                labels,
                out_path,
                min(n_2d, len(points)),
                varied_cols,
                held_cols,
                param_names,
                short_names,
                title_prefix,
            )
            written.append(out_path)

    if n_params >= 3:
        for varied in combinations(range(n_params), 3):
            varied_cols = list(varied)
            held_cols = sorted(all_cols.difference(varied_cols))
            out_path = out_dir / filename_for_slice("3d", varied_cols, held_cols, short_names)
            plot_3d_slice(
                points,
                labels,
                out_path,
                min(n_3d, len(points)),
                varied_cols,
                held_cols,
                param_names,
                short_names,
                title_prefix,
            )
            written.append(out_path)

    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        "--run-dir",
        "--dir",
        type=Path,
        default=DEFAULT_RUN,
        help="Directory with *_set.txt split files or params.txt, or a direct parameter file.",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--n-2d", type=int, default=14)
    parser.add_argument("--n-3d", type=int, default=28)
    parser.add_argument("--all-permutations", action="store_true", default=False)
    parser.add_argument("--title-prefix", default=None)
    parser.add_argument(
        "--split-names",
        default=",".join(SPLIT_ORDER),
        help="Comma-separated split prefixes to load as <name>_set.txt.",
    )
    parser.add_argument(
        "--params-filename",
        default="params.txt",
        help="Fallback parameter filename when split files are absent.",
    )
    parser.add_argument(
        "--param-names",
        default=None,
        help="Optional comma-separated parameter labels. Defaults to 'parameter 1', ...",
    )
    args = parser.parse_args()

    split_names = tuple(part.strip() for part in args.split_names.split(",") if part.strip())
    _, points, labels = load_dataset(
        args.source,
        split_names=split_names,
        params_filename=args.params_filename,
    )
    n_params = points.shape[1]
    param_names = (
        [part.strip() for part in args.param_names.split(",")]
        if args.param_names is not None
        else default_param_names(n_params)
    )
    if len(param_names) != n_params:
        raise ValueError(f"Expected {n_params} parameter names, got {len(param_names)}.")
    short_names = short_param_names(n_params)
    title_prefix = args.title_prefix or args.source.stem

    out_dir = args.out_dir or (
        args.source.parent / f"{args.source.stem}_parameter_slice_plots"
        if args.source.is_file()
        else args.source / "parameter_slice_plots"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.all_permutations:
        written = output_all_permutation_plots(
            points,
            labels,
            out_dir,
            args.n_2d,
            args.n_3d,
            param_names=param_names,
            title_prefix=title_prefix,
        )
    else:
        if n_params < 2:
            raise ValueError("Need at least two parameters for a 2D slice.")
        written = []
        varied_2d = [0, 1]
        held_2d = list(range(2, n_params))
        out_2d = out_dir / filename_for_slice("2d", varied_2d, held_2d, short_names)
        plot_2d_slice(
            points,
            labels,
            out_2d,
            min(args.n_2d, len(points)),
            varied_2d,
            held_2d,
            param_names,
            short_names,
            title_prefix,
        )
        written.append(out_2d)

        if n_params >= 3:
            varied_3d = [0, 1, 2]
            held_3d = list(range(3, n_params))
            out_3d = out_dir / filename_for_slice("3d", varied_3d, held_3d, short_names)
            plot_3d_slice(
                points,
                labels,
                out_3d,
                min(args.n_3d, len(points)),
                varied_3d,
                held_3d,
                param_names,
                short_names,
                title_prefix,
            )
            written.append(out_3d)

    print(f"Saved {len(written)} plot(s) in {out_dir}")


if __name__ == "__main__":
    main()
