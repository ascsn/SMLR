#!/usr/bin/env python3
"""Plot 48Ca 4D train/validation/test parameter slices."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_RUN = Path("runs_em1/gamow_teller_48Ca_4d_K0_n30_affinew_localcluster_split")


def load_split(run_dir: Path, name: str) -> np.ndarray:
    path = run_dir / f"{name}_set.txt"
    values = np.loadtxt(path, delimiter=",")
    if values.ndim == 1:
        values = values[None, :]
    return values


def normalized_slice_distance(points: np.ndarray, held_cols: list[int], held_values: np.ndarray) -> np.ndarray:
    ranges = np.ptp(points[:, held_cols], axis=0)
    ranges = np.where(ranges > 0.0, ranges, 1.0)
    scaled = (points[:, held_cols] - held_values[None, :]) / ranges[None, :]
    return np.linalg.norm(scaled, axis=1)


def split_labels(lengths: dict[str, int]) -> np.ndarray:
    labels = []
    for name, count in lengths.items():
        labels.extend([name] * count)
    return np.asarray(labels)


def plot_2d_slice(points: np.ndarray, labels: np.ndarray, out_path: Path, n_slice: int) -> None:
    x_col, y_col = 0, 1
    held_cols = [2, 3]
    held_values = np.median(points[:, held_cols], axis=0)
    dist = normalized_slice_distance(points, held_cols, held_values)
    chosen = np.argsort(dist)[:n_slice]

    colors = {"train": "tab:blue", "validation": "tab:orange", "test": "tab:green"}
    markers = {"train": "o", "validation": "^", "test": "s"}

    fig, ax = plt.subplots(figsize=(7, 5.6))
    for name in ["train", "validation", "test"]:
        mask = labels[chosen] == name
        if not np.any(mask):
            continue
        selected = chosen[mask]
        ax.scatter(
            points[selected, x_col],
            points[selected, y_col],
            s=72,
            c=colors[name],
            marker=markers[name],
            edgecolor="black",
            linewidth=0.5,
            label=f"{name} ({len(selected)})",
        )

    ax.set_xlabel("parameter 1")
    ax.set_ylabel("parameter 2")
    ax.set_title(
        "48Ca 4D local-cluster split: 2D nearest slice\n"
        f"holding p3≈{held_values[0]:.4g}, p4≈{held_values[1]:.4g}"
    )
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_3d_slice(points: np.ndarray, labels: np.ndarray, out_path: Path, n_slice: int) -> None:
    xyz_cols = [0, 1, 2]
    held_cols = [3]
    held_values = np.median(points[:, held_cols], axis=0)
    dist = normalized_slice_distance(points, held_cols, held_values)
    chosen = np.argsort(dist)[:n_slice]

    colors = {"train": "tab:blue", "validation": "tab:orange", "test": "tab:green"}
    markers = {"train": "o", "validation": "^", "test": "s"}

    fig = plt.figure(figsize=(7.5, 6.2))
    ax = fig.add_subplot(111, projection="3d")
    for name in ["train", "validation", "test"]:
        mask = labels[chosen] == name
        if not np.any(mask):
            continue
        selected = chosen[mask]
        ax.scatter(
            points[selected, xyz_cols[0]],
            points[selected, xyz_cols[1]],
            points[selected, xyz_cols[2]],
            s=56,
            c=colors[name],
            marker=markers[name],
            edgecolor="black",
            linewidth=0.4,
            depthshade=True,
            label=f"{name} ({len(selected)})",
        )

    ax.set_xlabel("parameter 1")
    ax.set_ylabel("parameter 2")
    ax.set_zlabel("parameter 3")
    ax.set_title(
        "48Ca 4D local-cluster split: 3D nearest slice\n"
        f"holding p4≈{held_values[0]:.4g}"
    )
    ax.view_init(elev=24, azim=-52)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--n-2d", type=int, default=14)
    parser.add_argument("--n-3d", type=int, default=28)
    args = parser.parse_args()

    arrays = {name: load_split(args.run_dir, name) for name in ["train", "validation", "test"]}
    points = np.vstack([arrays["train"], arrays["validation"], arrays["test"]])
    labels = split_labels({name: len(arrays[name]) for name in ["train", "validation", "test"]})

    out_dir = args.out_dir or (args.run_dir / "parameter_slice_plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_2d_slice(points, labels, out_dir / "48Ca_4d_2d_nearest_slice_p3_p4_fixed.png", args.n_2d)
    plot_3d_slice(points, labels, out_dir / "48Ca_4d_3d_nearest_slice_p4_fixed.png", args.n_3d)
    print(f"Saved plots in {out_dir}")


if __name__ == "__main__":
    main()
