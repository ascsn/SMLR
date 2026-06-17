#!/usr/bin/env python3
"""Plot Yukiya 48Ca 4D Gamow-Teller K0/K1 strength functions."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_params(root: Path) -> np.ndarray:
    params = np.loadtxt(root / "params.txt")
    if params.ndim == 1:
        params = params[None, :]
    return params


def load_sample(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"Expected at least two columns in {path}")
    return data[:, 0], data[:, 1]


def sample_path(root: Path, k_label: str, index: int) -> Path:
    return root / k_label / f"sample_{index}.dat"


def aligned_k0_k1(root: Path, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x0, y0 = load_sample(sample_path(root, "K0", index))
    x1, y1 = load_sample(sample_path(root, "K1", index))
    if len(x0) == len(x1) and np.allclose(x0, x1):
        return x0, y0, y1
    return x0, y0, np.interp(x0, x1, y1)


def plot_k0_overlay(root: Path, out_dir: Path, params: np.ndarray) -> Path:
    out = out_dir / "yukiya_48ca_4d_gt_k0_strength_overlay.png"
    fig, ax = plt.subplots(figsize=(10, 6))

    norm = plt.Normalize(params[:, 0].min(), params[:, 0].max())
    cmap = plt.get_cmap("viridis")

    ymax = 0.0
    for i, point in enumerate(params, start=1):
        x, y = load_sample(sample_path(root, "K0", i))
        ymax = max(ymax, float(np.nanmax(y)))
        ax.plot(x, y, lw=1.0, alpha=0.58, color=cmap(norm(point[0])))

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax, pad=0.015)
    cbar.set_label("p1")
    ax.set_title("Yukiya 48Ca 4D GT K0 strength functions")
    ax.set_xlabel("Energy")
    ax.set_ylabel("K0 strength")
    ax.set_xlim(0, 30)
    ax.set_ylim(0, ymax * 1.08 if ymax > 0 else 1)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_k1_overlay(root: Path, out_dir: Path, params: np.ndarray) -> Path:
    out = out_dir / "yukiya_48ca_4d_gt_k1_strength_overlay.png"
    fig, ax = plt.subplots(figsize=(10, 6))

    norm = plt.Normalize(params[:, 0].min(), params[:, 0].max())
    cmap = plt.get_cmap("viridis")

    ymax = 0.0
    for i, point in enumerate(params, start=1):
        x, y = load_sample(sample_path(root, "K1", i))
        ymax = max(ymax, float(np.nanmax(y)))
        ax.plot(x, y, lw=1.0, alpha=0.58, color=cmap(norm(point[0])))

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax, pad=0.015)
    cbar.set_label("p1")
    ax.set_title("Yukiya 48Ca 4D GT K1 strength functions")
    ax.set_xlabel("Energy")
    ax.set_ylabel("K1 strength")
    ax.set_xlim(0, 30)
    ax.set_ylim(0, ymax * 1.08 if ymax > 0 else 1)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_k0_k1_overlay(root: Path, out_dir: Path, params: np.ndarray) -> Path:
    out = out_dir / "yukiya_48ca_4d_gt_k0_k1_strength_overlay.png"
    fig, ax = plt.subplots(figsize=(10, 6))

    norm = plt.Normalize(params[:, 0].min(), params[:, 0].max())
    cmap = plt.get_cmap("viridis")

    ymax = 0.0
    for i, point in enumerate(params, start=1):
        color = cmap(norm(point[0]))
        x0, y0, y1 = aligned_k0_k1(root, i)
        ymax = max(ymax, float(np.nanmax(y0)), float(np.nanmax(y1)))
        ax.plot(x0, y0, lw=1.0, alpha=0.44, color=color, ls="-")
        ax.plot(x0, y1, lw=1.0, alpha=0.64, color=color, ls="--")

    ax.plot([], [], color="black", ls="-", label="K0")
    ax.plot([], [], color="black", ls="--", label="K1")
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax, pad=0.015)
    cbar.set_label("p1")
    ax.set_title("Yukiya 48Ca 4D GT K0 and K1 strength functions")
    ax.set_xlabel("Energy")
    ax.set_ylabel("Strength")
    ax.set_xlim(0, 30)
    ax.set_ylim(0, ymax * 1.08 if ymax > 0 else 1)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out

def plot_k0_k1_difference(root: Path, out_dir: Path, params: np.ndarray) -> Path:
    out = out_dir / "yukiya_48ca_4d_gt_k0_minus_k1_difference.png"
    fig, ax = plt.subplots(figsize=(10, 6))

    norm = plt.Normalize(params[:, 0].min(), params[:, 0].max())
    cmap = plt.get_cmap("coolwarm")

    max_abs = 0.0
    mean_abs_by_sample = []
    max_abs_by_sample = []
    for i, point in enumerate(params, start=1):
        x, y0, y1 = aligned_k0_k1(root, i)
        diff = y0 - y1
        max_abs = max(max_abs, float(np.nanmax(np.abs(diff))))
        mean_abs_by_sample.append(float(np.nanmean(np.abs(diff))))
        max_abs_by_sample.append(float(np.nanmax(np.abs(diff))))
        ax.plot(x, diff, lw=1.0, alpha=0.58, color=cmap(norm(point[0])))

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax, pad=0.015)
    cbar.set_label("p1")
    ax.axhline(0.0, color="black", lw=0.8, alpha=0.65)
    ax.set_title("Yukiya 48Ca 4D GT difference: K0 - K1")
    ax.set_xlabel("Energy")
    ax.set_ylabel("K0 - K1")
    ax.set_xlim(0, 30)
    ylim = max_abs * 1.08 if max_abs > 0 else 1.0
    ax.set_ylim(-ylim, ylim)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)

    summary = out_dir / "yukiya_48ca_4d_gt_k0_k1_difference_summary.txt"
    summary.write_text(
        "\n".join(
            [
                f"num_samples = {len(params)}",
                f"global_max_abs_K0_minus_K1 = {max_abs:.16e}",
                f"mean_sample_mean_abs_K0_minus_K1 = {np.mean(mean_abs_by_sample):.16e}",
                f"max_sample_mean_abs_K0_minus_K1 = {np.max(mean_abs_by_sample):.16e}",
                f"max_sample_max_abs_K0_minus_K1 = {np.max(max_abs_by_sample):.16e}",
            ]
        )
        + "\n"
    )
    return out


def plot_total_strength_overlay(root: Path, out_dir: Path, params: np.ndarray) -> Path:
    out = out_dir / "yukiya_48ca_4d_gt_total_strength_overlay.png"
    fig, ax = plt.subplots(figsize=(10, 6))

    norm = plt.Normalize(params[:, 0].min(), params[:, 0].max())
    cmap = plt.get_cmap("viridis")

    ymax = 0.0
    for i, point in enumerate(params, start=1):
        x, y0, y1 = aligned_k0_k1(root, i)
        total = y0 + 2.0 * y1
        ymax = max(ymax, float(np.nanmax(total)))
        ax.plot(x, total, lw=1.0, alpha=0.58, color=cmap(norm(point[0])))

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax, pad=0.015)
    cbar.set_label("p1")
    ax.set_title("Yukiya 48Ca 4D GT total strength: K0 + 2*K1")
    ax.set_xlabel("Energy")
    ax.set_ylabel("S_total")
    ax.set_xlim(0, 30)
    ax.set_ylim(0, ymax * 1.08 if ymax > 0 else 1)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing params.txt plus K0/ and K1/ sample files.",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    root = args.data_dir
    out_dir = args.out_dir or (root / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    params = load_params(root)
    outputs = [
        plot_k0_overlay(root, out_dir, params),
        plot_k1_overlay(root, out_dir, params),
        plot_k0_k1_overlay(root, out_dir, params),
        plot_k0_k1_difference(root, out_dir, params),
        plot_total_strength_overlay(root, out_dir, params),
    ]
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
