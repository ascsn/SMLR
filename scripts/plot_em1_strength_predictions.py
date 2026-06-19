#!/usr/bin/env python3
"""Plot EM1 predicted vs true strength spectra from a saved run."""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "Beta_decay_package" / "src"))
import helper_gpt as helper  # noqa: E402


def load_strength_dataset(data_dir: Path):
    pattern = re.compile(r"strength_(-?[0-9.]+)_(-?[0-9.]+)_(-?[0-9.]+)_(-?[0-9.]+)\.out")
    combined = []
    for fname in sorted(os.listdir(data_dir)):
        match = pattern.match(fname)
        if match:
            params = tuple(float(v) for v in match.groups())
            combined.append((params, str(data_dir / fname)))
    if not combined:
        raise ValueError(f"No strength_*.out files found in {data_dir}")
    return combined


def split_dataset(combined, train_ratio: float = 0.6, cv_ratio: float = 0.0):
    n_total = len(combined)
    n_train = int(n_total * train_ratio)
    n_cv = int(n_total * cv_ratio)
    return {
        "train": combined[:n_train],
        "cv": combined[n_train:n_train + n_cv],
        "test": combined[n_train + n_cv:],
        "all": combined,
    }


def infer_n(num_params: int, num_components: int) -> int:
    for n in range(1, 500):
        expected = 2 * n + num_components * (n * (n + 1) // 2) + num_components + 2
        if expected == num_params:
            return n
    raise ValueError(f"Could not infer n from {num_params} parameters and {num_components} components")


def default_params_file(run_dir: Path) -> Path:
    matches = sorted(run_dir.glob("params_best_n*_retain*.txt"))
    if not matches:
        matches = sorted(run_dir.glob("**/params_n*_retain*_seed*.txt"))
    if not matches:
        raise ValueError(f"No saved parameter file found below {run_dir}")
    return matches[0]


def predict(params, n, num_components, retain, entry, lor_true, central_point, coordinate_scales, fixed_width):
    params_tf = tf.constant(params, dtype=tf.float64)
    D_mod, S_list_mod, v0_mod, eta, width_params = helper.modified_DS_general(
        params_tf, n, num_components
    )
    M_true = helper.linear_matrix(D_mod, S_list_mod, entry, central_point, coordinate_scales)
    eigenvalues, eigenvectors = tf.linalg.eigh(M_true)

    n_i = eigenvalues.shape[0]
    k_keep = int(round(retain * n_i))
    k_keep = max(1, min(k_keep, n_i))
    left = (n_i - k_keep) // 2
    right = left + k_keep
    eigenvalues = eigenvalues[left:right]
    eigenvectors = eigenvectors[:, left:right]

    projections = tf.linalg.matvec(tf.transpose(eigenvectors), v0_mod)
    strengths = tf.square(projections)
    strengths = strengths * tf.cast((eigenvalues > 0) & (eigenvalues < 30), tf.float64)

    x = tf.constant(lor_true[:, 0], dtype=tf.float64)
    if fixed_width is None:
        width = helper.affine_width(eta, width_params, entry, central_point, coordinate_scales)
    else:
        width = tf.constant(float(fixed_width), dtype=tf.float64)
    y_pred = helper.give_me_Lorentzian(x, eigenvalues, strengths, width)
    return x.numpy(), y_pred.numpy(), lor_true[:, 1], eigenvalues.numpy(), strengths.numpy()


def plot_one(out_dir, split_name, row_index, entry, x, y_pred, y_true, poles, strengths):
    out_dir.mkdir(parents=True, exist_ok=True)
    params_label = ", ".join(f"{v:.4g}" for v in helper.dataset_entry_params(entry))
    out = out_dir / f"pred_vs_true_{split_name}_row{row_index}.png"

    fig, ax = plt.subplots(figsize=(9, 5.5))
    markerline, stemlines, baseline = ax.stem(
        poles, strengths, linefmt="0.55", markerfmt="o", basefmt=" "
    )
    plt.setp(markerline, markersize=3, alpha=0.5)
    plt.setp(stemlines, linewidth=0.8, alpha=0.3)
    ax.plot(x, y_pred, label="pred", lw=1.9)
    ax.plot(x, y_true, label="true", lw=1.5)
    ax.set_xlim(float(np.min(x)), float(np.max(x)))
    ax.set_ylim(0)
    ax.set_xlabel("Energy")
    ax.set_ylabel("Strength")
    ax.set_title(f"{split_name} row {row_index}: p=({params_label})")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)

    denom = np.trapezoid(y_true ** 2, x) + 1e-16
    rel_l2 = np.trapezoid((y_pred - y_true) ** 2, x) / denom
    metrics = out.with_suffix(".txt")
    metrics.write_text(
        "\n".join(
            [
                f"split = {split_name}",
                f"row = {row_index}",
                f"params = {params_label}",
                f"true_max = {float(np.max(y_true)):.16e}",
                f"pred_max = {float(np.max(y_pred)):.16e}",
                f"true_integral = {float(np.trapezoid(y_true, x)):.16e}",
                f"pred_integral = {float(np.trapezoid(y_pred, x)):.16e}",
                f"relative_l2 = {float(rel_l2):.16e}",
            ]
        )
        + "\n"
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data" / "gamow_teller_48Ca_4d" / "total_strength_K0")
    parser.add_argument("--params", type=Path, default=None, help="Saved params file. Defaults to params_best in --run-dir.")
    parser.add_argument("--split", choices=["train", "test", "all"], default="train")
    parser.add_argument("--index", type=int, default=1, help="1-based row within --split.")
    parser.add_argument("--all", action="store_true", help="Plot every row in --split.")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--retain", type=float, default=1.0)
    parser.add_argument("--reference-index", type=int, default=0, help="0-based reference index in sorted full dataset.")
    parser.add_argument("--fixed-width", type=float, default=None, help="Use a fixed width for prediction plots. Omit for affine-width runs.")
    parser.add_argument("--no-coordinate-normalization", action="store_true")
    args = parser.parse_args()

    run_dir = args.run_dir
    params_path = args.params or default_params_file(run_dir)
    params = np.loadtxt(params_path)

    combined = load_strength_dataset(args.data_dir)
    splits = split_dataset(combined)
    selected = splits[args.split]
    if not selected:
        raise ValueError(f"Split {args.split!r} is empty")

    num_components = len(helper.dataset_entry_params(combined[0]))
    n = infer_n(len(params), num_components)
    combined_ar = np.array([helper.dataset_entry_params(entry) for entry in combined], dtype=float)
    central_point = helper.dataset_entry_params(combined[max(0, min(args.reference_index, len(combined) - 1))])
    coordinate_scales = None
    if not args.no_coordinate_normalization:
        ranges = np.ptp(combined_ar, axis=0)
        coordinate_scales = tuple(float(v if v > 0 else 1.0) for v in ranges)

    out_dir = args.out_dir or (run_dir / "pred_vs_true")
    rows = range(1, len(selected) + 1) if args.all else [args.index]
    for row in rows:
        if row < 1 or row > len(selected):
            raise ValueError(f"--index must be between 1 and {len(selected)} for split {args.split}, got {row}")
        entry = selected[row - 1]
        lor_true = np.loadtxt(helper.dataset_entry_path(entry))
        x, y_pred, y_true, poles, strengths = predict(
            params=params,
            n=n,
            num_components=num_components,
            retain=args.retain,
            entry=entry,
            lor_true=lor_true,
            central_point=central_point,
            coordinate_scales=coordinate_scales,
            fixed_width=args.fixed_width,
        )
        out = plot_one(out_dir, args.split, row, entry, x, y_pred, y_true, poles, strengths)
        print(out)


if __name__ == "__main__":
    main()
