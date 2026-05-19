#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.colors import LogNorm
from numpy.polynomial.polynomial import Polynomial

try:
    from . import helper_gpt as helper
except ImportError:  # pragma: no cover - direct script execution
    import helper_gpt as helper


def parse_args():
    p = argparse.ArgumentParser(description="Beta-decay package diagnostics for EM1/EM2 runs.")
    p.add_argument("--mode", choices=["em1", "em2"], default="em1")
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--retain", type=float, default=0.9, help="EM1 retained eigenmode fraction.")
    p.add_argument("--save-dir", required=True, help="Run directory containing best parameters and train_set.txt.")
    p.add_argument("--params-file", default=None)
    p.add_argument("--fig-dir", default=None)
    p.add_argument("--nucnam", default="Ni_80")
    p.add_argument("--A", type=int, default=80)
    p.add_argument("--Z", type=int, default=28)
    p.add_argument("--g-A", dest="g_A", type=float, default=1.2)
    p.add_argument("--plots", choices=["save", "none"], default="save")
    return p.parse_args()


def load_train_set(path: Path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(tuple(line.split(",")))
    return rows


def load_all_points(nucnam: str):
    strength_dir = helper.beta_data_dir(nucnam)
    pattern = re.compile(r"lorm_" + re.escape(nucnam) + r"_([0-9.]+)_([0-9.]+)\.out")
    points = []
    for fname in sorted(os.listdir(strength_dir)):
        match = pattern.match(fname)
        if match:
            beta_val = match.group(1)
            alpha_val = match.group(2)
            points.append((alpha_val, beta_val))
    if not points:
        raise RuntimeError(f"No beta-decay strength files found in {strength_dir}")
    return points


def central_point_from(points):
    arr = np.asarray(points, dtype=float)
    center = 0.5 * (arr.min(axis=0) + arr.max(axis=0))
    return tuple(points[int(np.argmin(np.sum((arr - center[None, :]) ** 2, axis=1)))])


def tensor_float(value):
    try:
        return float(value.numpy())
    except AttributeError:
        return float(value)


def predict_em1(params, n, retain, points, coeffs, g_A, nucnam, central_point):
    Lors, HLs_true = helper.data_table(points, coeffs, g_A, nucnam)
    D, S1, S2, v0, fold, x1, x2, x3 = helper.modified_DS(params, n)

    strengths = []
    half_lives = []
    for idx, point in enumerate(points):
        alpha = float(point[0])
        beta = float(point[1])
        M = D + (alpha - float(central_point[0])) * S1 + (beta - float(central_point[1])) * S2
        eigenvalues, eigenvectors = tf.linalg.eigh(M)

        n_i = int(eigenvalues.shape[0])
        k_keep = max(1, min(int(round(retain * n_i)), n_i))
        left = (n_i - k_keep) // 2
        right = left + k_keep
        eigenvalues = eigenvalues[left:right]
        eigenvectors = eigenvectors[:, left:right]

        projections = tf.linalg.matvec(tf.transpose(eigenvectors), v0)
        B = tf.square(projections)
        mask = tf.cast((eigenvalues > -10) & (eigenvalues < 15), dtype=tf.float64)
        B = B * mask

        x = tf.constant(Lors[idx][:, 0], dtype=tf.float64)
        width = tf.sqrt(tf.square(fold) + tf.square(x1 + x2 * alpha + x3 * beta))
        strengths.append(helper.give_me_Lorentzian(x, eigenvalues, B, width).numpy())
        half_lives.append(tensor_float(helper.half_life_loss(eigenvalues, B, coeffs, g_A)))

    return np.asarray(strengths), np.asarray(half_lives), np.asarray([tensor_float(v) for v in HLs_true]), Lors


def predict_em2(params, n, points, coeffs, g_A, nucnam, central_point):
    HLs_true = helper.data_table_only_HL(points, coeffs, g_A, nucnam)
    D, S1, S2 = helper.modified_DS_only_HL(params, n)
    half_lives = []
    for point in points:
        alpha = float(point[0])
        beta = float(point[1])
        M = D + (alpha - float(central_point[0])) * S1 + (beta - float(central_point[1])) * S2
        eigenvalues, _ = tf.linalg.eigh(M)
        half_lives.append(tensor_float(10 ** eigenvalues[int(n / 2)]))
    return np.asarray(half_lives), np.asarray([tensor_float(v) for v in HLs_true])


def save_half_life_outputs(fig_dir, points, pred, true, plots):
    rel = np.abs(pred - true) / np.maximum(np.abs(true), 1e-12)
    with open(fig_dir / "half_life_predictions.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["idx", "alpha", "beta", "true_half_life", "pred_half_life", "relative_error"])
        for idx, (point, y_true, y_pred, err) in enumerate(zip(points, true, pred, rel)):
            writer.writerow([idx, point[0], point[1], y_true, y_pred, err])

    summary = {
        "mean_relative_error": float(np.mean(rel)),
        "median_relative_error": float(np.median(rel)),
        "max_relative_error": float(np.max(rel)),
    }
    with open(fig_dir / "summary.txt", "w") as f:
        for key, value in summary.items():
            f.write(f"{key}={value}\n")

    if plots == "none":
        return summary

    plt.figure(figsize=(5, 4))
    plt.scatter(true, pred)
    lo = min(float(np.min(true)), float(np.min(pred)))
    hi = max(float(np.max(true)), float(np.max(pred)))
    plt.plot([lo, hi], [lo, hi], color="black")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("True half-life")
    plt.ylabel("Predicted half-life")
    plt.savefig(fig_dir / "half_life_true_vs_pred.png", bbox_inches="tight", dpi=180)
    plt.close()

    xy = np.asarray(points, dtype=float)
    plt.figure(figsize=(5, 4))
    plt.scatter(xy[:, 0], xy[:, 1], c=np.clip(rel, 1e-12, None), marker="s", cmap="Spectral", norm=LogNorm())
    plt.colorbar(label="Half-life relative error")
    plt.xlabel("V0_is")
    plt.ylabel("g0")
    plt.savefig(fig_dir / "half_life_error_map.png", bbox_inches="tight", dpi=180)
    plt.close()
    return summary


def main():
    args = parse_args()
    save_dir = Path(args.save_dir)
    fig_dir = Path(args.fig_dir) if args.fig_dir else save_dir / "diagnostics"
    fig_dir.mkdir(parents=True, exist_ok=True)

    params_file = Path(args.params_file) if args.params_file else save_dir / (
        f"params_best_n{args.n}_retain{args.retain}.txt" if args.mode == "em1" else f"params_{args.n}_only_HL.txt"
    )
    params = np.loadtxt(params_file)

    train_set_path = save_dir / "train_set.txt"
    points = load_train_set(train_set_path) if train_set_path.exists() else load_all_points(args.nucnam)
    central_point = central_point_from(points)

    coeffs = Polynomial(helper.fit_phase_space(0, args.Z, args.A, 15)).coef

    if args.mode == "em1":
        _, pred, true, _ = predict_em1(params, args.n, args.retain, points, coeffs, args.g_A, args.nucnam, central_point)
    else:
        pred, true = predict_em2(params, args.n, points, coeffs, args.g_A, args.nucnam, central_point)

    summary = save_half_life_outputs(fig_dir, points, pred, true, args.plots)
    print("Diagnostics written to", fig_dir)
    print(summary)


if __name__ == "__main__":
    main()

