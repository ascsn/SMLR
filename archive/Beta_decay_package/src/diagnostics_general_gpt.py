#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from numpy.polynomial.polynomial import Polynomial

try:
    from smlr.diagnostics import (
        DiagnosticLabels,
        plot_detail_observable,
        plot_detail_spectrum,
        save_figure,
        write_standard_observable_diagnostics,
    )
except ModuleNotFoundError:  # pragma: no cover - source-tree execution before install
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from smlr.diagnostics import (
        DiagnosticLabels,
        plot_detail_observable,
        plot_detail_spectrum,
        save_figure,
        write_standard_observable_diagnostics,
    )

try:
    from . import helper_gpt as helper
except ImportError:  # pragma: no cover - direct script execution
    import archive.Beta_decay_package.src.helper_gpt as helper


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
    p.add_argument("--detail-idx", type=int, default=None)
    p.add_argument("--dpi", type=int, default=200)
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


def choose_detail_idx(n_samples, requested=None):
    if requested is None:
        return n_samples // 2
    if not 0 <= requested < n_samples:
        raise IndexError(f"detail_idx={requested} outside [0, {n_samples})")
    return requested


def predict_em1(params, n, retain, points, coeffs, g_A, nucnam, central_point):
    Lors, HLs_true = helper.data_table(points, coeffs, g_A, nucnam)
    D, S1, S2, v0, fold, x1, x2, x3 = helper.modified_DS(params, n)

    strengths = []
    half_lives = []
    eigs = []
    Bs = []
    widths = []
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
        eigs.append(eigenvalues.numpy())
        Bs.append(B.numpy())
        widths.append(tensor_float(width))

    return {
        "strengths": np.asarray(strengths),
        "half_lives": np.asarray(half_lives),
        "half_lives_true": np.asarray([tensor_float(v) for v in HLs_true]),
        "Lors": Lors,
        "eigs": eigs,
        "Bs": Bs,
        "widths": np.asarray(widths),
    }


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


def save_half_life_outputs(fig_dir, points, pred, true, plots, dpi):
    labels = DiagnosticLabels(
        observable_name="Half-life",
        observable_true="True half-life",
        observable_pred="Predicted half-life",
        relative_error="Half-life relative error",
        parameter_names=("V0_is", "g0"),
        prediction_title="Half-life: prediction vs truth",
        parameter_map_title="Parameter-space half-life error",
    )
    return write_standard_observable_diagnostics(
        fig_dir,
        points,
        true,
        pred,
        labels=labels,
        observable_key="half_life",
        log_scatter=True,
        plots=(plots != "none"),
        csv_alias="half_life_predictions.csv",
        scatter_alias="half_life_true_vs_pred.png",
        dpi=dpi,
    )


def save_em1_detail_spectrum(fig_dir, points, outputs, detail_idx, dpi):
    Lors = outputs["Lors"]
    x = Lors[detail_idx][:, 0]
    y_true = Lors[detail_idx][:, 1]
    y_pred = outputs["strengths"][detail_idx]
    eigs = outputs["eigs"][detail_idx]
    Bs = outputs["Bs"][detail_idx]
    point = points[detail_idx]

    labels = DiagnosticLabels(
        spectrum_x="E (MeV)",
        spectrum_y="Strength (1/MeV)",
        detail_spectrum_title="Detailed spectrum",
    )
    fig, _ = plot_detail_spectrum(
        x,
        y_true,
        y_pred,
        poles=eigs,
        pole_strengths=Bs,
        labels=labels,
        title_suffix=f" (idx={detail_idx}, V0_is={point[0]}, g0={point[1]})",
    )
    save_figure(fig, fig_dir, "detail_spectrum.png", dpi=dpi)
    plt.close(fig)


def save_em2_detail_observable(fig_dir, points, pred, true, detail_idx, dpi):
    point = points[detail_idx]
    rel = abs(pred[detail_idx] - true[detail_idx]) / max(abs(true[detail_idx]), 1e-12)
    labels = DiagnosticLabels(observable_name="Half-life", detail_observable_title="Detailed half-life")
    fig, _ = plot_detail_observable(
        true[detail_idx],
        pred[detail_idx],
        labels=labels,
        title_suffix=f" (idx={detail_idx}, rel={rel:.3e})\nV0_is={point[0]}, g0={point[1]}",
        log_scale=True,
    )
    save_figure(fig, fig_dir, "detail_observable.png", dpi=dpi)
    plt.close(fig)


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
    detail_idx = choose_detail_idx(len(points), args.detail_idx)

    if args.mode == "em1":
        outputs = predict_em1(params, args.n, args.retain, points, coeffs, args.g_A, args.nucnam, central_point)
        pred = outputs["half_lives"]
        true = outputs["half_lives_true"]
        if args.plots == "save":
            save_em1_detail_spectrum(fig_dir, points, outputs, detail_idx, args.dpi)
    else:
        pred, true = predict_em2(params, args.n, points, coeffs, args.g_A, args.nucnam, central_point)
        if args.plots == "save":
            save_em2_detail_observable(fig_dir, points, pred, true, detail_idx, args.dpi)

    summary = save_half_life_outputs(fig_dir, points, pred, true, args.plots, args.dpi)
    print("Diagnostics written to", fig_dir)
    print(summary)


if __name__ == "__main__":
    main()
