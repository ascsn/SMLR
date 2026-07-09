#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.colors import LogNorm

import helper_gpt as helper_gpt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Emulator 2 (alphaD-only) with the current helper_gpt workflow."
    )
    parser.add_argument("--run-dir", default="runs_em2")
    parser.add_argument("--strength-dir", default="data/nuclear/160Yb_2d/total_strength")
    parser.add_argument("--alphaD-dir", default="data/nuclear/160Yb_2d/total_alphaD")
    parser.add_argument(
        "--strength-regex",
        default=r"strength_(?P<p2>[0-9.]+)_(?P<p1>[0-9.]+)\.out",
    )
    parser.add_argument("--alphaD-regex", default=None)
    parser.add_argument("--filter-ranges", default='{"p1":[0.4,1.8],"p2":[1.5,4.0]}')
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--ansatz", default="linear_exp")
    parser.add_argument(
        "--alphaD-mode",
        choices=["mid_eigenvalue", "sum_inverse_positive"],
        default="mid_eigenvalue",
    )
    parser.add_argument("--plots", choices=["none", "save", "show"], default="save")
    parser.add_argument("--fig-dir", default=None)
    parser.add_argument("--max-label-points", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    params = np.loadtxt(run_dir / "best_params_global.txt").astype(np.float32)

    dataset = helper_gpt.load_generic_alphaD_dataset(
        strength_dir=args.strength_dir,
        alphaD_dir=args.alphaD_dir,
        strength_regex=args.strength_regex,
        alphaD_regex=args.alphaD_regex,
        filter_ranges=args.filter_ranges,
        central_point=None,
    )
    config = helper_gpt.AlphaDOnlyConfig(
        n=args.n,
        n_params=int(dataset.param_values.shape[1]),
        ansatz=args.ansatz,
        alphaD_mode=args.alphaD_mode,
    )
    expected = helper_gpt.count_alphaD_only_parameters(config)
    if params.shape != (expected,):
        raise ValueError(f"Expected params shape {(expected,)}, got {params.shape}")

    loss_fn = helper_gpt.make_alphaD_only_loss_fn_generic(
        n=args.n,
        param_values=dataset.param_values,
        alphaD_true=dataset.alphaD_values,
        central_point=dataset.central_point,
        ansatz=args.ansatz,
        alphaD_mode=args.alphaD_mode,
    )
    loss, alphaD_pred_tf = loss_fn(tf.convert_to_tensor(params, dtype=tf.float32))

    alphaD_opt = alphaD_pred_tf.numpy()
    alphaD_true = np.asarray(dataset.alphaD_values)
    rel_err = np.abs(alphaD_opt - alphaD_true) / np.maximum(np.abs(alphaD_true), 1e-8)

    print("Loaded dataset")
    print("  param names:", dataset.param_names)
    print("  samples:", len(dataset.alphaD_values))
    print("  central point:", dataset.central_point.tolist())
    print("  config:", helper_gpt.summarize_alphaD_only_config(config))
    print("Loss:", float(loss.numpy()))
    print("Mean relative alphaD error:", float(np.mean(rel_err)))
    print("Max relative alphaD error:", float(np.max(rel_err)))

    if args.plots == "none":
        return

    fig_dir = Path(args.fig_dir) if args.fig_dir else run_dir / "diagnostics_em2"
    if args.plots == "save":
        fig_dir.mkdir(parents=True, exist_ok=True)

    fig1, ax1 = plt.subplots(figsize=(6, 4))
    ax1.scatter(alphaD_true, alphaD_opt)
    if len(alphaD_opt) <= args.max_label_points:
        for i in range(len(alphaD_opt)):
            ax1.text(alphaD_true[i], alphaD_opt[i], str(i), fontsize=9, ha="right", va="bottom")
    xline = np.linspace(
        min(np.min(alphaD_true), np.min(alphaD_opt)),
        max(np.max(alphaD_true), np.max(alphaD_opt)),
        100,
    )
    ax1.plot(xline, xline, color="black")
    ax1.set_xlabel(r"True $\alpha_D$")
    ax1.set_ylabel(r"Predicted $\alpha_D$")
    ax1.set_title(f"Emulator 2, n = {args.n}")

    fig2, ax2 = plt.subplots()
    x = dataset.param_values[:, 0]
    y = dataset.param_values[:, 1]
    scatter = ax2.scatter(x, y, c=rel_err, marker="s", cmap="Spectral", norm=LogNorm())
    fig2.colorbar(scatter, ax=ax2, label=r"Relative error $\alpha_D$")
    if len(alphaD_opt) <= args.max_label_points:
        for idx, (xi, yi) in enumerate(zip(x, y)):
            ax2.text(xi, yi, str(idx), ha="center", va="center", fontsize=8, color="black")

    train_rect = patches.Rectangle(
        (float(np.min(x)), float(np.min(y))),
        float(np.max(x) - np.min(x)),
        float(np.max(y) - np.min(y)),
        linewidth=1.5,
        edgecolor="black",
        facecolor="none",
    )
    ax2.add_patch(train_rect)
    ax2.set_xlabel(dataset.param_names[0])
    ax2.set_ylabel(dataset.param_names[1])
    ax2.set_title(r"Parameter-space relative error in $\alpha_D$")

    if args.plots == "save":
        fig1.savefig(fig_dir / "alphaD_true_vs_pred_em2.png", bbox_inches="tight", dpi=200)
        fig2.savefig(fig_dir / "parameter_error_map_em2.png", bbox_inches="tight", dpi=200)
    else:
        plt.show()


if __name__ == "__main__":
    main()
