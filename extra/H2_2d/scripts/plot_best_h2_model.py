#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/Users/laurenjin/envs/smlr-frib/bin/python")
if not PYTHON.exists():
    PYTHON = Path(sys.executable)


def read_summary(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return sorted(rows, key=lambda r: float(r["strength_cost"]))


def run_diagnostics(best: dict) -> None:
    run_dir = Path(best["run_dir"])
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    cmd = [
        str(PYTHON),
        str(ROOT / "emulator_src/diagnostics_general_gpt.py"),
        "--strength-dir", str(ROOT / "data/total_strength"),
        "--strength-regex", r"strength_(?P<q>[0-9.]+)_(?P<theta>[0-9.]+)\.out",
        "--strength-only",
        "--n", best["n"],
        "--retain", best["retain"],
        "--ansatz", best["ansatz"],
        "--width-model", best["width_model"],
        "--plots", "save",
        "--save-dir", str(run_dir),
    ]
    subprocess.run(cmd, cwd=ROOT, check=True)


def plot_sweep_ranking(rows: list[dict], outdir: Path) -> None:
    names = [r["name"].split("_", 1)[0] for r in rows]
    vals = [float(r["strength_cost"]) for r in rows]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(names, vals, color="#4C78A8")
    ax.set_ylabel("Strength cost")
    ax.set_xlabel("Model")
    ax.set_title("H2 hyperparameter sweep ranking")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(outdir / "sweep_ranking.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_best_cost_history(best: dict, outdir: Path) -> None:
    run_dir = Path(best["run_dir"])
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    seed = best["best_seed"]
    hist = np.loadtxt(run_dir / f"seed_{seed}" / "cost_history.txt")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(hist, lw=1.0)
    ax.set_yscale("log")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cost")
    ax.set_title(f"Best model convergence ({best['name']}, seed {seed})")
    fig.tight_layout()
    fig.savefig(outdir / "best_cost_history.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_selected_predictions(best: dict, outdir: Path) -> None:
    run_dir = Path(best["run_dir"])
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    params_file = run_dir / "best_params_global.txt"
    summary_path = run_dir / "run_summary.json"
    strength_scale = 1.0
    if summary_path.exists():
        strength_scale = float(json.load(open(summary_path)).get("strength_scale", 1.0))

    src_dir = ROOT / "emulator_src"
    sys.path.insert(0, str(src_dir))
    import helper_gpt
    import tensorflow as tf

    dataset = helper_gpt.load_dataset(
        strength_dir=str(ROOT / "data/total_strength"),
        strength_regex=r"strength_(?P<q>[0-9.]+)_(?P<theta>[0-9.]+)\.out",
        alphaD_dir=None,
    )
    if strength_scale != 1.0:
        dataset = helper_gpt.GenericDataset(
            param_names=list(dataset.param_names),
            param_values=dataset.param_values.copy(),
            sample_ids=list(dataset.sample_ids),
            strengths=[s.copy() * np.array([1.0, strength_scale], dtype=np.float32) for s in dataset.strengths],
            alphaD_values=dataset.alphaD_values.copy(),
            alphaD_raw=list(dataset.alphaD_raw),
            central_point=dataset.central_point.copy(),
        )
    config = helper_gpt.AnsatzConfig(
        n=int(best["n"]),
        n_params=dataset.param_values.shape[1],
        ansatz=best["ansatz"],
        width_model=best["width_model"],
        use_vector_terms=True,
    )
    params = tf.convert_to_tensor(np.loadtxt(params_file).astype(np.float32), dtype=tf.float32)
    M_batch, v_batch, eta_batch, _ = helper_gpt.build_model_matrices_and_vectors(
        params=params,
        config=config,
        param_values=tf.convert_to_tensor(dataset.param_values, dtype=tf.float32),
        central_point=tf.convert_to_tensor(dataset.central_point, dtype=tf.float32),
    )
    eigvals, eigvecs = tf.linalg.eigh(M_batch)
    n_i = int(eigvals.shape[1])
    k_keep = max(1, min(int(round(float(best["retain"]) * n_i)), n_i))
    left = (n_i - k_keep) // 2
    right = left + k_keep
    eigvals = eigvals[:, left:right]
    eigvecs = eigvecs[:, :, left:right]
    proj = tf.matmul(tf.transpose(eigvecs, perm=[0, 2, 1]), v_batch[:, :, None])
    B_batch = tf.square(tf.squeeze(proj, axis=-1))
    omega = dataset.strengths[0][:, 0].astype(np.float32)
    omega_tf = tf.convert_to_tensor(omega, dtype=tf.float32)

    q = dataset.param_values[:, 0]
    theta = dataset.param_values[:, 1]
    targets = [
        (q.min(), theta.min()),
        (q.min(), np.median(np.unique(theta))),
        (q.min(), theta.max()),
        (np.median(np.unique(q)), theta.min()),
        (np.median(np.unique(q)), np.median(np.unique(theta))),
        (np.median(np.unique(q)), theta.max()),
        (q.max(), theta.min()),
        (q.max(), np.median(np.unique(theta))),
        (q.max(), theta.max()),
    ]
    indices = []
    for tq, tt in targets:
        idx = int(np.argmin((q - tq) ** 2 + (theta - tt) ** 2))
        if idx not in indices:
            indices.append(idx)

    fig, axes = plt.subplots(3, 3, figsize=(12, 9), sharex=True)
    for ax, idx in zip(axes.ravel(), indices):
        y_pred = helper_gpt.give_me_Lorentzian(
            energy=omega_tf,
            poles=eigvals[idx],
            strength=B_batch[idx],
            width=eta_batch[idx] / 2.0,
        ).numpy() / strength_scale
        y_true = dataset.strengths[idx][:, 1] / strength_scale
        ax.plot(omega, y_true, color="#222222", lw=1.2, label="true")
        ax.plot(omega, y_pred, color="#D62728", lw=1.0, label="pred")
        ax.set_title(f"q={q[idx]:.3g}, theta={theta[idx]:.3g}")
        ax.grid(True, alpha=0.2)
    axes[0, 0].legend(frameon=False, fontsize=8)
    for ax in axes[-1, :]:
        ax.set_xlabel("Energy (eV)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Strength")
    fig.suptitle("Best model selected spectrum predictions", y=1.01)
    fig.tight_layout()
    fig.savefig(outdir / "best_selected_predictions.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    sweep_dir = ROOT / "results/sweep_runs"
    rows = read_summary(sweep_dir / "sweep_summary.csv")
    best = rows[0]
    outdir = sweep_dir / "best_plots"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "best_model.json").write_text(json.dumps(best, indent=2))

    run_diagnostics(best)
    plot_sweep_ranking(rows, outdir)
    plot_best_cost_history(best, outdir)
    plot_selected_predictions(best, outdir)
    print(f"Best model: {best['name']}")
    print(f"Strength cost: {best['strength_cost']}")
    print(f"Plots: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
