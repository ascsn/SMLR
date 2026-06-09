#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import tensorflow as tf


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "emulator_src"
sys.path.insert(0, str(SRC_DIR))
XLIM = (0.0, 40.0)

import helper_gpt


def load_best() -> dict:
    best_path = Path(os.environ.get("H2_BEST_MODEL_JSON", ROOT / "results/sweep_runs/best_plots/best_model.json"))
    with best_path.open() as f:
        return json.load(f)


def load_model_outputs(best: dict):
    run_dir = Path(best["run_dir"])
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    summary_path = run_dir / "run_summary.json"
    strength_scale = 1.0
    if summary_path.exists():
        strength_scale = float(json.load(open(summary_path)).get("strength_scale", 1.0))

    dataset = helper_gpt.load_dataset(
        strength_dir=str(ROOT / "data/total_strength"),
        alphaD_dir=None,
        strength_regex=r"strength_(?P<q>[0-9.]+)_(?P<theta>[0-9.]+)\.out",
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
    params = tf.convert_to_tensor(np.loadtxt(run_dir / "best_params_global.txt").astype(np.float32), dtype=tf.float32)
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

    pred = []
    for i in range(len(dataset.strengths)):
        pred.append(helper_gpt.give_me_Lorentzian(
            energy=omega_tf,
            poles=eigvals[i],
            strength=B_batch[i],
            width=eta_batch[i] / 2.0,
        ).numpy())
    pred = np.asarray(pred) / strength_scale
    true = np.stack([s[:, 1] for s in dataset.strengths], axis=0) / strength_scale
    return dataset, true, pred, eigvals.numpy(), B_batch.numpy(), eta_batch.numpy()


def rmse(true: np.ndarray, pred: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((true - pred) ** 2, axis=1))


def rel_l2(true: np.ndarray, pred: np.ndarray) -> np.ndarray:
    num = np.sqrt(np.sum((true - pred) ** 2, axis=1))
    den = np.sqrt(np.sum(true ** 2, axis=1)) + 1e-12
    return num / den


def choose_indices(dataset, per_rmse):
    q = dataset.param_values[:, 0]
    theta = dataset.param_values[:, 1]
    q_unique = np.unique(q)
    theta_unique = np.unique(theta)
    targets = {
        "q_min_theta_min": (q_unique[0], theta_unique[0]),
        "q_min_theta_mid": (q_unique[0], theta_unique[len(theta_unique) // 2]),
        "q_min_theta_max": (q_unique[0], theta_unique[-1]),
        "q_mid_theta_min": (q_unique[len(q_unique) // 2], theta_unique[0]),
        "q_mid_theta_mid": (q_unique[len(q_unique) // 2], theta_unique[len(theta_unique) // 2]),
        "q_mid_theta_max": (q_unique[len(q_unique) // 2], theta_unique[-1]),
        "q_max_theta_min": (q_unique[-1], theta_unique[0]),
        "q_max_theta_mid": (q_unique[-1], theta_unique[len(theta_unique) // 2]),
        "q_max_theta_max": (q_unique[-1], theta_unique[-1]),
    }
    selected = {}
    for name, (tq, tt) in targets.items():
        q_match = np.isclose(q, tq, rtol=0.0, atol=1e-6)
        theta_match = np.isclose(theta, tt, rtol=0.0, atol=1e-6)
        matches = np.flatnonzero(q_match & theta_match)
        if len(matches) == 0:
            selected[name] = int(np.argmin(((q - tq) / max(np.ptp(q), 1e-12)) ** 2 + ((theta - tt) / max(np.ptp(theta), 1e-12)) ** 2))
        else:
            selected[name] = int(matches[0])
    order = np.argsort(per_rmse)
    selected["best_rmse"] = int(order[0])
    selected["median_rmse"] = int(order[len(order) // 2])
    selected["worst_rmse"] = int(order[-1])
    return selected


def plot_strength_true_vs_prediction(true, pred, outdir):
    x = true.reshape(-1)
    y = pred.reshape(-1)
    positive = (x > 0) & (y > 0)
    lim_hi = max(float(np.max(x)), float(np.max(y)))
    lim_lo = max(min(float(np.min(x[positive])), float(np.min(y[positive]))) if np.any(positive) else 1e-12, 1e-12)

    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.hexbin(np.maximum(x, lim_lo), np.maximum(y, lim_lo), gridsize=80, bins="log", mincnt=1, cmap="viridis")
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], color="black", lw=1.0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("True strength")
    ax.set_ylabel("Predicted strength")
    ax.set_title("Strength: prediction vs truth")
    cbar = fig.colorbar(ax.collections[0], ax=ax)
    cbar.set_label("log10(point count)")
    fig.tight_layout()
    fig.savefig(outdir / "strength_true_vs_prediction.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_parameter_error_map(dataset, metric, outdir, filename, title, label):
    q = dataset.param_values[:, 0]
    theta = dataset.param_values[:, 1]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    positive = np.maximum(metric, 1e-15)
    norm = LogNorm(vmin=positive.min(), vmax=positive.max()) if positive.max() / positive.min() > 20 else None
    sc = ax.scatter(q, theta, c=positive, marker="s", s=230, cmap="Spectral_r", norm=norm)
    for i, (qi, ti) in enumerate(zip(q, theta)):
        ax.text(qi, ti, str(i), ha="center", va="center", fontsize=7, color="black")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(label)
    ax.set_xlabel("q")
    ax.set_ylabel("theta (rad)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(outdir / filename, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_index_selection_map(dataset, selected, outdir):
    q = dataset.param_values[:, 0]
    theta = dataset.param_values[:, 1]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(q, theta, marker="s", s=180, color="#DDDDDD", edgecolor="#777777")
    for i, (qi, ti) in enumerate(zip(q, theta)):
        ax.text(qi, ti, str(i), ha="center", va="center", fontsize=7)
    for name, idx in selected.items():
        ax.scatter(q[idx], theta[idx], s=380, facecolors="none", edgecolors="#D62728", linewidths=1.8)
    ax.set_xlabel("q")
    ax.set_ylabel("theta (rad)")
    ax.set_title("Sample index map; red rings mark detailed diagnostics")
    fig.tight_layout()
    fig.savefig(outdir / "index_selection_map.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def visible_poles(eigvals, strengths, idx):
    lo, hi = XLIM
    mask = (eigvals[idx] >= lo) & (eigvals[idx] <= hi)
    return eigvals[idx][mask], strengths[idx][mask]


def plot_detail_spectrum(dataset, true, pred, eigvals, strengths, idx, outdir, label):
    omega = dataset.strengths[idx][:, 0]
    residual = pred[idx] - true[idx]
    q, theta = dataset.param_values[idx]

    fig, (ax, rx) = plt.subplots(
        2, 1, figsize=(8.5, 6.4), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08}
    )
    ax.plot(omega, true[idx], color="black", lw=1.8, label="true")
    ax.plot(omega, pred[idx], color="#D62728", lw=1.5, ls="--", label="pred")
    poles_x, poles_b = visible_poles(eigvals, strengths, idx)
    ax.stem(poles_x, poles_b, linefmt="C0-", markerfmt="C0o", basefmt=" ", label="poles")
    ax.set_xlim(*XLIM)
    ax.set_ylabel("Strength")
    ax.set_title(f"{label}: index {idx}, q={q:.4g}, theta={theta:.4g}")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, alpha=0.2)

    rx.axhline(0.0, color="black", lw=0.8)
    rx.plot(omega, residual, color="#4C78A8", lw=1.0)
    rx.set_xlabel("Energy (eV)")
    rx.set_ylabel("Pred - true")
    rx.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(outdir / f"detail_spectrum_{label}_idx{idx:02d}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_detail_grid(dataset, true, pred, selected, outdir):
    # Separate from the older overlay: no poles, thicker lines, and split into readable panels.
    labels = [
        "q_min_theta_min", "q_min_theta_mid", "q_min_theta_max",
        "q_mid_theta_min", "q_mid_theta_mid", "q_mid_theta_max",
        "q_max_theta_min", "q_max_theta_mid", "q_max_theta_max",
    ]
    omega = dataset.strengths[0][:, 0]
    fig, axes = plt.subplots(3, 3, figsize=(13.5, 9), sharex=True)
    for ax, name in zip(axes.ravel(), labels):
        idx = selected[name]
        q, theta = dataset.param_values[idx]
        ax.plot(omega, true[idx], color="black", lw=1.7, label="true")
        ax.plot(omega, pred[idx], color="#D62728", lw=1.35, ls="--", label="pred")
        ax.set_xlim(*XLIM)
        ax.set_title(f"idx {idx}: q={q:.3g}, theta={theta:.3g}")
        ax.grid(True, alpha=0.2)
    axes[0, 0].legend(frameon=False, fontsize=9)
    for ax in axes[-1, :]:
        ax.set_xlabel("Energy (eV)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Strength")
    fig.suptitle("Selected H2 strength predictions", y=1.01)
    fig.tight_layout()
    fig.savefig(outdir / "detail_spectrum_selected_grid.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    best = load_best()
    outdir = Path(os.environ.get(
        "H2_DIAG_OUTDIR",
        ROOT / "results/sweep_runs/best_plots/original_style_strength_diagnostics_xlim_0_40",
    ))
    outdir.mkdir(parents=True, exist_ok=True)
    dataset, true, pred, eigvals, strengths, eta = load_model_outputs(best)
    per_rmse = rmse(true, pred)
    per_rel = rel_l2(true, pred)
    selected = choose_indices(dataset, per_rmse)

    with (outdir / "per_sample_strength_metrics.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "q", "theta", "sample_id", "rmse", "relative_l2_error", "selected_labels"])
        inverse_selected = {}
        for label, idx in selected.items():
            inverse_selected.setdefault(idx, []).append(label)
        for idx, (params, sample_id, sample_rmse, sample_rel) in enumerate(zip(dataset.param_values, dataset.sample_ids, per_rmse, per_rel)):
            writer.writerow([
                idx,
                float(params[0]),
                float(params[1]),
                "_".join(sample_id),
                float(sample_rmse),
                float(sample_rel),
                ";".join(inverse_selected.get(idx, [])),
            ])

    plot_strength_true_vs_prediction(true, pred, outdir)
    plot_parameter_error_map(dataset, per_rmse, outdir, "parameter_error_map.png", "Parameter-space spectrum RMSE", "Spectrum RMSE")
    plot_parameter_error_map(dataset, per_rel, outdir, "parameter_relative_l2_error_map.png", "Parameter-space relative L2 error", "Relative L2 error")
    plot_index_selection_map(dataset, selected, outdir)
    plot_detail_grid(dataset, true, pred, selected, outdir)
    for label in ["best_rmse", "median_rmse", "worst_rmse", "q_mid_theta_mid"]:
        plot_detail_spectrum(dataset, true, pred, eigvals, strengths, selected[label], outdir, label)

    summary = {
        "best_model": best,
        "global_rmse": float(np.sqrt(np.mean((true - pred) ** 2))),
        "mean_sample_rmse": float(np.mean(per_rmse)),
        "median_sample_rmse": float(np.median(per_rmse)),
        "max_sample_rmse": float(np.max(per_rmse)),
        "selected_indices": selected,
    }
    (outdir / "diagnostic_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote diagnostics to {outdir}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
