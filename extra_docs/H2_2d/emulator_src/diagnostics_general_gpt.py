#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.colors import LogNorm

import helper_gpt #as helper_gpt


"""
Generalized performance diagnostics for Emulator 1 under the current
helper_gpt / main_gpt2 workflow.

This script:
1. Loads the dataset using the same CLI-style inputs as training.
2. Reconstructs spectra from best_params_global.txt.
3. Shows one detailed example spectrum and pole strengths.
4. Reports global and per-sample RMSE over all spectra.
5. Optionally plots predicted-vs-true alpha_D.
6. Plots parameter-space diagnostic maps:
   - 1 varying parameter  -> line/scatter along that parameter
   - 2 varying parameters -> 2D scatter map
   - >2 varying parameters -> first two varying parameters, colored by error
"""


def parse_args():
    p = argparse.ArgumentParser(description="Generalized diagnostics for Emulator 1")

    # Dataset / model inputs
    p.add_argument("--strength-dir", required=True)
    p.add_argument("--alphaD-dir", default=None)
    p.add_argument("--strength-regex", required=True)
    p.add_argument("--alphaD-regex", default=None)
    p.add_argument("--filter-ranges", default=None,
                   help="JSON dict, e.g. '{\"p1\":[-0.75,-0.51]}'")
    p.add_argument("--strength-only", action="store_true",
                   help="Skip alphaD diagnostics and report/plot strength-function errors only.")

    # Emulator structure
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--retain", type=float, required=True)
    p.add_argument("--ansatz", required=True)
    p.add_argument("--width-model", required=True)
    p.add_argument("--use-vector-terms", dest="use_vector_terms",
                   action="store_true", default=True)
    p.add_argument("--no-vector-terms", dest="use_vector_terms",
                   action="store_false")

    # Diagnostics inputs / behavior
    p.add_argument("--save-dir", required=True,
                   help="Directory containing best_params_global.txt; also used for saved figures unless overridden.")
    p.add_argument("--params-file", default=None,
                   help="Override path to best_params_global.txt")
    p.add_argument("--detail-idx", type=int, default=None,
                   help="Index to use for detailed spectrum view. Defaults to middle sample.")
    p.add_argument("--plots", choices=["show", "save", "both"], default="show")
    p.add_argument("--fig-dir", default=None,
                   help="Directory for saved figures. Defaults to <save-dir>/diagnostics")
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--max-label-points", type=int, default=60,
                   help="Only annotate point indices when sample count is <= this threshold.")
    p.add_argument("--yscale", choices=["linear", "log"], default="linear")

    return p.parse_args()



def maybe_parse_json_dict(text):
    if text is None:
        return None
    if isinstance(text, dict):
        return text
    text = text.strip()
    if text.lower() == "none" or text == "":
        return None
    return json.loads(text)



def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)



def choose_detail_idx(n_samples, requested=None):
    if n_samples <= 0:
        raise ValueError("Dataset is empty.")
    if requested is None:
        return n_samples // 2
    if not (0 <= requested < n_samples):
        raise IndexError(f"detail_idx={requested} out of range for {n_samples} samples")
    return requested



def infer_varying_dimensions(param_values, tol=1e-12):
    spans = np.ptp(param_values, axis=0)
    varying = np.where(spans > tol)[0]
    return varying, spans



def build_outputs(dataset, params, n, retain, ansatz, width_model, use_vector_terms):
    n_params = dataset.param_values.shape[1]

    config = helper_gpt.AnsatzConfig(
        n=n,
        n_params=n_params,
        ansatz=ansatz,
        width_model=width_model,
        use_vector_terms=use_vector_terms,
    )

    param_values_tf = tf.convert_to_tensor(dataset.param_values, dtype=tf.float32)
    central_point_tf = tf.convert_to_tensor(dataset.central_point, dtype=tf.float32)
    params_tf = tf.convert_to_tensor(params, dtype=tf.float32)

    M_batch, v_batch, eta_batch, _ = helper_gpt.build_model_matrices_and_vectors(
        params=params_tf,
        config=config,
        param_values=param_values_tf,
        central_point=central_point_tf,
    )

    eigenvalues, eigenvectors = tf.linalg.eigh(M_batch)

    n_i = int(eigenvalues.shape[1])
    k_keep = int(round(retain * n_i))
    k_keep = max(1, min(k_keep, n_i))

    left = (n_i - k_keep) // 2
    right = left + k_keep

    eigvals_kept = eigenvalues[:, left:right]
    eigvecs_kept = eigenvectors[:, :, left:right]

    proj = tf.matmul(tf.transpose(eigvecs_kept, perm=[0, 2, 1]), v_batch[:, :, None])
    proj = tf.squeeze(proj, axis=-1)
    B_batch = tf.square(proj)

    omega = dataset.strengths[0][:, 0].astype(np.float32)
    omega_tf = tf.convert_to_tensor(omega, dtype=tf.float32)

    opt_strength = []
    alphaD_opt = []
    opt_eigs = []
    opt_Bs = []

    for i in range(len(dataset.strengths)):
        y_pred = helper_gpt.give_me_Lorentzian(
            energy=omega_tf,
            poles=eigvals_kept[i],
            strength=B_batch[i],
            width=eta_batch[i] / 2.0,
        ).numpy()

        opt_strength.append(y_pred)
        opt_eigs.append(eigvals_kept[i].numpy())
        opt_Bs.append(B_batch[i].numpy())
        alphaD_opt.append(
            helper_gpt.calculate_alphaD(
                eigvals_kept[i].numpy(),
                B_batch[i].numpy()
            )
        )

    return {
        "config": config,
        "eta_batch": eta_batch.numpy(),
        "opt_strength": np.asarray(opt_strength),
        "alphaD_opt": np.asarray(alphaD_opt),
        "opt_eigs": opt_eigs,
        "opt_Bs": opt_Bs,
    }



def compute_rmse(dataset, opt_strength):
    residuals = []
    per_sample_rmse = []
    for i in range(len(dataset.strengths)):
        diff = dataset.strengths[i][:, 1] - opt_strength[i]
        residuals.append(diff)
        per_sample_rmse.append(np.sqrt(np.mean(diff ** 2)))

    all_residuals = np.concatenate(residuals)
    global_rmse = np.sqrt(np.mean(all_residuals ** 2))
    return global_rmse, np.asarray(per_sample_rmse)



def plot_detail(dataset, outputs, detail_idx, yscale="linear"):
    fig, ax = plt.subplots(figsize=(8, 5))

    x = dataset.strengths[detail_idx][:, 0]
    y_true = dataset.strengths[detail_idx][:, 1]
    y_pred = outputs["opt_strength"][detail_idx]

    ax.plot(x, y_pred, label="pred")
    ax.plot(x, y_true, label="true")
    ax.stem(outputs["opt_eigs"][detail_idx], outputs["opt_Bs"][detail_idx], basefmt=" ")

    if yscale == "log":
        ax.set_yscale("log")

    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel("Strength")
    ax.set_title(f"Detailed spectrum check (sample {detail_idx})")
    ax.legend()
    fig.tight_layout()
    return fig



def plot_alphaD_true_vs_pred(alphaD_true, alphaD_opt, label_points=True):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(alphaD_true, alphaD_opt)

    lo = min(np.min(alphaD_true), np.min(alphaD_opt))
    hi = max(np.max(alphaD_true), np.max(alphaD_opt))
    xline = np.linspace(lo, hi, 200)
    ax.plot(xline, xline, color="black")

    if label_points:
        for i in range(len(alphaD_opt)):
            ax.text(alphaD_true[i], alphaD_opt[i], str(i), fontsize=8,
                    ha="right", va="bottom")

    ax.set_xlabel(r"True $\alpha_D$")
    ax.set_ylabel(r"Predicted $\alpha_D$")
    ax.set_title(r"$\alpha_D$: prediction vs truth")
    fig.tight_layout()
    return fig



def plot_parameter_error_map(dataset, alphaD_true, alphaD_opt, max_label_points=60):
    param_values = np.asarray(dataset.param_values)
    param_names = list(dataset.param_names)
    rel_err = np.abs(alphaD_opt - alphaD_true) / np.maximum(np.abs(alphaD_true), 1e-12)
    varying, spans = infer_varying_dimensions(param_values)
    label_points = len(rel_err) <= max_label_points

    # No varying parameter: just index vs error
    if len(varying) == 0:
        fig, ax = plt.subplots(figsize=(7, 4))
        idx = np.arange(len(rel_err))
        ax.scatter(idx, rel_err)
        ax.set_xlabel("Sample index")
        ax.set_ylabel(r"Relative error in $\alpha_D$")
        ax.set_title(r"No varying parameter detected")
        fig.tight_layout()
        return fig

    # One varying parameter: line of points, no rectangle
    if len(varying) == 1:
        j = varying[0]
        x = param_values[:, j]
        order = np.argsort(x)

        fig, ax = plt.subplots(figsize=(7, 4.5))
        sc = ax.scatter(x[order], rel_err[order], c=rel_err[order], cmap="Spectral")
        ax.plot(x[order], rel_err[order], alpha=0.7)
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(r"Relative error $\alpha_D$")

        if label_points:
            for idx in order:
                ax.text(x[idx], rel_err[idx], str(idx), fontsize=8,
                        ha="center", va="bottom")

        ax.set_xlabel(param_names[j])
        ax.set_ylabel(r"Relative error in $\alpha_D$")
        ax.set_title(r"1D parameter-space error trace")
        fig.tight_layout()
        return fig

    # Two or more varying parameters: show first two varying dimensions
    j0, j1 = varying[:2]
    x = param_values[:, j0]
    y = param_values[:, j1]

    positive_rel_err = np.maximum(rel_err, 1e-15)
    use_lognorm = np.any(positive_rel_err > 0) and (positive_rel_err.max() / positive_rel_err.min() > 50)
    norm = LogNorm(vmin=positive_rel_err.min(), vmax=positive_rel_err.max()) if use_lognorm else None

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sc = ax.scatter(x, y, c=positive_rel_err, marker="s", cmap="Spectral", norm=norm)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(r"Relative error $\alpha_D$")

    if label_points:
        for idx, (xi, yi) in enumerate(zip(x, y)):
            ax.text(xi, yi, str(idx), ha="center", va="center", fontsize=8, color="black")

    ax.set_xlabel(param_names[j0])
    ax.set_ylabel(param_names[j1])

    extra = ""
    if len(varying) > 2:
        hidden = ", ".join(param_names[k] for k in varying[2:])
        extra = f" (first two varying dims shown; also varies: {hidden})"
    ax.set_title(r"Parameter-space relative error in $\alpha_D$" + extra)
    fig.tight_layout()
    return fig


def plot_parameter_metric_map(dataset, metric, metric_label, title, max_label_points=60):
    param_values = np.asarray(dataset.param_values)
    param_names = list(dataset.param_names)
    metric = np.asarray(metric)
    varying, spans = infer_varying_dimensions(param_values)
    label_points = len(metric) <= max_label_points

    if len(varying) == 0:
        fig, ax = plt.subplots(figsize=(7, 4))
        idx = np.arange(len(metric))
        ax.scatter(idx, metric)
        ax.set_xlabel("Sample index")
        ax.set_ylabel(metric_label)
        ax.set_title(title)
        fig.tight_layout()
        return fig

    if len(varying) == 1:
        j = varying[0]
        x = param_values[:, j]
        order = np.argsort(x)

        fig, ax = plt.subplots(figsize=(7, 4.5))
        sc = ax.scatter(x[order], metric[order], c=metric[order], cmap="Spectral")
        ax.plot(x[order], metric[order], alpha=0.7)
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(metric_label)

        if label_points:
            for idx in order:
                ax.text(x[idx], metric[idx], str(idx), fontsize=8,
                        ha="center", va="bottom")

        ax.set_xlabel(param_names[j])
        ax.set_ylabel(metric_label)
        ax.set_title(title)
        fig.tight_layout()
        return fig

    j0, j1 = varying[:2]
    x = param_values[:, j0]
    y = param_values[:, j1]

    positive_metric = np.maximum(metric, 1e-15)
    use_lognorm = np.any(positive_metric > 0) and (positive_metric.max() / positive_metric.min() > 50)
    norm = LogNorm(vmin=positive_metric.min(), vmax=positive_metric.max()) if use_lognorm else None

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sc = ax.scatter(x, y, c=positive_metric, marker="s", cmap="Spectral", norm=norm)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(metric_label)

    if label_points:
        for idx, (xi, yi) in enumerate(zip(x, y)):
            ax.text(xi, yi, str(idx), ha="center", va="center", fontsize=8, color="black")

    ax.set_xlabel(param_names[j0])
    ax.set_ylabel(param_names[j1])
    ax.set_title(title)
    fig.tight_layout()
    return fig



def save_figure(fig, outdir, filename, dpi=200):
    ensure_dir(outdir)
    path = os.path.join(outdir, filename)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"Saved: {path}")



def main():
    args = parse_args()
    filter_ranges = maybe_parse_json_dict(args.filter_ranges)

    params_file = args.params_file or os.path.join(args.save_dir, "best_params_global.txt")
    params = np.loadtxt(params_file).astype(np.float32)

    dataset = helper_gpt.load_dataset(
        strength_dir=args.strength_dir,
        alphaD_dir=args.alphaD_dir,
        strength_regex=args.strength_regex,
        alphaD_regex=args.alphaD_regex,
        filter_ranges=filter_ranges,
        central_point=None,
    )

    n_samples = len(dataset.strengths)
    detail_idx = choose_detail_idx(n_samples, args.detail_idx)

    print("Loaded dataset")
    print("  param names:", dataset.param_names)
    print("  samples:", n_samples)
    print("  n_params:", dataset.param_values.shape[1])
    print("  central point:", dataset.central_point.tolist())
    print("  params file:", params_file)

    outputs = build_outputs(
        dataset=dataset,
        params=params,
        n=args.n,
        retain=args.retain,
        ansatz=args.ansatz,
        width_model=args.width_model,
        use_vector_terms=args.use_vector_terms,
    )

    alphaD_true = np.asarray(dataset.alphaD_values)
    alphaD_opt = outputs["alphaD_opt"]

    global_rmse, per_sample_rmse = compute_rmse(dataset, outputs["opt_strength"])

    print("Detailed point:", dataset.param_values[detail_idx])
    print("eta:", float(outputs["eta_batch"][detail_idx]))
    print("Global RMSE:", global_rmse)
    print("Mean sample RMSE:", float(np.mean(per_sample_rmse)))
    print("Median sample RMSE:", float(np.median(per_sample_rmse)))
    if not args.strength_only:
        print("alpha_D true / pred:", alphaD_true[detail_idx], alphaD_opt[detail_idx])
        print("Max alpha_D rel. err:", float(np.max(np.abs(alphaD_opt - alphaD_true) / np.maximum(np.abs(alphaD_true), 1e-12))))

    fig1 = plot_detail(dataset, outputs, detail_idx, yscale=args.yscale)
    if args.strength_only:
        fig2 = plot_parameter_metric_map(
            dataset,
            per_sample_rmse,
            metric_label="Spectrum RMSE",
            title="Parameter-space spectrum RMSE",
            max_label_points=args.max_label_points,
        )
        figures = [
            (fig1, "detail_spectrum.png"),
            (fig2, "parameter_spectrum_rmse.png"),
        ]
    else:
        fig2 = plot_alphaD_true_vs_pred(
            alphaD_true,
            alphaD_opt,
            label_points=(n_samples <= args.max_label_points),
        )
        fig3 = plot_parameter_error_map(
            dataset,
            alphaD_true,
            alphaD_opt,
            max_label_points=args.max_label_points,
        )
        figures = [
            (fig1, "detail_spectrum.png"),
            (fig2, "alphaD_true_vs_pred.png"),
            (fig3, "parameter_error_map.png"),
        ]

    if args.plots in {"save", "both"}:
        fig_dir = args.fig_dir or os.path.join(args.save_dir, "diagnostics")
        for fig, filename in figures:
            save_figure(fig, fig_dir, filename, dpi=args.dpi)

    if args.plots in {"show", "both"}:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
