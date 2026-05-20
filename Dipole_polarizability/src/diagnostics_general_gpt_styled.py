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
import matplotlib.colors as mcolors
import matplotlib.patches as patches

import helper_gpt #as helper_gpt
try:
    from smlr import metrics as smlr_metrics
    from smlr.diagnostics import (
        DiagnosticLabels,
        save_figure as save_standard_figure,
        write_standard_observable_diagnostics,
    )
except ModuleNotFoundError:  # pragma: no cover - source-tree execution before install
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from smlr import metrics as smlr_metrics
    from smlr.diagnostics import (
        DiagnosticLabels,
        save_figure as save_standard_figure,
        write_standard_observable_diagnostics,
    )


"""
Generalized performance diagnostics for Emulator 1 under the current
helper_gpt / main_gpt2 workflow.

This script:
1. Loads the dataset using the same CLI-style inputs as training.
2. Reconstructs spectra and alpha_D values from best_params_global.txt.
3. Shows one detailed example spectrum and pole strengths.
4. Reports global and per-sample RMSE over all spectra.
5. Plots predicted-vs-true alpha_D.
6. Plots parameter-space alpha_D relative error:
   - 1 varying parameter  -> line/scatter along that parameter
   - 2 varying parameters -> 2D scatter map
   - >2 varying parameters -> first two varying parameters, colored by error
"""


def parse_args():
    p = argparse.ArgumentParser(description="Generalized diagnostics for Emulator 1")

    # Dataset / model inputs
    p.add_argument("--strength-dir", required=True)
    p.add_argument("--alphaD-dir", required=True)
    p.add_argument("--strength-regex", required=True)
    p.add_argument("--alphaD-regex", default=None)
    p.add_argument("--filter-ranges", default=None,
                   help="JSON dict, e.g. '{\"p1\":[-0.75,-0.51]}'")

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
    p.add_argument("--max-label-points", type=int, default=1000,
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



def apply_notebook_plot_style():
    plt.rcParams.update({'font.size': 16})
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'


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
    params = coerce_params_layout(params, config)

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
            width=eta_batch[i],
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


def coerce_params_layout(params, config):
    params = np.asarray(params, dtype=np.float32).reshape(-1)
    layout = helper_gpt.get_packed_layout(config)
    if params.size == layout.total_size:
        return params

    if config.ansatz != "paper_dipole" or config.n_params != 2:
        raise ValueError(
            f"Parameter vector has length {params.size}, expected {layout.total_size}."
        )

    n = int(config.n)
    n_upper = n * (n + 1) // 2
    legacy_size = 1 + 3 * n + n + 4 * n_upper + 4
    if params.size != legacy_size:
        raise ValueError(
            f"Parameter vector has length {params.size}; expected current layout "
            f"{layout.total_size} or legacy paper layout {legacy_size}."
        )

    idx = 0
    eta = params[idx:idx + 1]
    idx += 1
    v0 = params[idx:idx + n]
    idx += n
    v1 = params[idx:idx + n]
    idx += n
    v2 = params[idx:idx + n]
    idx += n
    d_diag = params[idx:idx + n]
    idx += n
    s1 = params[idx:idx + n_upper]
    idx += n_upper
    s2 = params[idx:idx + n_upper]
    idx += n_upper
    s3 = params[idx:idx + n_upper]
    idx += n_upper
    idx += n_upper  # Legacy S4 block is packed but unused by the paper EM1 equation.
    x1, x2, x3, x4 = params[idx:idx + 4]

    converted = np.zeros(layout.total_size, dtype=np.float32)
    converted[0] = eta[0]
    converted[layout.v0_slice] = v0
    converted[layout.v_linear_slice] = np.concatenate([v1, v2])
    converted[layout.d_diag_slice] = d_diag
    converted[layout.basis_slice] = np.concatenate([s1, s2, s3])
    converted[layout.width_bias_slice] = np.array([x2], dtype=np.float32)
    converted[layout.width_linear_slice] = np.array([x3, x4], dtype=np.float32)
    converted[layout.feature_param_slice] = np.array([x1], dtype=np.float32)

    print(
        f"Converted legacy paper parameter layout ({legacy_size}) "
        f"to current helper_gpt layout ({layout.total_size})."
    )
    return converted



def compute_rmse(dataset, opt_strength):
    residuals = []
    per_sample_rmse = []
    for i in range(len(dataset.strengths)):
        diff = dataset.strengths[i][:, 1] - opt_strength[i]
        residuals.append(diff)
        per_sample_rmse.append(smlr_metrics.rmse(diff, np.zeros_like(diff)))

    all_residuals = np.concatenate(residuals)
    global_rmse = smlr_metrics.rmse(all_residuals, np.zeros_like(all_residuals))
    return global_rmse, np.asarray(per_sample_rmse)



def draw_detail_axis(ax, dataset, outputs, detail_idx, yscale="linear"):
    ax.clear()

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
    params = ", ".join(
        f"{name}={value:g}"
        for name, value in zip(dataset.param_names, dataset.param_values[detail_idx])
    )
    ax.set_title(f"Detailed spectrum check (sample {detail_idx}: {params})")
    ax.legend()


def plot_detail(dataset, outputs, detail_idx, yscale="linear"):
    fig, ax = plt.subplots(figsize=(8, 5))
    draw_detail_axis(ax, dataset, outputs, detail_idx, yscale=yscale)
    fig.tight_layout()
    return fig



def plot_alphaD_true_vs_pred(alphaD_true, alphaD_opt, label_points=True):
    fig, ax = plt.subplots(figsize=(6, 5))
    scatter = ax.scatter(alphaD_true, alphaD_opt, picker=True, pickradius=5)

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
    return fig, scatter


def connect_alphaD_picker(fig_alphaD, scatter, fig_detail, dataset, outputs, yscale="linear"):
    detail_ax = fig_detail.axes[0]

    def on_pick(event):
        if event.artist is not scatter or len(event.ind) == 0:
            return
        if event.mouseevent is not None and len(event.ind) > 1:
            offsets = scatter.get_offsets()[event.ind]
            click_xy = np.array([event.mouseevent.xdata, event.mouseevent.ydata], dtype=float)
            if np.any(~np.isfinite(click_xy)):
                detail_idx = int(event.ind[0])
            else:
                detail_idx = int(event.ind[np.argmin(np.sum((offsets - click_xy[None, :]) ** 2, axis=1))])
        else:
            detail_idx = int(event.ind[0])

        draw_detail_axis(detail_ax, dataset, outputs, detail_idx, yscale=yscale)
        fig_detail.tight_layout()
        fig_detail.canvas.draw_idle()
        print(
            f"Selected sample {detail_idx}: params={dataset.param_values[detail_idx].tolist()} "
            f"alphaD true={dataset.alphaD_values[detail_idx]:.8g} "
            f"pred={outputs['alphaD_opt'][detail_idx]:.8g}"
        )

    cid = fig_alphaD.canvas.mpl_connect("pick_event", on_pick)
    return cid



def plot_parameter_error_map(dataset, alphaD_true, alphaD_opt, max_label_points=60, filter_ranges=None):
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
        ax.set_ylabel(r"Relative error in $lpha_D$")
        ax.set_title(r"No varying parameter detected")
        fig.tight_layout()
        return fig

    # One varying parameter: styled trace with notebook-like markers
    if len(varying) == 1:
        j = varying[0]
        x = param_values[:, j]
        order = np.argsort(x)
        positive_rel_err = np.clip(rel_err[order], 1e-16, None)
        use_lognorm = np.any(positive_rel_err > 0)
        norm = LogNorm(vmin=positive_rel_err.min(), vmax=positive_rel_err.max()) if use_lognorm else None

        fig, ax = plt.subplots(figsize=(7, 4.75))
        sc = ax.scatter(
            x[order], rel_err[order], c=positive_rel_err,
            marker="o", cmap="coolwarm", norm=norm,
            edgecolors="0.3", linewidths=0.6, s=170,
        )
        ax.scatter(x[order], rel_err[order], marker=".", color="black", s=20)
        ax.plot(x[order], rel_err[order], color="0.25", alpha=0.5, linewidth=1.25)
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label(r"Relative error on $lpha_D$")

        if label_points:
            for idx in order:
                ax.text(x[idx], rel_err[idx], str(idx), fontsize=8,
                        ha="center", va="bottom")

        ax.set_xlabel(param_names[j])
        ax.set_ylabel(r"Relative error in $lpha_D$")
        ax.set_title(r"1D parameter-space error trace")
        fig.tight_layout()
        return fig

    # Two or more varying parameters: notebook-style parameter map
    j0, j1 = varying[:2]
    x = param_values[:, j0]
    y = param_values[:, j1]

    positive_rel_err = np.clip(rel_err, 1e-16, None)
    use_lognorm = np.any(positive_rel_err > 0)
    norm = LogNorm(vmin=positive_rel_err.min(), vmax=positive_rel_err.max()) if use_lognorm else None

    fig, ax = plt.subplots(figsize=(6.6, 5.6), dpi=150)
    sc = ax.scatter(
        x, y, c=positive_rel_err,
        marker="o", cmap="coolwarm", norm=norm,
        edgecolors="0.3", linewidths=0.6, s=170,
    )
    ax.scatter(x, y, marker=".", color="black", s=20)

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(r"Relative error on alpha_D")

    if filter_ranges is not None and j0 < len(param_names) and j1 < len(param_names):
        name_x = param_names[j0]
        name_y = param_names[j1]
        if name_x in filter_ranges and name_y in filter_ranges:
            x_lo, x_hi = filter_ranges[name_x]
            y_lo, y_hi = filter_ranges[name_y]

            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            x0 = max(x_lo, xmin)
            x1 = min(x_hi, xmax)
            y0 = max(y_lo, ymin)
            y1 = min(y_hi, ymax)

            if (x1 > x0) and (y1 > y0):
                face_rgba = mcolors.to_rgba('0.6', 0.30)
                fill = patches.Rectangle(
                    (x0, y0), x1 - x0, y1 - y0,
                    facecolor=face_rgba, edgecolor='none', zorder=0.5
                )
                edge = patches.Rectangle(
                    (x0, y0), x1 - x0, y1 - y0,
                    facecolor='none', edgecolor='k', linewidth=2.2, zorder=0.6
                )
                ax.add_patch(fill)
                ax.add_patch(edge)

    if label_points:
        for idx, (xi, yi) in enumerate(zip(x, y)):
            ax.text(xi, yi, str(idx), ha="center", va="center", fontsize=8, color="black")

    ax.set_xlabel(param_names[j0], size=18)
    ax.set_ylabel(param_names[j1], size=18)

    extra = ""
    if len(varying) > 2:
        hidden = ", ".join(param_names[k] for k in varying[2:])
        extra = f" (first two varying dims shown; also varies: {hidden})"
    ax.set_title(r"Parameter-space relative error in alpha_D" + extra)
    fig.tight_layout()
    return fig



def save_figure(fig, outdir, filename, dpi=200):
    ensure_dir(outdir)
    path = os.path.join(outdir, filename)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"Saved: {path}")



def main():
    args = parse_args()
    apply_notebook_plot_style()
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
    print("alpha_D true / pred:", alphaD_true[detail_idx], alphaD_opt[detail_idx])
    print("eta:", float(outputs["eta_batch"][detail_idx]))
    print("Global RMSE:", global_rmse)
    print("Mean sample RMSE:", float(np.mean(per_sample_rmse)))
    print("Median sample RMSE:", float(np.median(per_sample_rmse)))
    print("Max alpha_D rel. err:", float(np.max(np.abs(alphaD_opt - alphaD_true) / np.maximum(np.abs(alphaD_true), 1e-12))))

    fig1 = plot_detail(dataset, outputs, detail_idx, yscale=args.yscale)
    fig2, alphaD_scatter = plot_alphaD_true_vs_pred(
        alphaD_true,
        alphaD_opt,
        label_points=(n_samples <= args.max_label_points),
    )
    alphaD_picker_cid = connect_alphaD_picker(
        fig_alphaD=fig2,
        scatter=alphaD_scatter,
        fig_detail=fig1,
        dataset=dataset,
        outputs=outputs,
        yscale=args.yscale,
    )
    # Keep the connection id alive for interactive backends.
    fig2._alphaD_picker_cid = alphaD_picker_cid
    fig3 = plot_parameter_error_map(
        dataset,
        alphaD_true,
        alphaD_opt,
        max_label_points=args.max_label_points,
        filter_ranges=filter_ranges,
    )

    if args.plots in {"save", "both"}:
        fig_dir = args.fig_dir or os.path.join(args.save_dir, "diagnostics")
        save_standard_figure(fig1, fig_dir, "detail_spectrum.png", dpi=args.dpi)
        labels = DiagnosticLabels(
            observable_name="alphaD",
            observable_true=r"True $\alpha_D$",
            observable_pred=r"Predicted $\alpha_D$",
            relative_error=r"Relative error $\alpha_D$",
            parameter_names=tuple(dataset.param_names),
            prediction_title=r"$\alpha_D$: prediction vs truth",
            parameter_map_title=r"Parameter-space relative error in alpha_D",
        )
        write_standard_observable_diagnostics(
            fig_dir,
            dataset.param_values,
            alphaD_true,
            alphaD_opt,
            labels=labels,
            observable_key="alphaD",
            filter_ranges=filter_ranges,
            max_label_points=args.max_label_points,
            dpi=args.dpi,
            plots=True,
            scatter_alias="alphaD_true_vs_pred.png",
        )

    if args.plots in {"show", "both"}:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
