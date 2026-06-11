#!/usr/bin/env python3
from __future__ import annotations

import csv
import statistics
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from numpy.polynomial.polynomial import Polynomial

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from archive.Beta_decay_package.src import diagnostics_general_gpt as diag
from archive.Beta_decay_package.src import helper_gpt as helper


NUCNAM = "Ni_80"
G_A = 1.2
FIT_ALPHA_RANGE = (0.2, 1.8)
FIT_BETA_RANGE = (0.1, 0.9)
PLOT_ERROR_FLOOR = 1e-6
CONTRAST_STYLES = {
    ("EM1", 8): {"color": "#0047AB", "marker": "o"},
    ("EM1", 13): {"color": "#E66100", "marker": "^"},
    ("EM2", 6): {"color": "#009E73", "marker": "s"},
    ("EM2", 9): {"color": "#CC79A7", "marker": "D"},
}

RUNS = [
    {
        "model": "EM1",
        "n": 8,
        "retain": 0.9,
        "run_dir": ROOT / "Beta_decay_package/runs_em1/n8_retain0p9_w1p0_seed42",
        "params": "params_best_n8_retain0.9.txt",
    },
    {
        "model": "EM1",
        "n": 13,
        "retain": 0.9,
        "run_dir": ROOT / "Beta_decay_package/runs_em1/n13_retain0p9_w1p0_seed42",
        "params": "params_best_n13_retain0.9.txt",
    },
    {
        "model": "EM2",
        "n": 6,
        "run_dir": ROOT / "Beta_decay_package/runs_em2/n6_seed42",
        "params": "params_6_only_HL.txt",
    },
    {
        "model": "EM2",
        "n": 9,
        "run_dir": ROOT / "Beta_decay_package/runs_em2/n9_seed42",
        "params": "params_9_only_HL.txt",
    },
]


def benchmark_once(fn, repeats: int = 7, warmups: int = 2) -> tuple[float, object]:
    value = None
    for _ in range(warmups):
        value = fn()

    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        value = fn()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples), value


def point_region(point) -> str:
    alpha = float(point[0])
    beta = float(point[1])
    if FIT_ALPHA_RANGE[0] <= alpha <= FIT_ALPHA_RANGE[1] and FIT_BETA_RANGE[0] <= beta <= FIT_BETA_RANGE[1]:
        return "fit region"
    return "test/extrapolation region"


def relative_l2_error(pred, true, energy) -> float:
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    energy = np.asarray(energy, dtype=float)
    numer = np.trapezoid((pred - true) ** 2, energy)
    denom = max(float(np.trapezoid(true**2, energy)), 1e-16)
    return float(np.sqrt(numer / denom))


def half_life_relative_error(pred: float, true: float) -> float:
    return abs(float(pred) - float(true)) / max(abs(float(true)), 1e-12)


def em1_prediction_fn(params, n, retain, point, central_point, x_grid, coeffs, g_A):
    D, S1, S2, v0, eta, x1, x2, x3 = helper.modified_DS(params, n)
    alpha = float(point[0])
    beta = float(point[1])

    def predict():
        matrix = D + (alpha - float(central_point[0])) * S1 + (beta - float(central_point[1])) * S2
        eigenvalues, eigenvectors = tf.linalg.eigh(matrix)
        n_i = int(eigenvalues.shape[0])
        k_keep = max(1, min(int(round(retain * n_i)), n_i))
        left = (n_i - k_keep) // 2
        right = left + k_keep
        eigenvalues_kept = eigenvalues[left:right]
        eigenvectors_kept = eigenvectors[:, left:right]
        projections = tf.linalg.matvec(tf.transpose(eigenvectors_kept), v0)
        strengths = tf.square(projections)
        mask = tf.cast((eigenvalues_kept > -10) & (eigenvalues_kept < 15), dtype=tf.float64)
        strengths = strengths * mask
        width = tf.sqrt(tf.square(eta) + tf.square(x1 + x2 * alpha + x3 * beta))
        spectrum = helper.give_me_Lorentzian(x_grid, eigenvalues_kept, strengths, width)
        half_life = helper.half_life_loss(eigenvalues_kept, strengths, coeffs, g_A)
        return spectrum.numpy(), float(half_life.numpy())

    return predict


def em2_prediction_fn(params, n, point, central_point):
    D, S1, S2 = helper.modified_DS_only_HL(params, n)
    alpha = float(point[0])
    beta = float(point[1])

    def predict():
        matrix = D + (alpha - float(central_point[0])) * S1 + (beta - float(central_point[1])) * S2
        eigenvalues, _ = tf.linalg.eigh(matrix)
        return float((10 ** eigenvalues[int(n / 2)]).numpy())

    return predict


def high_fidelity_reference_fn(point, coeffs):
    def evaluate_reference():
        spectra, half_lives = helper.data_table([point], coeffs, G_A, NUCNAM)
        return spectra[0], float(half_lives[0].numpy())

    return evaluate_reference


def reference_tables(points, coeffs):
    true_spectra, true_half_lives_tf = helper.data_table(points, coeffs, G_A, NUCNAM)
    true_half_lives = [float(value.numpy()) for value in true_half_lives_tf]
    return true_spectra, true_half_lives


def rows_for_high_fidelity(points, coeffs) -> list[dict]:
    rows = []
    for idx, point in enumerate(points):
        elapsed, _ = benchmark_once(high_fidelity_reference_fn(point, coeffs), repeats=3, warmups=1)
        rows.append(
            {
                "model": "High fidelity",
                "n": "",
                "retain": "",
                "region": point_region(point),
                "idx": idx,
                "V0_is": float(point[0]),
                "g0": float(point[1]),
                "time_s": elapsed,
                "time_ms": 1000.0 * elapsed,
                "max_relative_observable_error": 0.0,
                "spectrum_relative_l2_error": 0.0,
                "spectrum_pointwise_max_relative_error": 0.0,
                "half_life_relative_error": 0.0,
            }
        )
    return rows


def rows_for_run(run: dict, points, true_spectra, true_half_lives, coeffs) -> list[dict]:
    run_dir = run["run_dir"]
    training_points = diag.load_train_set(run_dir / "train_set.txt")
    central_point = diag.central_point_from(training_points)
    params = np.loadtxt(run_dir / run["params"])

    rows = []
    if run["model"] == "EM1":
        for idx, point in enumerate(points):
            x_grid = tf.constant(true_spectra[idx][:, 0], dtype=tf.float64)
            true_spectrum = true_spectra[idx][:, 1]
            predict = em1_prediction_fn(params, run["n"], run["retain"], point, central_point, x_grid, coeffs, G_A)
            elapsed, (pred_spectrum, pred_half_life) = benchmark_once(predict)
            spectrum_l2 = relative_l2_error(pred_spectrum, true_spectrum, true_spectra[idx][:, 0])
            spectrum_max = float(
                np.max(np.abs(np.asarray(pred_spectrum) - np.asarray(true_spectrum)))
                / max(float(np.max(np.abs(true_spectrum))), 1e-12)
            )
            hl_error = half_life_relative_error(pred_half_life, true_half_lives[idx])
            rows.append(
                {
                    "model": run["model"],
                    "n": run["n"],
                    "retain": run["retain"],
                    "region": point_region(point),
                    "idx": idx,
                    "V0_is": float(point[0]),
                    "g0": float(point[1]),
                    "time_s": elapsed,
                    "time_ms": 1000.0 * elapsed,
                    "max_relative_observable_error": max(spectrum_l2, hl_error),
                    "spectrum_relative_l2_error": spectrum_l2,
                    "spectrum_pointwise_max_relative_error": spectrum_max,
                    "half_life_relative_error": hl_error,
                }
            )
    else:
        for idx, point in enumerate(points):
            predict = em2_prediction_fn(params, run["n"], point, central_point)
            elapsed, pred_half_life = benchmark_once(predict)
            hl_error = half_life_relative_error(pred_half_life, true_half_lives[idx])
            rows.append(
                {
                    "model": run["model"],
                    "n": run["n"],
                    "retain": "",
                    "region": point_region(point),
                    "idx": idx,
                    "V0_is": float(point[0]),
                    "g0": float(point[1]),
                    "time_s": elapsed,
                    "time_ms": 1000.0 * elapsed,
                    "max_relative_observable_error": hl_error,
                    "spectrum_relative_l2_error": "",
                    "spectrum_pointwise_max_relative_error": "",
                    "half_life_relative_error": hl_error,
                }
            )
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    fieldnames = [
        "model",
        "n",
        "retain",
        "region",
        "idx",
        "V0_is",
        "g0",
        "time_s",
        "time_ms",
        "max_relative_observable_error",
        "spectrum_relative_l2_error",
        "spectrum_pointwise_max_relative_error",
        "half_life_relative_error",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(rows: list[dict], path: Path) -> None:
    fieldnames = [
        "model",
        "n",
        "region",
        "count",
        "median_time_ms",
        "min_time_ms",
        "max_time_ms",
        "median_max_relative_observable_error",
        "max_max_relative_observable_error",
        "median_spectrum_relative_l2_error",
        "median_half_life_relative_error",
        "max_half_life_relative_error",
    ]
    grouped: dict[tuple[str, str, str], list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["model"], str(row["n"]), row["region"]), []).append(row)

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for (model, n, region), group in sorted(grouped.items()):
            times = [row["time_ms"] for row in group]
            errs = [row["max_relative_observable_error"] for row in group]
            spectrum_l2 = [
                row["spectrum_relative_l2_error"]
                for row in group
                if isinstance(row["spectrum_relative_l2_error"], float)
            ]
            hl_errs = [
                row["half_life_relative_error"]
                for row in group
                if isinstance(row["half_life_relative_error"], float)
            ]
            writer.writerow(
                {
                    "model": model,
                    "n": n,
                    "region": region,
                    "count": len(group),
                    "median_time_ms": statistics.median(times),
                    "min_time_ms": min(times),
                    "max_time_ms": max(times),
                    "median_max_relative_observable_error": statistics.median(errs),
                    "max_max_relative_observable_error": max(errs),
                    "median_spectrum_relative_l2_error": statistics.median(spectrum_l2) if spectrum_l2 else "",
                    "median_half_life_relative_error": statistics.median(hl_errs) if hl_errs else "",
                    "max_half_life_relative_error": max(hl_errs) if hl_errs else "",
                }
            )


def plot_cat(rows: list[dict], out_base: Path) -> None:
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.8,
        }
    )
    fig, ax = plt.subplots(figsize=(8.6, 4.8), dpi=220)
    styles = CONTRAST_STYLES

    for key, style in styles.items():
        model, n = key
        for region, alpha, face in [
            ("fit region", 0.72, style["color"]),
            ("test/extrapolation region", 0.82, "none"),
        ]:
            group = [row for row in rows if row["model"] == model and row["n"] == n and row["region"] == region]
            if not group:
                continue
            ax.scatter(
                [row["time_ms"] for row in group],
                [100.0 * max(row["max_relative_observable_error"], PLOT_ERROR_FLOOR) for row in group],
                s=24,
                alpha=alpha,
                linewidths=0.75,
                edgecolors=style["color"],
                facecolors=face,
                label=f"{model}, n={n}" if region == "fit region" else None,
                marker=style["marker"],
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Time per Ni80 evaluation (ms)")
    ax.set_ylabel("Native relative error (%)")
    ax.set_title("Ni80 Computational Accuracy vs Time")

    model_legend = ax.legend(
        frameon=False,
        ncol=1,
        columnspacing=1.0,
        handletextpad=0.35,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.2,
    )
    ax.add_artist(model_legend)
    region_handles = [
        mlines.Line2D([], [], color="#444444", marker="o", linestyle="None", markerfacecolor="#444444", label="fit region"),
        mlines.Line2D(
            [],
            [],
            color="#444444",
            marker="o",
            linestyle="None",
            markerfacecolor="none",
            label="test/extrapolation region",
        ),
    ]
    ax.legend(handles=region_handles, frameon=False, loc="lower left", bbox_to_anchor=(1.01, 0.0), borderaxespad=0.2)

    fig.tight_layout(rect=(0, 0, 0.76, 1))
    fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_half_life_cat(rows: list[dict], out_base: Path) -> None:
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.8,
        }
    )
    fig, ax = plt.subplots(figsize=(8.6, 4.8), dpi=220)

    for key, style in CONTRAST_STYLES.items():
        model, n = key
        for region, alpha, face in [
            ("fit region", 0.72, style["color"]),
            ("test/extrapolation region", 0.82, "none"),
        ]:
            group = [
                row
                for row in rows
                if row["model"] == model
                and row["n"] == n
                and row["region"] == region
                and isinstance(row["half_life_relative_error"], float)
            ]
            if not group:
                continue
            ax.scatter(
                [row["time_ms"] for row in group],
                [100.0 * max(row["half_life_relative_error"], PLOT_ERROR_FLOOR) for row in group],
                s=24,
                alpha=alpha,
                linewidths=0.75,
                edgecolors=style["color"],
                facecolors=face,
                label=f"{model}, n={n}" if region == "fit region" else None,
                marker=style["marker"],
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Time per Ni80 evaluation (ms)")
    ax.set_ylabel("Half-life relative error (%)")
    ax.set_title("Ni80 CAT: Half-Life Error Only")

    model_legend = ax.legend(
        frameon=False,
        ncol=1,
        columnspacing=1.0,
        handletextpad=0.35,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.2,
    )
    ax.add_artist(model_legend)
    region_handles = [
        mlines.Line2D([], [], color="#444444", marker="o", linestyle="None", markerfacecolor="#444444", label="fit region"),
        mlines.Line2D(
            [],
            [],
            color="#444444",
            marker="o",
            linestyle="None",
            markerfacecolor="none",
            label="test/extrapolation region",
        ),
    ]
    ax.legend(handles=region_handles, frameon=False, loc="lower left", bbox_to_anchor=(1.01, 0.0), borderaxespad=0.2)

    fig.tight_layout(rect=(0, 0, 0.76, 1))
    fig.savefig(out_base.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    all_points = diag.load_all_points(NUCNAM)
    coeffs_np = Polynomial(helper.fit_phase_space(0, 28, 80, 15)).coef
    coeffs = tf.constant(coeffs_np, dtype=tf.float64)
    true_spectra, true_half_lives = reference_tables(all_points, coeffs)

    rows = []
    for run in RUNS:
        rows.extend(rows_for_run(run, all_points, true_spectra, true_half_lives, coeffs))

    write_csv(rows, out_dir / "ni80_cat_data.csv")
    write_summary(rows, out_dir / "ni80_cat_summary.csv")
    plot_cat(rows, out_dir / "ni80_cat_plot")
    plot_half_life_cat(rows, out_dir / "ni80_cat_half_life_only")
    print(f"Wrote {out_dir / 'ni80_cat_data.csv'}")
    print(f"Wrote {out_dir / 'ni80_cat_summary.csv'}")
    print(f"Wrote {out_dir / 'ni80_cat_plot.png'}")
    print(f"Wrote {out_dir / 'ni80_cat_plot.pdf'}")
    print(f"Wrote {out_dir / 'ni80_cat_half_life_only.png'}")
    print(f"Wrote {out_dir / 'ni80_cat_half_life_only.pdf'}")


if __name__ == "__main__":
    main()
