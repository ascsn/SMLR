#!/usr/bin/env python
"""Fit independent Lorentzian snapshots for QRPA strength compression.

This script is the shared preprocessing stage for experiments that learn smooth
reduced coordinates after independent per-spectrum compression. It mirrors the
workflow from ``tests/representation_compression_error_Yb.ipynb`` while making
the cache and snapshot products reusable from later scripts.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "smlr_mplconfig"))
SRC_DIR = ROOT / "src"
DIPOLE_SRC_DIR = ROOT / "Dipole_polarizability" / "src"
for path in (str(SRC_DIR), str(DIPOLE_SRC_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

import helper_gpt  # noqa: E402


DEFAULT_STRENGTH_REGEX = r"strength_(?P<p2>[0-9.]+)_(?P<p1>[0-9.]+)\.out"
DEFAULT_ALPHAD_REGEX = r"alphaD_(?P<p2>[0-9.]+)_(?P<p1>[0-9.]+)\.out"
METRIC_COLUMNS = (
    "sample_index",
    "n_lorentz",
    "elapsed_seconds",
    "p1",
    "p2",
    "sample_id",
    "rmse",
    "weighted_rmse",
    "mae",
    "max_abs",
    "rel_l2",
    "nrmse_peak",
    "nrmse_range",
    "r2",
    "alphaD_true",
    "alphaD_true_from_strength",
    "alphaD_hat",
    "alphaD_abs_error",
    "alphaD_relerr",
    "alphaD_abs_error_from_strength",
    "alphaD_relerr_from_strength",
)


@dataclass
class LoadedData:
    dataset: object
    omega: np.ndarray
    strength_matrix: np.ndarray
    omega_weights: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit independent Lorentzian compressions and assemble n-star snapshots."
    )
    parser.add_argument("--strength-dir", type=Path, default=ROOT / "dipole_polarizability_160Yb" / "total_strength")
    parser.add_argument("--alphaD-dir", type=Path, default=ROOT / "dipole_polarizability_160Yb" / "total_alphaD")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "tests" / "cache" / "yb_lorentzian_compression")
    parser.add_argument("--output-dir", type=Path, default=None, help="Defaults to --cache-dir.")
    parser.add_argument("--strength-regex", default=DEFAULT_STRENGTH_REGEX)
    parser.add_argument("--alphaD-regex", default=DEFAULT_ALPHAD_REGEX)
    parser.add_argument("--n-min", type=int, default=2)
    parser.add_argument("--n-max", type=int, default=30)
    parser.add_argument("--eta", type=float, default=2.0)
    parser.add_argument("--min-spacing", type=float, default=0.01)
    parser.add_argument("--grid-m", type=int, default=None)
    parser.add_argument("--sample-limit", type=int, default=None)
    parser.add_argument("--force-refit", action="store_true")
    parser.add_argument("--n-star", type=int, default=None, help="Override automatic saturation selection.")
    parser.add_argument(
        "--saturation-window",
        type=int,
        default=3,
        help="Number of subsequent n values used to declare RMSE saturation.",
    )
    parser.add_argument(
        "--saturation-rel-improvement",
        type=float,
        default=0.02,
        help="Median RMSE must improve by less than this fraction over the window.",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.0,
        help="Optional deterministic validation subset fraction for n-star and plot metrics. Default uses all spectra.",
    )
    parser.add_argument("--validation-seed", type=int, default=1234)
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def trapz_weights(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    dx = np.diff(x)
    if dx.size == 0:
        raise ValueError("omega grid must contain at least two points")
    weights = np.empty_like(x)
    weights[0] = dx[0] / 2.0
    weights[-1] = dx[-1] / 2.0
    weights[1:-1] = (x[2:] - x[:-2]) / 2.0
    return weights


def load_strength_data(args: argparse.Namespace) -> LoadedData:
    filter_ranges = {"p1": [0.4, 1.8], "p2": [1.5, 4.0]}
    dataset = helper_gpt.load_dataset(
        strength_dir=str(args.strength_dir),
        alphaD_dir=str(args.alphaD_dir) if args.alphaD_dir else None,
        strength_regex=args.strength_regex,
        alphaD_regex=args.alphaD_regex,
        filter_ranges=filter_ranges,
    )

    if args.sample_limit is not None:
        keep = np.arange(min(int(args.sample_limit), len(dataset.strengths)))
        dataset.strengths = [dataset.strengths[i] for i in keep]
        dataset.sample_ids = [dataset.sample_ids[i] for i in keep]
        dataset.param_values = dataset.param_values[keep]
        dataset.alphaD_values = dataset.alphaD_values[keep]
        dataset.alphaD_raw = [dataset.alphaD_raw[i] for i in keep]

    omega = np.asarray(dataset.strengths[0][:, 0], dtype=np.float32)
    for i, strength in enumerate(dataset.strengths[1:], start=1):
        if strength.shape[0] != omega.size or not np.allclose(strength[:, 0], omega):
            raise ValueError(f"Spectrum {i} does not share the common omega grid.")

    strength_matrix = np.stack([np.asarray(s[:, 1], dtype=np.float32) for s in dataset.strengths], axis=0)
    return LoadedData(dataset=dataset, omega=omega, strength_matrix=strength_matrix, omega_weights=trapz_weights(omega))


def alphaD_from_strength(omega: np.ndarray, y: np.ndarray) -> float:
    return float(helper_gpt.ALPHAD_FAC * np.trapezoid(y / np.maximum(omega, 1e-6), omega))


def score_fit(
    omega: np.ndarray,
    weights: np.ndarray,
    y: np.ndarray,
    yhat: np.ndarray,
    alphaD_true: float,
) -> dict[str, float]:
    y = np.asarray(y, dtype=np.float64)
    yhat = np.asarray(yhat, dtype=np.float64)
    residual = yhat - y
    rmse = float(np.sqrt(np.mean(residual**2)))
    weighted_rmse = float(np.sqrt(np.sum(weights * residual**2) / np.sum(weights)))
    l2_true = float(np.sqrt(np.sum(weights * y**2)))
    denom_var = float(np.sum((y - np.mean(y)) ** 2))
    alphaD_true_from_strength = alphaD_from_strength(omega, y)
    alphaD_hat = alphaD_from_strength(omega, yhat)
    return {
        "rmse": rmse,
        "weighted_rmse": weighted_rmse,
        "mae": float(np.mean(np.abs(residual))),
        "max_abs": float(np.max(np.abs(residual))),
        "rel_l2": float(np.sqrt(np.sum(weights * residual**2)) / max(l2_true, 1e-12)),
        "nrmse_peak": float(rmse / max(np.max(np.abs(y)), 1e-12)),
        "nrmse_range": float(rmse / max(np.ptp(y), 1e-12)),
        "r2": float(1.0 - np.sum(residual**2) / denom_var) if denom_var > 0 else np.nan,
        "alphaD_true": float(alphaD_true),
        "alphaD_true_from_strength": alphaD_true_from_strength,
        "alphaD_hat": alphaD_hat,
        "alphaD_abs_error": float(abs(alphaD_hat - alphaD_true)),
        "alphaD_relerr": float(abs(alphaD_hat - alphaD_true) / max(abs(alphaD_true), 1e-12)),
        "alphaD_abs_error_from_strength": float(abs(alphaD_hat - alphaD_true_from_strength)),
        "alphaD_relerr_from_strength": float(
            abs(alphaD_hat - alphaD_true_from_strength) / max(abs(alphaD_true_from_strength), 1e-12)
        ),
    }


def sample_cache_path(cache_dir: Path, dataset: object, eta: float, min_spacing: float, sample_index: int, n_lorentz: int) -> Path:
    sid = "_".join(dataset.sample_ids[sample_index])
    n_dir = cache_dir / f"n{n_lorentz:02d}"
    return n_dir / f"sample{sample_index:03d}_{sid}_n{n_lorentz:02d}_eta{eta:g}_spacing{min_spacing:g}.npz"


def fit_one(
    args: argparse.Namespace,
    loaded: LoadedData,
    sample_index: int,
    n_lorentz: int,
) -> tuple[dict[str, float | int | str], np.ndarray, np.ndarray, np.ndarray]:
    path = sample_cache_path(args.cache_dir, loaded.dataset, args.eta, args.min_spacing, sample_index, n_lorentz)
    y = loaded.strength_matrix[sample_index]
    if path.exists() and not args.force_refit:
        packed = np.load(path, allow_pickle=False)
        E = np.asarray(packed["E"], dtype=np.float32)
        B = np.asarray(packed["B"], dtype=np.float32)
        yhat = np.asarray(packed["yhat"], dtype=np.float32)
        elapsed = float(packed.get("elapsed_seconds", np.array(np.nan)))
    else:
        start = time.perf_counter()
        E, B, yhat = helper_gpt.fit_strength_with_tf_lorentzian(
            loaded.omega,
            y,
            int(n_lorentz),
            float(args.eta),
            grid_M=args.grid_m,
            min_spacing=float(args.min_spacing),
        )
        elapsed = time.perf_counter() - start
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, E=E, B=B, yhat=yhat, elapsed_seconds=np.array(elapsed))

    scores = score_fit(
        loaded.omega,
        loaded.omega_weights,
        y,
        yhat,
        alphaD_true=float(loaded.dataset.alphaD_values[sample_index]),
    )
    row = {
        "sample_index": int(sample_index),
        "n_lorentz": int(n_lorentz),
        "elapsed_seconds": elapsed,
        "sample_id": "_".join(loaded.dataset.sample_ids[sample_index]),
        **{name: float(loaded.dataset.param_values[sample_index, j]) for j, name in enumerate(loaded.dataset.param_names)},
        **scores,
    }
    return row, E, B, yhat


def validation_indices(n_samples: int, fraction: float, seed: int) -> np.ndarray:
    if fraction <= 0.0 or fraction >= 1.0:
        return np.arange(n_samples)
    rng = np.random.default_rng(seed)
    n_val = max(1, int(round(fraction * n_samples)))
    return np.sort(rng.choice(n_samples, size=n_val, replace=False))


def summarize_rows(rows: Sequence[dict[str, object]], val_idx: np.ndarray) -> list[dict[str, float | int]]:
    val_set = {int(i) for i in val_idx}
    n_values = sorted({int(row["n_lorentz"]) for row in rows})
    summary = []
    for n_lorentz in n_values:
        selected = [row for row in rows if int(row["n_lorentz"]) == n_lorentz and int(row["sample_index"]) in val_set]
        rmse = np.asarray([float(row["rmse"]) for row in selected], dtype=float)
        rel_l2 = np.asarray([float(row["rel_l2"]) for row in selected], dtype=float)
        r2 = np.asarray([float(row["r2"]) for row in selected], dtype=float)
        alphaD_relerr = np.asarray([float(row["alphaD_relerr"]) for row in selected], dtype=float)
        summary.append(
            {
                "n_lorentz": int(n_lorentz),
                "spectra": int(len(selected)),
                "rmse_mean": float(np.mean(rmse)),
                "rmse_median": float(np.median(rmse)),
                "rmse_p90": float(np.quantile(rmse, 0.90)),
                "rel_l2_mean": float(np.mean(rel_l2)),
                "rel_l2_median": float(np.median(rel_l2)),
                "rel_l2_p90": float(np.quantile(rel_l2, 0.90)),
                "r2_median": float(np.median(r2)),
                "alphaD_relerr_median": float(np.median(alphaD_relerr)),
            }
        )
    return summary


def select_n_star(
    summary: Sequence[dict[str, float | int]],
    explicit_n_star: int | None,
    window: int,
    rel_improvement: float,
) -> int:
    if explicit_n_star is not None:
        return int(explicit_n_star)
    n_values = np.asarray([int(row["n_lorentz"]) for row in summary], dtype=int)
    rmse = np.asarray([float(row["rmse_median"]) for row in summary], dtype=float)
    if len(n_values) <= 1:
        return int(n_values[0])
    window = max(1, int(window))
    for i, n_lorentz in enumerate(n_values):
        j = min(i + window, len(rmse) - 1)
        if j == i:
            break
        improvement = (rmse[i] - rmse[j]) / max(rmse[i], 1e-12)
        if improvement <= rel_improvement:
            return int(n_lorentz)
    return int(n_values[-1])


def write_csv(path: Path, rows: Sequence[dict[str, object]], columns: Sequence[str] | None = None) -> None:
    if not rows:
        raise ValueError(f"no rows to write to {path}")
    columns = tuple(columns or rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(output_dir: Path, summary: Sequence[dict[str, object]], n_star: int, validation_label: str) -> Path:
    import matplotlib.pyplot as plt

    n_values = np.asarray([int(row["n_lorentz"]) for row in summary])
    rmse_median = np.asarray([float(row["rmse_median"]) for row in summary])
    rmse_p90 = np.asarray([float(row["rmse_p90"]) for row in summary])
    rel_l2_median = np.asarray([float(row["rel_l2_median"]) for row in summary])
    rel_l2_p90 = np.asarray([float(row["rel_l2_p90"]) for row in summary])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].plot(n_values, rmse_median, marker="o", label="median")
    axes[0].plot(n_values, rmse_p90, marker="s", label="p90")
    axes[0].axvline(n_star, color="black", lw=1, ls="--", label=f"n*={n_star}")
    axes[0].set_xlabel("number of Lorentzians")
    axes[0].set_ylabel("RMSE")
    axes[0].set_title(f"{validation_label} RMSE")
    axes[0].legend()

    axes[1].plot(n_values, rel_l2_median, marker="o", label="median")
    axes[1].plot(n_values, rel_l2_p90, marker="s", label="p90")
    axes[1].axvline(n_star, color="black", lw=1, ls="--", label=f"n*={n_star}")
    axes[1].set_xlabel("number of Lorentzians")
    axes[1].set_ylabel("relative integrated L2")
    axes[1].set_title(f"{validation_label} relative error")
    axes[1].legend()

    path = output_dir / "lorentzian_compression_rmse_vs_n.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def main() -> None:
    args = parse_args()
    if args.n_min < 1 or args.n_max < args.n_min:
        raise ValueError("Require 1 <= --n-min <= --n-max.")

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    output_dir = args.output_dir or args.cache_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    loaded = load_strength_data(args)
    n_values = list(range(args.n_min, args.n_max + 1))
    n_samples, n_grid = loaded.strength_matrix.shape
    print(f"Loaded {n_samples} spectra on {n_grid} omega points.")
    print(f"Fitting/scoring n={args.n_min}..{args.n_max}; cache: {args.cache_dir}")

    E_by_n = np.full((n_samples, len(n_values), args.n_max), np.nan, dtype=np.float32)
    B_by_n = np.full_like(E_by_n, np.nan)
    S_fit_by_n = np.empty((n_samples, len(n_values), n_grid), dtype=np.float32)
    rmse_by_n = np.empty((n_samples, len(n_values)), dtype=np.float64)
    alphaD_relerr_by_n = np.empty_like(rmse_by_n)

    rows: list[dict[str, object]] = []
    total = n_samples * len(n_values)
    start = time.perf_counter()
    for n_pos, n_lorentz in enumerate(n_values):
        for sample_index in range(n_samples):
            row, E, B, yhat = fit_one(args, loaded, sample_index, n_lorentz)
            rows.append(row)
            E_by_n[sample_index, n_pos, :n_lorentz] = E
            B_by_n[sample_index, n_pos, :n_lorentz] = B
            S_fit_by_n[sample_index, n_pos] = yhat
            rmse_by_n[sample_index, n_pos] = float(row["rmse"])
            alphaD_relerr_by_n[sample_index, n_pos] = float(row["alphaD_relerr"])
            count = len(rows)
            if count == 1 or count % 50 == 0 or count == total:
                print(f"{count:4d}/{total} fits scored after {time.perf_counter() - start:7.1f} s", end="\r")
    print()

    val_idx = validation_indices(n_samples, args.validation_fraction, args.validation_seed)
    summary = summarize_rows(rows, val_idx)
    n_star = select_n_star(summary, args.n_star, args.saturation_window, args.saturation_rel_improvement)
    if n_star not in n_values:
        raise ValueError(f"n_star={n_star} is outside fitted range {args.n_min}..{args.n_max}.")
    n_star_pos = n_values.index(n_star)

    metrics_csv = output_dir / f"fit_metrics_n{args.n_min}_to_n{args.n_max}_eta{args.eta:g}_spacing{args.min_spacing:g}.csv"
    summary_csv = output_dir / f"summary_n{args.n_min}_to_n{args.n_max}_eta{args.eta:g}_spacing{args.min_spacing:g}.csv"
    write_csv(metrics_csv, rows, METRIC_COLUMNS)
    write_csv(summary_csv, summary)

    sweep_npz = output_dir / f"lorentzian_sweep_n{args.n_min}_to_n{args.n_max}_eta{args.eta:g}_spacing{args.min_spacing:g}.npz"
    np.savez_compressed(
        sweep_npz,
        alpha_points=np.asarray(loaded.dataset.param_values, dtype=np.float32),
        param_names=np.asarray(loaded.dataset.param_names),
        sample_ids=np.asarray(["_".join(sid) for sid in loaded.dataset.sample_ids]),
        omega=loaded.omega,
        S_true=loaded.strength_matrix,
        n_values=np.asarray(n_values, dtype=np.int32),
        eta=np.asarray(args.eta, dtype=np.float32),
        E_by_n=E_by_n,
        B_by_n=B_by_n,
        S_fit_by_n=S_fit_by_n,
        rmse_by_n=rmse_by_n,
        alphaD_relerr_by_n=alphaD_relerr_by_n,
        validation_indices=val_idx,
    )

    snapshot_npz = output_dir / "snapshots_nstar.npz"
    np.savez_compressed(
        snapshot_npz,
        alpha_points=np.asarray(loaded.dataset.param_values, dtype=np.float32),
        param_names=np.asarray(loaded.dataset.param_names),
        sample_ids=np.asarray(["_".join(sid) for sid in loaded.dataset.sample_ids]),
        omega=loaded.omega,
        S_true=loaded.strength_matrix,
        eta=np.asarray(args.eta, dtype=np.float32),
        n_star=np.asarray(n_star, dtype=np.int32),
        E_raw=E_by_n[:, n_star_pos, :n_star],
        B_raw=B_by_n[:, n_star_pos, :n_star],
        S_fit=S_fit_by_n[:, n_star_pos],
        rmse=rmse_by_n[:, n_star_pos],
        alphaD_relerr=alphaD_relerr_by_n[:, n_star_pos],
        validation_indices=val_idx,
    )

    plot_path = None
    if not args.no_plot:
        validation_label = "validation" if 0.0 < args.validation_fraction < 1.0 else "all-spectra"
        plot_path = plot_summary(output_dir, summary, n_star, validation_label)

    print(f"Selected n_star={n_star}")
    print(f"Saved metrics: {metrics_csv}")
    print(f"Saved summary: {summary_csv}")
    print(f"Saved sweep arrays: {sweep_npz}")
    print(f"Saved snapshots: {snapshot_npz}")
    if plot_path is not None:
        print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()
