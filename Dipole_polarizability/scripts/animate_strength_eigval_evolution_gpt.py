#!/usr/bin/env python3
"""Reusable EM1 diagnostics for strength, parameter errors, and eigenvalue sweeps.

The original version of this file was a hard-coded 160Yb animation script.  This
version is importable from notebooks and works with the EM1 run directories under
``runs_em1`` for both dipole-style 2D runs and beta/strength-only 4D runs.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import plot_em1_strength_predictions as em1  # noqa: E402


ALPHAD_FAC = 8.0 * np.pi * (7.29735e-3) * 197.33 / 9.0


@dataclass
class EM1Run:
    run_dir: Path
    data_dir: Path
    params: np.ndarray
    metadata: dict
    model_family: str
    retain: float
    combined: list[tuple[tuple[float, ...], str]]
    split_entries: dict[str, list[tuple[tuple[float, ...], str]]]
    n: int | None = None
    central_point: tuple[float, ...] | None = None
    coordinate_scales: tuple[float, ...] | None = None


def _key(entry) -> tuple[float, ...]:
    return em1.rounded_param_key(em1.helper.dataset_entry_params(entry))


def _relative_l2(x: np.ndarray, y_pred: np.ndarray, y_true: np.ndarray) -> float:
    denom = np.trapezoid(y_true**2, x) + 1.0e-16
    return float(np.trapezoid((y_pred - y_true) ** 2, x) / denom)


def _alphaD_from_strength(x: np.ndarray, y: np.ndarray) -> float:
    return float(ALPHAD_FAC * np.trapezoid(y / np.maximum(x, 1.0e-6), x))


def _m0_from_strength(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.trapezoid(y, x))


def _split_entries(run_dir: Path, data_dir: Path) -> dict[str, list[tuple[tuple[float, ...], str]]]:
    splits = {}
    for name in ("train", "validation", "test"):
        entries = em1.load_saved_split(run_dir, data_dir, name)
        if entries:
            splits[name] = entries
    return splits


def load_em1_run(
    run_dir: str | Path,
    *,
    data_dir: str | Path | None = None,
    params_path: str | Path | None = None,
    retain: float | None = None,
    model_family: str = "auto",
    reference_index: int = 0,
    no_coordinate_normalization: bool = False,
) -> EM1Run:
    """Load one saved EM1 run and infer its dataset/model layout."""

    run_dir = Path(run_dir)
    metadata = em1.load_run_metadata(run_dir)
    inferred_data_dir = em1.infer_data_dir(run_dir, Path(data_dir) if data_dir else None)
    family = em1.infer_model_family(run_dir, metadata, model_family)
    params = np.loadtxt(Path(params_path) if params_path else em1.default_params_file(run_dir))
    keep = float(retain if retain is not None else (metadata.get("retain") or 1.0))
    combined = em1.load_strength_dataset(inferred_data_dir, metadata.get("strength_regex"))
    splits = _split_entries(run_dir, inferred_data_dir)

    n = None
    central_point = None
    coordinate_scales = None
    if family == "beta":
        num_components = len(em1.helper.dataset_entry_params(combined[0]))
        n = em1.infer_n(len(params), num_components)
        combined_ar = np.asarray([em1.helper.dataset_entry_params(entry) for entry in combined], dtype=float)
        central_point = (
            tuple(float(v) for v in metadata["central_point"])
            if metadata.get("central_point") is not None
            else em1.helper.dataset_entry_params(combined[max(0, min(reference_index, len(combined) - 1))])
        )
        if not no_coordinate_normalization:
            ranges = np.ptp(combined_ar, axis=0)
            coordinate_scales = tuple(float(v if v > 0 else 1.0) for v in ranges)

    return EM1Run(
        run_dir=run_dir,
        data_dir=inferred_data_dir,
        params=params,
        metadata=metadata,
        model_family=family,
        retain=keep,
        combined=combined,
        split_entries=splits,
        n=n,
        central_point=central_point,
        coordinate_scales=coordinate_scales,
    )


def predict_entry(run: EM1Run, entry) -> dict:
    """Predict one point and return curves, poles, strengths, and errors."""

    lor_true = np.loadtxt(em1.helper.dataset_entry_path(entry))
    if run.model_family == "dipole":
        x, y_pred, y_true, poles, strengths = em1.predict_dipole(
            params=run.params,
            metadata=run.metadata,
            retain=run.retain,
            entry=entry,
            lor_true=lor_true,
        )
        obs_true = _alphaD_from_strength(x, y_true)
        obs_pred = _alphaD_from_strength(x, y_pred)
        observable_name = r"$\alpha_D$"
    else:
        x, y_pred, y_true, poles, strengths = em1.predict(
            params=run.params,
            n=run.n,
            num_components=len(em1.helper.dataset_entry_params(entry)),
            retain=run.retain,
            entry=entry,
            lor_true=lor_true,
            central_point=run.central_point,
            coordinate_scales=run.coordinate_scales,
            fixed_width=None,
        )
        obs_true = _m0_from_strength(x, y_true)
        obs_pred = _m0_from_strength(x, y_pred)
        observable_name = r"$m_0=\int S(E)dE$"

    rel_obs = abs(obs_pred - obs_true) / max(abs(obs_true), 1.0e-12)
    return {
        "params": em1.helper.dataset_entry_params(entry),
        "x": x,
        "y_true": y_true,
        "y_pred": y_pred,
        "poles": poles,
        "strengths": strengths,
        "observable_true": obs_true,
        "observable_pred": obs_pred,
        "observable_relerr": float(rel_obs),
        "observable_name": observable_name,
        "strength_rel_l2": _relative_l2(x, y_pred, y_true),
    }


def evaluate_run(run: EM1Run) -> list[dict]:
    """Evaluate all dataset points in a run."""

    return [predict_entry(run, entry) for entry in run.combined]


def split_label_for_points(run: EM1Run) -> list[str]:
    labels = []
    split_by_key = {
        _key(entry): name
        for name, entries in run.split_entries.items()
        for entry in entries
    }
    for entry in run.combined:
        labels.append(split_by_key.get(_key(entry), "all"))
    return labels


def choose_point_index(results: list[dict], chosen_index: int | None = None, chosen_params=None) -> int:
    if chosen_params is not None:
        target = np.asarray(chosen_params, dtype=float)
        points = np.asarray([row["params"] for row in results], dtype=float)
        return int(np.argmin(np.sum((points - target[None, :]) ** 2, axis=1)))
    return int(chosen_index or 0)


def nearest_slice_indices(points: np.ndarray, varied_cols: list[int], center: np.ndarray, n_slice: int) -> np.ndarray:
    held_cols = [col for col in range(points.shape[1]) if col not in varied_cols]
    if not held_cols:
        return np.arange(points.shape[0])
    ranges = np.ptp(points[:, held_cols], axis=0)
    ranges = np.where(ranges > 0.0, ranges, 1.0)
    scaled = (points[:, held_cols] - center[held_cols][None, :]) / ranges[None, :]
    dist = np.linalg.norm(scaled, axis=1)
    return np.argsort(dist)[: min(n_slice, len(points))]


def plot_parameter_error_and_strength(
    run: EM1Run,
    results: list[dict],
    *,
    chosen_index: int | None = 0,
    chosen_params=None,
    varied_cols: tuple[int, int] = (0, 1),
    n_slice: int = 30,
):
    """Plot parameter grid/slice colored by observable error plus one strength comparison."""

    point_idx = choose_point_index(results, chosen_index=chosen_index, chosen_params=chosen_params)
    chosen = results[point_idx]
    points = np.asarray([row["params"] for row in results], dtype=float)
    errors = np.asarray([row["observable_relerr"] for row in results], dtype=float)
    center = np.asarray(chosen["params"], dtype=float)
    n_params = points.shape[1]

    if n_params == 1:
        slice_idx = np.arange(len(results))
    elif n_params == 2:
        slice_idx = np.arange(len(results))
    else:
        slice_idx = nearest_slice_indices(points, list(varied_cols), center, n_slice)

    fig = plt.figure(figsize=(13, 5.2))
    ax_grid = fig.add_subplot(1, 2, 1)
    ax_strength = fig.add_subplot(1, 2, 2)

    if n_params == 1:
        sc = ax_grid.scatter(points[slice_idx, 0], errors[slice_idx], c=errors[slice_idx], cmap="Spectral_r")
        ax_grid.set_xlabel("parameter 1")
        ax_grid.set_ylabel(f"relative error {chosen['observable_name']}")
    else:
        xcol, ycol = varied_cols
        sc = ax_grid.scatter(
            points[slice_idx, xcol],
            points[slice_idx, ycol],
            c=errors[slice_idx],
            cmap="Spectral_r",
            s=64,
            edgecolor="black",
            linewidth=0.35,
        )
        ax_grid.scatter(center[xcol], center[ycol], marker="x", s=110, c="black", linewidth=2)
        ax_grid.set_xlabel(f"parameter {xcol + 1}")
        ax_grid.set_ylabel(f"parameter {ycol + 1}")
        if n_params > 2:
            held = [col + 1 for col in range(n_params) if col not in varied_cols]
            ax_grid.set_title(f"Nearest slice; held params {held} near chosen point")
        else:
            ax_grid.set_title("Parameter grid")
    fig.colorbar(sc, ax=ax_grid, label=f"relative error {chosen['observable_name']}")
    ax_grid.grid(alpha=0.25)

    ax_strength.plot(chosen["x"], chosen["y_true"], label="true", lw=1.7)
    ax_strength.plot(chosen["x"], chosen["y_pred"], label="emulated", lw=1.9)
    markerline, stemlines, _ = ax_strength.stem(
        chosen["poles"], chosen["strengths"], linefmt="0.55", markerfmt="o", basefmt=" "
    )
    plt.setp(markerline, markersize=3, alpha=0.5)
    plt.setp(stemlines, linewidth=0.8, alpha=0.3)
    ax_strength.set_xlabel("Energy")
    ax_strength.set_ylabel("Strength")
    ax_strength.set_ylim(bottom=0)
    ax_strength.grid(alpha=0.25)
    ax_strength.legend()
    ax_strength.set_title(
        "chosen point: "
        + ", ".join(f"p{i + 1}={value:.4g}" for i, value in enumerate(chosen["params"]))
        + f"\nrel. observable err={chosen['observable_relerr']:.3e}; strength L2={chosen['strength_rel_l2']:.3e}"
    )
    fig.suptitle(run.run_dir.name)
    fig.tight_layout()
    return fig


def choose_sweep_indices(
    points: np.ndarray,
    *,
    vary_col: int,
    center: np.ndarray,
    n_sweep: int | None = None,
) -> np.ndarray:
    held_cols = [col for col in range(points.shape[1]) if col != vary_col]
    if not held_cols:
        order = np.argsort(points[:, vary_col])
        return order if n_sweep is None else order[:n_sweep]

    ranges = np.ptp(points[:, held_cols], axis=0)
    ranges = np.where(ranges > 0.0, ranges, 1.0)
    dist = np.linalg.norm((points[:, held_cols] - center[held_cols][None, :]) / ranges[None, :], axis=1)

    if n_sweep is None:
        n_sweep = min(20, len(points))
    chosen = np.argsort(dist)[:n_sweep]
    return chosen[np.argsort(points[chosen, vary_col])]


def plot_sweep_eigenvalue_evolution(
    run: EM1Run,
    results: list[dict],
    *,
    vary_col: int = 0,
    chosen_index: int | None = 0,
    chosen_params=None,
    n_sweep: int | None = None,
):
    """Plot parameter error map plus retained eigenvalue evolution along a sweep."""

    point_idx = choose_point_index(results, chosen_index=chosen_index, chosen_params=chosen_params)
    points = np.asarray([row["params"] for row in results], dtype=float)
    errors = np.asarray([row["observable_relerr"] for row in results], dtype=float)
    center = np.asarray(results[point_idx]["params"], dtype=float)
    sweep_idx = choose_sweep_indices(points, vary_col=vary_col, center=center, n_sweep=n_sweep)

    fig = plt.figure(figsize=(13, 5.2))
    ax_grid = fig.add_subplot(1, 2, 1)
    ax_eigs = fig.add_subplot(1, 2, 2)

    if points.shape[1] == 1:
        sc = ax_grid.scatter(points[:, 0], errors, c=errors, cmap="Spectral_r")
        ax_grid.set_xlabel("parameter 1")
        ax_grid.set_ylabel("relative observable error")
    else:
        xcol = 0 if vary_col != 0 else min(1, points.shape[1] - 1)
        ycol = vary_col
        sc = ax_grid.scatter(points[:, xcol], points[:, ycol], c=errors, cmap="Spectral_r", s=56)
        ax_grid.plot(points[sweep_idx, xcol], points[sweep_idx, ycol], "k.-", lw=1.1, ms=7)
        ax_grid.scatter(center[xcol], center[ycol], marker="x", s=110, c="black", linewidth=2)
        ax_grid.set_xlabel(f"parameter {xcol + 1}")
        ax_grid.set_ylabel(f"parameter {ycol + 1}")
    fig.colorbar(sc, ax=ax_grid, label="relative observable error")
    ax_grid.set_title("Parameter grid and chosen sweep")
    ax_grid.grid(alpha=0.25)

    x = points[sweep_idx, vary_col]
    max_modes = max(len(results[idx]["poles"]) for idx in sweep_idx)
    eig_matrix = np.full((len(sweep_idx), max_modes), np.nan)
    for row, idx in enumerate(sweep_idx):
        poles = np.asarray(results[idx]["poles"], dtype=float)
        eig_matrix[row, : len(poles)] = poles
    for mode in range(eig_matrix.shape[1]):
        ax_eigs.plot(x, eig_matrix[:, mode], "o-", ms=3, lw=1.0, alpha=0.8)
    ax_eigs.set_xlabel(f"parameter {vary_col + 1}")
    ax_eigs.set_ylabel("retained eigenvalues")
    ax_eigs.set_title("Eigenvalue evolution along sweep")
    ax_eigs.grid(alpha=0.25)
    fig.suptitle(run.run_dir.name)
    fig.tight_layout()
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--chosen-index", type=int, default=0)
    parser.add_argument("--vary-col", type=int, default=0)
    parser.add_argument("--n-slice", type=int, default=30)
    parser.add_argument("--n-sweep", type=int, default=None)
    args = parser.parse_args()

    run = load_em1_run(args.run_dir, data_dir=args.data_dir)
    results = evaluate_run(run)
    out_dir = args.out_dir or (run.run_dir / "diagnostic_plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = plot_parameter_error_and_strength(run, results, chosen_index=args.chosen_index, n_slice=args.n_slice)
    fig.savefig(out_dir / "parameter_error_and_strength.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    fig = plot_sweep_eigenvalue_evolution(
        run,
        results,
        chosen_index=args.chosen_index,
        vary_col=args.vary_col,
        n_sweep=args.n_sweep,
    )
    fig.savefig(out_dir / "parameter_grid_eigenvalue_sweep.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved diagnostics in {out_dir}")


if __name__ == "__main__":
    main()
