from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from smlr import metrics


@dataclass(frozen=True)
class DiagnosticLabels:
    observable_name: str = "Observable"
    observable_true: str = "True observable"
    observable_pred: str = "Predicted observable"
    relative_error: str = "Relative error"
    parameter_names: tuple[str, ...] = ("p1", "p2")
    spectrum_x: str = "Energy"
    spectrum_y: str = "Strength"
    detail_spectrum_title: str = "Detailed spectrum"
    detail_observable_title: str = "Detailed observable"
    prediction_title: str = "Prediction vs truth"
    parameter_map_title: str = "Parameter-space relative error"


STANDARD_FILES = {
    "prediction_scatter": "prediction_scatter.png",
    "parameter_error_map": "parameter_error_map.png",
    "detail_spectrum": "detail_spectrum.png",
    "detail_observable": "detail_observable.png",
    "predictions_csv": "predictions.csv",
    "summary": "summary.txt",
}


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_observable_predictions_csv(
    out_path: str | Path,
    points,
    true,
    predicted,
    *,
    parameter_names: Sequence[str] = ("p1", "p2"),
    observable_name: str = "observable",
    eps: float = 1e-12,
) -> None:
    """Write a standardized observable prediction CSV."""

    true = np.asarray(true, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    rel = metrics.relative_error(predicted, true, eps=eps)
    abs_err = metrics.absolute_error(predicted, true)
    parameter_names = tuple(parameter_names)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["idx", *parameter_names, f"true_{observable_name}", f"pred_{observable_name}", "absolute_error", "relative_error"]
        )
        for idx, (point, y_true, y_pred, ae, re) in enumerate(zip(points, true, predicted, abs_err, rel)):
            writer.writerow([idx, *point, y_true, y_pred, ae, re])


def write_summary(out_path: str | Path, summary: dict[str, float]) -> None:
    with open(out_path, "w") as f:
        for key, value in summary.items():
            f.write(f"{key}={value}\n")


def plot_prediction_scatter(
    true,
    predicted,
    *,
    labels: DiagnosticLabels = DiagnosticLabels(),
    log_scale: bool = False,
    label_points: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    true = np.asarray(true, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(true, predicted, s=18)
    lo = min(float(np.min(true)), float(np.min(predicted)))
    hi = max(float(np.max(true)), float(np.max(predicted)))
    ax.plot([lo, hi], [lo, hi], color="black")
    if log_scale:
        ax.set_xscale("log")
        ax.set_yscale("log")
    if label_points:
        for idx, (x, y) in enumerate(zip(true, predicted)):
            ax.text(x, y, str(idx), fontsize=8, ha="right", va="bottom")
    ax.set_xlabel(labels.observable_true)
    ax.set_ylabel(labels.observable_pred)
    ax.set_title(labels.prediction_title)
    fig.tight_layout()
    return fig, ax


def plot_parameter_error_map(
    points,
    errors,
    *,
    labels: DiagnosticLabels = DiagnosticLabels(),
    max_label_points: int = 1000,
    filter_ranges: dict[str, tuple[float, float]] | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    points = np.asarray(points, dtype=float)
    errors = np.asarray(errors, dtype=float).reshape(-1)
    if points.ndim != 2:
        raise ValueError(f"points must be 2D, got shape {points.shape}.")
    if points.shape[0] != errors.size:
        raise ValueError("points and errors must have the same sample count.")

    fig, ax = plt.subplots(figsize=(6, 5))
    positive = np.clip(errors, 1e-16, None)
    norm = LogNorm(vmin=float(np.min(positive)), vmax=float(np.max(positive))) if np.max(positive) > np.min(positive) else None

    if points.shape[1] == 1:
        order = np.argsort(points[:, 0])
        sc = ax.scatter(points[order, 0], errors[order], c=positive[order], cmap="Spectral", norm=norm)
        ax.plot(points[order, 0], errors[order], color="0.25", alpha=0.5, linewidth=1.25)
        ax.set_xlabel(labels.parameter_names[0])
        ax.set_ylabel(labels.relative_error)
    else:
        sc = ax.scatter(points[:, 0], points[:, 1], c=positive, marker="s", cmap="Spectral", norm=norm)
        ax.set_xlabel(labels.parameter_names[0])
        ax.set_ylabel(labels.parameter_names[1])
        if filter_ranges:
            _draw_filter_box(ax, labels.parameter_names, filter_ranges)

    fig.colorbar(sc, ax=ax, label=labels.relative_error)
    if len(errors) <= max_label_points:
        if points.shape[1] == 1:
            for idx, x in enumerate(points[:, 0]):
                ax.text(x, errors[idx], str(idx), fontsize=8, ha="center", va="bottom")
        else:
            for idx, (x, y) in enumerate(points[:, :2]):
                ax.text(x, y, str(idx), fontsize=8, ha="center", va="center", color="black")
    ax.set_title(labels.parameter_map_title)
    fig.tight_layout()
    return fig, ax


def _draw_filter_box(ax, parameter_names, filter_ranges):
    if len(parameter_names) < 2:
        return
    x_name, y_name = parameter_names[0], parameter_names[1]
    if x_name not in filter_ranges or y_name not in filter_ranges:
        return
    x0, x1 = filter_ranges[x_name]
    y0, y1 = filter_ranges[y_name]
    if x1 <= x0 or y1 <= y0:
        return
    import matplotlib.patches as patches

    ax.add_patch(patches.Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor="0.6", alpha=0.3, edgecolor="none"))
    ax.add_patch(patches.Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor="none", edgecolor="black", linewidth=2.0))


def plot_detail_spectrum(
    energy,
    true_strength,
    predicted_strength,
    *,
    poles=None,
    pole_strengths=None,
    labels: DiagnosticLabels = DiagnosticLabels(),
    title_suffix: str = "",
    yscale: str = "linear",
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(energy, predicted_strength, label="pred")
    ax.plot(energy, true_strength, label="true", alpha=0.85)
    if poles is not None and pole_strengths is not None:
        ax.stem(poles, pole_strengths, basefmt=" ", linefmt="C2-", markerfmt="C2o", label="poles")
    ax.set_xlabel(labels.spectrum_x)
    ax.set_ylabel(labels.spectrum_y)
    ax.set_yscale(yscale)
    ax.set_ylim(bottom=0 if yscale == "linear" else None)
    ax.set_title(f"{labels.detail_spectrum_title}{title_suffix}")
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_detail_observable(
    true_value,
    predicted_value,
    *,
    labels: DiagnosticLabels = DiagnosticLabels(),
    title_suffix: str = "",
    log_scale: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["true", "pred"], [true_value, predicted_value], color=["black", "tab:red"], alpha=0.8)
    if log_scale:
        ax.set_yscale("log")
    ax.set_ylabel(labels.observable_name)
    ax.set_title(f"{labels.detail_observable_title}{title_suffix}")
    fig.tight_layout()
    return fig, ax


def save_figure(fig, out_dir: str | Path, filename: str, *, dpi: int = 200) -> Path:
    out_dir = ensure_dir(out_dir)
    path = out_dir / filename
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


def write_standard_observable_diagnostics(
    out_dir: str | Path,
    points,
    true,
    predicted,
    *,
    labels: DiagnosticLabels = DiagnosticLabels(),
    observable_key: str = "observable",
    log_scatter: bool = False,
    max_label_points: int = 1000,
    filter_ranges: dict[str, tuple[float, float]] | None = None,
    dpi: int = 200,
    plots: bool = True,
    csv_alias: str | None = None,
    scatter_alias: str | None = None,
) -> dict[str, float]:
    """Write standardized CSV, summary, prediction scatter, and parameter map."""

    out_dir = ensure_dir(out_dir)
    true = np.asarray(true, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    rel = metrics.relative_error(predicted, true)
    summary = metrics.observable_error_summary(predicted, true)
    write_observable_predictions_csv(
        out_dir / STANDARD_FILES["predictions_csv"],
        points,
        true,
        predicted,
        parameter_names=labels.parameter_names,
        observable_name=observable_key,
    )
    if csv_alias:
        write_observable_predictions_csv(
            out_dir / csv_alias,
            points,
            true,
            predicted,
            parameter_names=labels.parameter_names,
            observable_name=observable_key,
        )
    write_summary(out_dir / STANDARD_FILES["summary"], summary)
    if not plots:
        return summary

    fig, _ = plot_prediction_scatter(
        true,
        predicted,
        labels=labels,
        log_scale=log_scatter,
        label_points=(len(true) <= max_label_points),
    )
    save_figure(fig, out_dir, STANDARD_FILES["prediction_scatter"], dpi=dpi)
    if scatter_alias:
        save_figure(fig, out_dir, scatter_alias, dpi=dpi)
    plt.close(fig)

    fig, _ = plot_parameter_error_map(
        points,
        rel,
        labels=labels,
        max_label_points=max_label_points,
        filter_ranges=filter_ranges,
    )
    save_figure(fig, out_dir, STANDARD_FILES["parameter_error_map"], dpi=dpi)
    plt.close(fig)
    return summary
