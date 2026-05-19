#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


STRENGTH_REGEX = r"strength_(?P<p1>-?[0-9.]+)_(?P<p2>-?[0-9.]+)_(?P<p3>-?[0-9.]+)_(?P<p4>-?[0-9.]+)\.out"


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[2]


def load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def collect_rows(sweep_dir: Path) -> list[dict]:
    allowed_names = None
    index_path = sweep_dir / "sweep_index.json"
    if index_path.exists():
        index = load_json(index_path)
        allowed_names = {cfg["name"] for cfg in index.get("configs", [])}

    rows = []
    for run_dir in sorted(sweep_dir.iterdir()):
        if not run_dir.is_dir() or run_dir.name == "logs":
            continue
        if allowed_names is not None and run_dir.name not in allowed_names:
            continue
        summary_path = run_dir / "run_summary.json"
        if not summary_path.exists():
            rows.append({"name": run_dir.name, "status": "missing", "run_dir": str(run_dir)})
            continue
        summary = load_json(summary_path)
        train_metrics = summary.get("global_best_meta", {}).get("train_metrics") or {}
        cfg = summary.get("config") or {}
        cli_args = summary.get("args") or {}
        rows.append({
            "name": run_dir.name,
            "status": "done",
            "strength_cost": train_metrics.get("strength_cost"),
            "cost": train_metrics.get("cost"),
            "global_best_cost": summary.get("global_best_cost"),
            "seed": summary.get("global_best_meta", {}).get("seed"),
            "best_iter": summary.get("global_best_meta", {}).get("iter"),
            "n": cfg.get("n"),
            "retain": cli_args.get("retain"),
            "fold": cli_args.get("fold"),
            "ansatz": cfg.get("ansatz"),
            "width_model": cfg.get("width_model"),
            "learning_rate": cli_args.get("learning_rate"),
            "run_dir": str(run_dir),
        })
    rows.sort(key=lambda row: (row["status"] != "done", float("inf") if row.get("strength_cost") is None else row["strength_cost"]))
    return rows


def write_summary(rows: list[dict], output: Path) -> None:
    fields = [
        "name", "status", "strength_cost", "cost", "global_best_cost", "seed",
        "best_iter", "n", "retain", "fold", "ansatz", "width_model",
        "learning_rate", "run_dir",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_diagnostics(repo_root: Path, best: dict, fig_dir: Path, python_exe: str) -> None:
    command = [
        python_exe,
        str(repo_root / "Dipole_polarizability/src/diagnostics_general_gpt.py"),
        "--strength-dir", str(repo_root / "extra_docs/yukiya_4d/total_strength"),
        "--alphaD-dir", str(repo_root / "extra_docs/yukiya_4d/total_alphaD"),
        "--strength-regex", STRENGTH_REGEX,
        "--n", str(best["n"]),
        "--retain", str(best["retain"]),
        "--ansatz", str(best["ansatz"]),
        "--width-model", str(best["width_model"]),
        "--plots", "save",
        "--save-dir", str(best["run_dir"]),
        "--fig-dir", str(fig_dir),
        "--max-label-points", "100",
    ]
    subprocess.run(command, cwd=repo_root, check=True)


def load_strength_files(strength_dir: Path) -> list[tuple[Path, np.ndarray]]:
    files = sorted(strength_dir.glob("strength_*.out"))
    return [(path, np.loadtxt(path)) for path in files]


def plot_dataset_overlays(strength_dir: Path, fig_dir: Path) -> None:
    spectra = load_strength_files(strength_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    for _, arr in spectra:
        ax.plot(arr[:, 0], arr[:, 1], color="0.25", alpha=0.25, linewidth=1)
    ax.set_xlabel("Excitation energy")
    ax.set_ylabel("Strength")
    ax.set_title("All Yukiya Ca48 GT K1 spectra")
    fig.tight_layout()
    fig.savefig(fig_dir / "all_true_spectra_overlay.png", dpi=200)
    plt.close(fig)

    integrals = np.array([np.trapz(arr[:, 1], arr[:, 0]) for _, arr in spectra])
    peaks = np.array([arr[np.argmax(arr[:, 1]), 0] for _, arr in spectra])
    max_strength = np.array([np.max(arr[:, 1]) for _, arr in spectra])

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    axes[0].hist(integrals, bins=12, color="C0", alpha=0.85)
    axes[0].set_title("Integrated strength")
    axes[1].hist(peaks, bins=12, color="C1", alpha=0.85)
    axes[1].set_title("Peak energy")
    axes[2].hist(max_strength, bins=12, color="C2", alpha=0.85)
    axes[2].set_title("Peak strength")
    for ax in axes:
        ax.tick_params(direction="in")
    fig.tight_layout()
    fig.savefig(fig_dir / "true_spectra_summary_histograms.png", dpi=200)
    plt.close(fig)


def main() -> int:
    repo_root = repo_root_from_script()
    default_python = Path("/Users/laurenjin/envs/smlr-frib/bin/python")
    default_python_exe = str(default_python) if default_python.exists() else sys.executable

    parser = argparse.ArgumentParser(description="Summarize Yukiya sweep results and plot diagnostics for the best run.")
    parser.add_argument("--sweep-dir", type=Path, default=repo_root / "extra_docs/yukiya_4d/runs_smlr/yukiya_strength_sweep")
    parser.add_argument("--python-exe", default=default_python_exe)
    args = parser.parse_args()

    rows = collect_rows(args.sweep_dir)
    summary_path = args.sweep_dir / "sweep_summary.csv"
    write_summary(rows, summary_path)
    done = [row for row in rows if row["status"] == "done"]
    if not done:
        raise RuntimeError(f"No completed runs found under {args.sweep_dir}")

    best = done[0]
    best_dir = Path(best["run_dir"])
    fig_dir = best_dir / "best_plots"
    run_diagnostics(repo_root, best, fig_dir, args.python_exe)
    plot_dataset_overlays(repo_root / "extra_docs/yukiya_4d/total_strength", fig_dir)

    print(f"Wrote {summary_path}")
    print("Best run:")
    print(json.dumps(best, indent=2))
    print(f"Plots: {fig_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
