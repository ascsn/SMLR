#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_grid_indices(value: str | None) -> list[int] | None:
    if value is None:
        return None
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def default_sample_indices(rows: list[dict[str, str]]) -> list[int]:
    q_values = sorted({float(row["q_bohr_inv"]) for row in rows})
    theta_values = sorted({float(row["theta_deg"]) for row in rows})

    selected_q = [q_values[0], q_values[len(q_values) // 2], q_values[-1]]
    selected_theta = [theta_values[0], theta_values[len(theta_values) // 2], theta_values[-1]]

    chosen = []
    for row in rows:
        q = float(row["q_bohr_inv"])
        theta = float(row["theta_deg"])
        if q in selected_q and theta in selected_theta:
            chosen.append(int(row["grid_index"]))
    return chosen


def main() -> None:
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Overlay selected H2 folded strength-function samples from total_strength."
    )
    parser.add_argument("--strength-dir", type=Path, default=script_dir / "total_strength")
    parser.add_argument("--manifest", type=Path, default=script_dir / "total_strength_manifest.csv")
    parser.add_argument(
        "--grid-indices",
        type=str,
        default=None,
        help="Comma-separated grid_index values to overlay. Defaults to low/mid/high q at 0/45/90 deg.",
    )
    parser.add_argument("--output", type=Path, default=script_dir / "total_strength_overlay_samples.png")
    parser.add_argument("--xlim", type=float, nargs=2, default=None, metavar=("MIN_EV", "MAX_EV"))
    parser.add_argument("--log-y", action="store_true", help="Use a logarithmic y-axis.")
    args = parser.parse_args()

    rows = load_manifest(args.manifest)
    by_grid = {int(row["grid_index"]): row for row in rows}
    grid_indices = parse_grid_indices(args.grid_indices) or default_sample_indices(rows)

    fig, ax = plt.subplots(figsize=(9.0, 5.8))
    cmap = plt.get_cmap("viridis")
    colors = cmap(np.linspace(0.08, 0.92, len(grid_indices)))

    for color, grid_index in zip(colors, grid_indices):
        if grid_index not in by_grid:
            raise ValueError(f"grid_index {grid_index} is not present in {args.manifest}")

        row = by_grid[grid_index]
        data = np.loadtxt(args.strength_dir / row["filename"])
        omega = data[:, 0]
        strength = data[:, 1]
        label = f"q={float(row['q_bohr_inv']):.3g}, theta={float(row['theta_deg']):.1f} deg"
        ax.plot(omega, strength, lw=1.6, color=color, label=label)

    ax.set_xlabel("Excitation energy (eV)")
    ax.set_ylabel("Strength")
    ax.set_title("H2 folded strength functions")
    if args.xlim is not None:
        ax.set_xlim(args.xlim)
    if args.log_y:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8, ncol=1)
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {args.output}")
    print("Overlayed grid_index values:", ", ".join(str(i) for i in grid_indices))


if __name__ == "__main__":
    main()
