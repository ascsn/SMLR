from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np

from ..data import StrengthDataset, StrengthSample
from ..emulator import StrengthEmulator
from ..lorentz import LorentzianMixture, lorentzian_sum
from ..plotting import plot_comparison


def _synthetic_strength(params: Tuple[float, float], energy: np.ndarray) -> np.ndarray:
    a, b = params
    centers = np.array([-1.5 + 0.4 * a, 0.2 + 0.3 * b])
    strengths = np.array([2.5 + 0.2 * a, 1.2 + 0.5 * b])
    width = np.full(2, 0.35 + 0.05 * (a + b))
    return lorentzian_sum(energy, centers, strengths, width)


def build_dataset() -> StrengthDataset:
    energy = np.linspace(-3.0, 3.0, 200)
    params_grid = np.stack(np.meshgrid(np.linspace(0, 1, 3), np.linspace(0, 1, 3)), axis=-1).reshape(-1, 2)
    samples = []
    for i, p in enumerate(params_grid):
        y = _synthetic_strength((p[0], p[1]), energy)
        samples.append(StrengthSample(params=p, energy=energy, strength=y, label=f"pt-{i}"))
    return StrengthDataset(samples)


def main(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = build_dataset()
    emulator = StrengthEmulator(n_components=2, width_mode="global", random_state=0)
    emulator.fit(dataset)

    target_params = np.array([0.3, 0.7])
    energy = dataset.energy_grids()[0]
    true_strength = _synthetic_strength(tuple(target_params), energy)
    result = emulator.predict(target_params, energy)

    fig = plot_comparison(energy, true_strength, result.spectrum, title="Synthetic demo")
    fig.savefig(out_dir / "synthetic_demo.png", dpi=150, bbox_inches="tight")

    np.savez(
        out_dir / "demo_output.npz",
        params=target_params,
        energies=result.poles,
        strengths=result.strengths,
        widths=result.widths,
        energy_grid=energy,
        pred_strength=result.spectrum,
        true_strength=true_strength,
    )
    print(f"Demo complete. Artifacts written to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and eval a small synthetic emulator")
    parser.add_argument("--out", type=Path, default=Path("runs/demo"), help="Output directory")
    args = parser.parse_args()
    main(args.out)
