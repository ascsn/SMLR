"""Paper reproduction helper using the new `smlr` package.

Usage:
    uv run python examples/paper_repro.py --mode synthetic --out runs/demo-script
    uv run python examples/paper_repro.py --mode paper --save-dir runs_em1 --dipole-dir runs_dipole

- synthetic: quick smoke test using `smlr.demo.synthetic` (CI-friendly).
- paper: fits small emulators on the original datasets via the new package (beta-decay + dipole).
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import numpy as np
from smlr.demo.synthetic import main as demo_main
from smlr import Surrogate, StrengthDataset, StrengthSample
from smlr.metrics import normalized_l2
from smlr.plotting import plot_comparison
import matplotlib.pyplot as plt


def main() -> None:
    p = argparse.ArgumentParser(description="Paper reproduction helper")
    p.add_argument("--mode", choices=["synthetic", "paper"], default="synthetic")
    p.add_argument("--out", type=Path, default=Path("runs/demo-script"), help="Output dir for synthetic mode")
    p.add_argument("--max-files", type=int, default=6, help="Max files to load per dataset in paper mode")
    p.add_argument("--plots-dir", type=Path, default=Path("runs/plots-script"), help="Where to write comparison plots")
    args = p.parse_args()

    if args.mode == "synthetic":
        args.out.mkdir(parents=True, exist_ok=True)
        demo_main(args.out)
        print(f"Synthetic demo complete -> {args.out}")
        return

    def load_beta(max_files: int):
        root = Path("beta_decay_data_Ni_80")
        if not root.exists():
            return None
        pattern = re.compile(r"lorm_.*_([0-9.]+)_([0-9.]+)\.out")
        samples = []
        for f in sorted(root.glob("lorm_*.out"))[:max_files]:
            m = pattern.match(f.name)
            if not m:
                continue
            beta_val, alpha_val = map(float, m.groups())
            data = np.loadtxt(f)
            if data.ndim != 2 or data.shape[1] < 2:
                continue
            samples.append(StrengthSample(np.array([alpha_val, beta_val]), data[:, 0], data[:, 1], f.name))
        return StrengthDataset(samples) if samples else None

    def load_dipole(max_files: int):
        root = Path("dipoles_data_all")
        if not root.exists():
            return None
        candidates = sorted(root.glob("**/strength_*_*.out"))[:max_files]
        pattern = re.compile(r"strength_([0-9.]+)_([0-9.]+)\.out")
        samples = []
        for f in candidates:
            m = pattern.search(f.name)
            if not m:
                continue
            beta_val, alpha_val = map(float, m.groups())
            data = np.loadtxt(f)
            if data.ndim != 2 or data.shape[1] < 2:
                continue
            samples.append(StrengthSample(np.array([alpha_val, beta_val]), data[:, 0], data[:, 1], f.name))
        return StrengthDataset(samples) if samples else None

    def fit_one(ds: StrengthDataset, n_components: int = 3):
        model = Surrogate("regression", n_components=n_components, width_mode="global", random_state=0)
        model.fit(ds)
        point = ds.parameters().mean(axis=0)
        energy_grid = np.linspace(ds.energy_grids()[0].min(), ds.energy_grids()[0].max(), 300)
        result = model.predict(point, energy_grid)
        idx = np.argmin(np.linalg.norm(ds.parameters() - point, axis=1))
        ref = np.interp(energy_grid, ds.samples[idx].energy, ds.samples[idx].strength)
        err = normalized_l2(result.spectrum, ref, energy_grid)
        return err, energy_grid, result.spectrum, ref

    beta_ds = load_beta(args.max_files)
    dipole_ds = load_dipole(args.max_files)

    args.plots_dir.mkdir(parents=True, exist_ok=True)

    if beta_ds:
        beta_err, energy, pred, ref = fit_one(beta_ds)
        print(f"Beta-decay emulator normalized L2 vs nearest spectrum: {beta_err:.3f}")
        fig = plot_comparison(energy, ref, pred, title=f"Beta-decay (L2={beta_err:.3f})", labels=("reference", "emulator"))
        fig.savefig(args.plots_dir / "beta_decay.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        print("Beta-decay data not found; skipped.")

    if dipole_ds:
        dipole_err, energy, pred, ref = fit_one(dipole_ds)
        print(f"Dipole emulator normalized L2 vs nearest spectrum: {dipole_err:.3f}")
        fig = plot_comparison(energy, ref, pred, title=f"Dipole (L2={dipole_err:.3f})", labels=("reference", "emulator"))
        fig.savefig(args.plots_dir / "dipole.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        print("Dipole data not found; skipped.")

    if not (beta_ds or dipole_ds):
        print("No data found; falling back to synthetic demo.")
        args.out.mkdir(parents=True, exist_ok=True)
        demo_main(args.out)


if __name__ == "__main__":
    main()
