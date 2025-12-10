"""Materials science spectroscopy emulation example.

This example demonstrates SMLR applied to optical/X-ray absorption spectroscopy
in materials science, showing applications in:

- X-ray absorption near-edge structure (XANES)
- UV-Vis absorption spectroscopy
- Photoluminescence spectroscopy
- Raman spectroscopy peak fitting

The approach works for any spectral data that can be represented as a
sum of peaks/resonances.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from smlr.data import StrengthDataset, StrengthSample
from smlr.emulator import StrengthEmulator
from smlr.lorentz import lorentzian_sum
from smlr.metrics import normalized_l2
from smlr.observables import SumRule, CustomObservable, ObservableSet


def generate_absorption_spectrum(
    params: np.ndarray,
    energy: np.ndarray,
) -> np.ndarray:
    """Generate a synthetic X-ray absorption spectrum.
    
    Simulates XANES-like spectra parameterized by material properties.
    
    Parameters
    ----------
    params : array of shape (6,)
        - params[0]: Edge energy shift (oxidation state effect)
        - params[1]: Pre-edge feature intensity
        - params[2]: White-line intensity
        - params[3]: Coordination number effect
        - params[4]: Disorder parameter (affects broadening)
        - params[5]: Bond length effect (shifts features)
    energy : array
        Photon energy grid in eV.
        
    Returns
    -------
    array
        Normalized absorption coefficient.
    """
    edge_shift, pre_edge, white_line, coord, disorder, bond_length = params
    
    # Base edge energy (e.g., Fe K-edge ~7112 eV, normalized here)
    E_edge = 50.0 + 2.0 * edge_shift
    
    # Pre-edge feature (1s -> 3d transitions)
    pre_edge_pos = E_edge - 5.0 + 0.5 * bond_length
    pre_edge_strength = 0.3 * pre_edge * (1 - 0.1 * coord)
    pre_edge_width = 1.0 + 0.3 * disorder
    
    # White-line (main absorption edge)
    white_line_pos = E_edge + 2.0 + 0.3 * coord
    white_line_strength = 2.0 * white_line * (1 + 0.2 * coord)
    white_line_width = 1.5 + 0.5 * disorder
    
    # EXAFS oscillations (simplified as additional peaks)
    n_exafs = 4
    exafs_positions = []
    exafs_strengths = []
    exafs_widths = []
    
    for n in range(1, n_exafs + 1):
        pos = E_edge + 10.0 + 8.0 * n * (1 + 0.1 * bond_length)
        strength = 0.5 * coord / (n + 1) * np.exp(-0.1 * n * disorder)
        width = 3.0 + 1.0 * disorder + 0.5 * n
        
        exafs_positions.append(pos)
        exafs_strengths.append(strength)
        exafs_widths.append(width)
    
    # Combine all features
    all_positions = np.array([pre_edge_pos, white_line_pos] + exafs_positions)
    all_strengths = np.array([pre_edge_strength, white_line_strength] + exafs_strengths)
    all_widths = np.array([pre_edge_width, white_line_width] + exafs_widths)
    
    # Ensure physical values
    all_strengths = np.maximum(all_strengths, 0.01)
    all_widths = np.maximum(all_widths, 0.5)
    
    spectrum = lorentzian_sum(energy, all_positions, all_strengths, all_widths)
    
    # Add edge step (arctangent)
    edge_step = 0.5 * (1 + np.tanh((energy - E_edge) / 2.0))
    
    return spectrum + 0.3 * edge_step


def generate_raman_spectrum(
    params: np.ndarray,
    wavenumber: np.ndarray,
) -> np.ndarray:
    """Generate a synthetic Raman spectrum.
    
    Simulates Raman spectra of a crystalline material with vibrational modes
    that depend on structural parameters.
    
    Parameters
    ----------
    params : array of shape (4,)
        - params[0]: Lattice strain (shifts peak positions)
        - params[1]: Crystallinity (affects peak widths)
        - params[2]: Defect concentration (adds disorder peaks)
        - params[3]: Temperature (affects intensities via Bose factor)
    wavenumber : array
        Raman shift in cm^-1.
        
    Returns
    -------
    array
        Raman intensity.
    """
    strain, crystallinity, defects, temperature = params
    
    # Main phonon modes (e.g., for a perovskite-like material)
    base_modes = np.array([150, 250, 400, 550, 750])  # cm^-1
    base_intensities = np.array([0.8, 1.2, 0.5, 1.5, 0.4])
    
    # Strain shifts all modes
    strain_shift = 10 * strain
    mode_positions = base_modes + strain_shift
    
    # Crystallinity affects widths (lower crystallinity = broader peaks)
    base_width = 5.0
    widths = base_width * (2.0 - crystallinity)
    
    # Temperature affects intensities (Bose factor approximation)
    bose_factor = 1.0 / (1.0 - np.exp(-base_modes / (200 * (1 + temperature))))
    intensities = base_intensities * bose_factor
    
    # Defect-induced features
    defect_positions = np.array([180, 320, 480])
    defect_intensities = 0.3 * defects * np.array([0.5, 0.3, 0.4])
    defect_widths = np.full(3, 15.0)  # Broad defect peaks
    
    # Combine
    all_pos = np.concatenate([mode_positions, defect_positions])
    all_int = np.concatenate([intensities, defect_intensities])
    all_width = np.concatenate([np.full_like(mode_positions, widths), defect_widths])
    
    return lorentzian_sum(wavenumber, all_pos, all_int, all_width)


def build_xanes_dataset(n_samples: int = 60, seed: int = 42) -> StrengthDataset:
    """Build XANES spectroscopy dataset."""
    rng = np.random.default_rng(seed)
    energy = np.linspace(30, 100, 250)
    
    samples = []
    for i in range(n_samples):
        params = np.array([
            rng.uniform(-1, 1),    # edge_shift
            rng.uniform(0.2, 1.5), # pre_edge
            rng.uniform(0.5, 2.0), # white_line
            rng.uniform(2, 8),     # coord (coordination number)
            rng.uniform(0.1, 1.0), # disorder
            rng.uniform(-0.5, 0.5), # bond_length
        ])
        spectrum = generate_absorption_spectrum(params, energy)
        samples.append(StrengthSample(params=params, energy=energy, strength=spectrum))
    
    return StrengthDataset(samples)


def build_raman_dataset(n_samples: int = 50, seed: int = 42) -> StrengthDataset:
    """Build Raman spectroscopy dataset."""
    rng = np.random.default_rng(seed)
    wavenumber = np.linspace(100, 900, 200)
    
    samples = []
    for i in range(n_samples):
        params = np.array([
            rng.uniform(-1, 1),    # strain
            rng.uniform(0.3, 1.0), # crystallinity
            rng.uniform(0, 1),     # defects
            rng.uniform(0.2, 1.5), # temperature
        ])
        spectrum = generate_raman_spectrum(params, wavenumber)
        samples.append(StrengthSample(params=params, energy=wavenumber, strength=spectrum))
    
    return StrengthDataset(samples)


def main(
    spectroscopy_type: str = "xanes",
    n_train: int = 50,
    n_test: int = 10,
    out_dir: Path = Path("runs/materials_demo"),
) -> None:
    """Run materials spectroscopy emulation demo."""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Materials Science Spectroscopy Emulation: {spectroscopy_type.upper()}")
    print("=" * 60)
    
    # Build dataset
    if spectroscopy_type == "xanes":
        full_ds = build_xanes_dataset(n_samples=n_train + n_test)
        x_label = "Photon Energy (eV)"
        y_label = "Absorption (arb.)"
        n_components = 6
        param_names = ["Edge Shift", "Pre-edge", "White-line", "Coord.", "Disorder", "Bond Length"]
    else:  # raman
        full_ds = build_raman_dataset(n_samples=n_train + n_test)
        x_label = "Raman Shift (cm⁻¹)"
        y_label = "Intensity (arb.)"
        n_components = 8
        param_names = ["Strain", "Crystallinity", "Defects", "Temperature"]
    
    train_ds, _, test_ds = full_ds.train_val_test_split(
        train=n_train / (n_train + n_test), val=0.0, seed=42
    )
    
    # Fit emulators with different methods for comparison
    methods = ["linear", "ridge", "polynomial"]
    results = {}
    
    for method in methods:
        print(f"\nFitting {method} emulator...")
        emu = StrengthEmulator(
            n_components=n_components,
            regression_method=method,
            poly_degree=2 if method == "polynomial" else 2,
            width_mode="per_component",
            random_state=42,
        )
        emu.fit(train_ds)
        
        # Evaluate
        x_grid = train_ds.energy_grids()[0]
        errors = []
        for sample in test_ds.samples:
            pred = emu.predict_spectrum(sample.params, x_grid)
            errors.append(normalized_l2(pred, sample.strength, x_grid))
        
        results[method] = {
            "emulator": emu,
            "errors": np.array(errors),
            "mean_error": np.mean(errors),
        }
        print(f"  Mean L2 error: {results[method]['mean_error']:.4f}")

    # -----------------------------------------------------------------
    # PMM backend evaluation
    # -----------------------------------------------------------------
    try:
        from smlr.backends import get_emulator
        print("\nFitting PMM backend for comparison...")
        pmm_emulator = get_emulator("pmm", n_poles=21)
        pmm_emulator.fit(train_ds)

        pmm_errors = []
        x_grid = train_ds.energy_grids()[0]
        for sample in test_ds.samples:
            res = pmm_emulator.predict(sample.params, x_grid)
            spec = res.spectrum if hasattr(res, 'spectrum') else res
            pmm_errors.append(normalized_l2(spec, sample.strength, x_grid))

        results['pmm'] = {
            "emulator": pmm_emulator,
            "errors": np.array(pmm_errors),
            "mean_error": np.mean(pmm_errors),
        }
        print(f"  PMM Mean L2 error: {results['pmm']['mean_error']:.4f}")
    except Exception as e:
        print("PMM evaluation skipped due to error:", e)
    
    # Visualization
    x_grid = train_ds.energy_grids()[0]
    
    # Method comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    sample = test_ds.samples[0]
    
    for ax, method in zip(axes, methods):
        emu = results[method]["emulator"]
        pred = emu.predict_spectrum(sample.params, x_grid)
        # Regression/ML emulator prediction
        ax.plot(x_grid, sample.strength, 'b-', label='Reference', linewidth=2)
        ax.plot(x_grid, pred, 'r--', label=f'{method.capitalize()} Emulator', linewidth=1.5)

        # If PMM was evaluated, overlay its spectrum for direct backend comparison
        if 'pmm' in results:
            try:
                pmm_emu = results['pmm']['emulator']
                pmm_pred = pmm_emu.predict(sample.params, x_grid)
                pmm_spec = pmm_pred.spectrum if hasattr(pmm_pred, 'spectrum') else pmm_pred
                ax.plot(x_grid, pmm_spec, color='#9b59b6', linestyle=':', label='PMM', linewidth=1.5)
            except Exception:
                pass
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(f'{method.capitalize()}\nL2 = {normalized_l2(pred, sample.strength, x_grid):.4f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(f'{spectroscopy_type.upper()} Emulation Method Comparison')
    fig.tight_layout()
    fig.savefig(out_dir / f"{spectroscopy_type}_methods.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    # Best method detailed results
    best_method = min(results, key=lambda m: results[m]["mean_error"])
    best_emu = results[best_method]["emulator"]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for i, ax in enumerate(axes.flat):
        if i >= len(test_ds.samples):
            break
        sample = test_ds.samples[i]
        pred = best_emu.predict_spectrum(sample.params, x_grid)
        err = normalized_l2(pred, sample.strength, x_grid)
        ax.plot(x_grid, sample.strength, 'b-', label='Reference', linewidth=2)
        ax.plot(x_grid, pred, 'r--', label=f'{best_method.capitalize()} Emulator', linewidth=1.5)

        # Overlay PMM prediction if available for direct comparison
        if 'pmm' in results:
            try:
                pmm_emu = results['pmm']['emulator']
                pmm_pred = pmm_emu.predict(sample.params, x_grid)
                pmm_spec = pmm_pred.spectrum if hasattr(pmm_pred, 'spectrum') else pmm_pred
                ax.plot(x_grid, pmm_spec, color='#9b59b6', linestyle=':', label='PMM', linewidth=1.5)
            except Exception:
                pass
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        
        # Show parameters in title
        param_str = ", ".join([f"{n[:3]}={v:.2f}" for n, v in zip(param_names[:3], sample.params[:3])])
        ax.set_title(f'{param_str}\nL2 error = {err:.4f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(f'{spectroscopy_type.upper()} Emulation ({best_method.capitalize()} method)')
    fig.tight_layout()
    fig.savefig(out_dir / f"{spectroscopy_type}_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    # Error distribution comparison (include PMM if available)
    fig, ax = plt.subplots(figsize=(10, 6))
    methods_plot = list(methods)
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    if 'pmm' in results:
        methods_plot.append('pmm')
        colors.append('#9b59b6')

    x_pos = np.arange(len(methods_plot))
    means = [results[m]['mean_error'] for m in methods_plot]
    stds = [np.std(results[m]['errors']) for m in methods_plot]

    bars = ax.bar(x_pos, means, yerr=stds, capsize=5, color=colors[:len(methods_plot)])
    ax.set_xticks(x_pos)
    ax.set_xticklabels([m.capitalize() for m in methods_plot])
    ax.set_ylabel("Mean Normalized L2 Error")
    ax.set_title(f"{spectroscopy_type.upper()} Emulator Accuracy by Method")
    ax.grid(True, alpha=0.3, axis='y')

    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{mean:.4f}', ha='center', va='bottom')
    
    fig.tight_layout()
    fig.savefig(out_dir / f"{spectroscopy_type}_error_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    print(f"\n{'='*60}")
    print(f"Best method: {best_method} (mean L2 = {results[best_method]['mean_error']:.4f})")
    print(f"Plots saved to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Materials spectroscopy emulation demo")
    parser.add_argument("--type", choices=["xanes", "raman"], default="xanes",
                        help="Type of spectroscopy")
    parser.add_argument("--n-train", type=int, default=50)
    parser.add_argument("--n-test", type=int, default=10)
    parser.add_argument("--out", type=Path, default=Path("runs/materials_demo"))
    args = parser.parse_args()
    
    main(
        spectroscopy_type=args.type,
        n_train=args.n_train,
        n_test=args.n_test,
        out_dir=args.out,
    )
