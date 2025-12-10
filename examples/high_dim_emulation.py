"""High-dimensional nuclear response emulation example.

This example demonstrates SMLR's ability to handle 5-15 parameter dimensions,
which is typical of nuclear energy density functional calculations where
multiple coupling constants are varied simultaneously.

Use cases:
- Nuclear mass table generation with varied Skyrme/RMF parameters
- Systematic uncertainty quantification in nuclear structure
- Bayesian inference with high-dimensional posterior sampling
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

from smlr.data import StrengthDataset, StrengthSample
from smlr.emulator import StrengthEmulator
from smlr.lorentz import lorentzian_sum
from smlr.metrics import normalized_l2
from smlr.plotting import plot_comparison


def generate_synthetic_high_dim_spectrum(
    params: np.ndarray, 
    energy: np.ndarray,
    n_poles: int = 5,
) -> np.ndarray:
    """Generate a synthetic strength function that depends on many parameters.
    
    This mimics a QRPA calculation where the pole positions and strengths
    depend on nuclear structure parameters (coupling constants, masses, etc.).
    
    Parameters
    ----------
    params : array of shape (n_params,)
        Physics parameters. Supports any dimension >= 2.
    energy : array
        Energy grid for evaluation.
    n_poles : int
        Number of poles in the synthetic spectrum.
        
    Returns
    -------
    array
        Strength function values on the energy grid.
    """
    n_params = len(params)
    
    # Use parameters to set pole positions (each param affects multiple poles)
    pole_centers = np.zeros(n_poles)
    pole_strengths = np.zeros(n_poles)
    
    for i in range(n_poles):
        # Linear combination of parameters determines pole position
        # Different weights for each pole create complex parameter dependence
        weights = np.sin(np.arange(n_params) * (i + 1) * 0.5) / n_params
        pole_centers[i] = 5.0 + 10.0 * i / n_poles + 2.0 * np.dot(weights, params)
        
        # Strengths also depend on parameters
        strength_weights = np.cos(np.arange(n_params) * (i + 0.5)) / n_params
        pole_strengths[i] = 1.0 + 0.5 * np.dot(strength_weights, params)
    
    # Ensure physical constraints
    pole_strengths = np.maximum(pole_strengths, 0.1)
    
    # Width depends on average of parameters
    width = 0.5 + 0.2 * np.mean(params)
    
    return lorentzian_sum(energy, pole_centers, pole_strengths, np.full(n_poles, width))


def build_high_dim_dataset(
    n_params: int = 5,
    n_samples: int = 100,
    seed: int = 42,
) -> StrengthDataset:
    """Build a synthetic dataset with high-dimensional parameters.
    
    Parameters
    ----------
    n_params : int
        Number of physics parameters (e.g., 5, 10, 15).
    n_samples : int
        Number of training samples.
    seed : int
        Random seed.
        
    Returns
    -------
    StrengthDataset
        Dataset ready for emulator training.
    """
    rng = np.random.default_rng(seed)
    energy = np.linspace(0, 40, 200)
    
    # Latin hypercube-like sampling in parameter space
    samples = []
    for i in range(n_samples):
        # Parameters uniformly distributed in [0, 1]^n_params
        params = rng.uniform(0, 1, size=n_params)
        strength = generate_synthetic_high_dim_spectrum(params, energy)
        samples.append(StrengthSample(
            params=params,
            energy=energy,
            strength=strength,
            label=f"sample_{i}",
        ))
    
    return StrengthDataset(samples)


def evaluate_emulator(
    emulator,
    test_ds: StrengthDataset,
    energy: np.ndarray,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Evaluate emulator on test set.
    
    Returns
    -------
    mean_error : float
        Mean normalized L2 error.
    errors : array
        Per-sample errors.
    predictions : array
        Predicted spectra.
    """
    errors = []
    predictions = []
    
    for sample in test_ds.samples:
        # PMM returns result with .spectrum attribute
        result = emulator.predict(sample.params, energy)
        pred = result.spectrum if hasattr(result, 'spectrum') else result
        err = normalized_l2(pred, sample.strength, energy)
        errors.append(err)
        predictions.append(pred)
    
    return float(np.mean(errors)), np.array(errors), np.array(predictions)


def main(
    n_params: int = 5,
    n_train: int = 80,
    n_test: int = 20,
    n_poles: int = 10,
    backend: str = "pmm",
    out_dir: Path = Path("runs/high_dim_demo"),
) -> None:
    """Run the high-dimensional emulation demo.
    
    Parameters
    ----------
    n_params : int
        Number of physics parameters (2-15 typical).
    n_train : int
        Number of training samples.
    n_test : int
        Number of test samples.
    n_poles : int
        Number of poles for PMM backend.
    backend : str
        Backend to use: "pmm" or "regression".
    out_dir : Path
        Output directory for plots.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"High-dimensional emulation demo: {n_params}D parameter space")
    print(f"Training samples: {n_train}, Test samples: {n_test}")
    print(f"Backend: {backend}")
    
    # Build datasets
    full_ds = build_high_dim_dataset(n_params=n_params, n_samples=n_train + n_test)
    train_ds, _, test_ds = full_ds.train_val_test_split(
        train=n_train / (n_train + n_test),
        val=0.0,
        seed=42,
    )
    
    # Fit emulator with PMM backend
    from smlr.backends import get_emulator
    emulator = get_emulator(backend, n_poles=n_poles)
    print("Fitting emulator...")
    emulator.fit(train_ds)
    
    # Evaluate
    energy = train_ds.energy_grids()[0]
    mean_err, errors, predictions = evaluate_emulator(emulator, test_ds, energy)
    print(f"\nTest set mean normalized L2 error: {mean_err:.4f}")
    print(f"Min error: {errors.min():.4f}, Max error: {errors.max():.4f}")
    
    # Plot some comparisons
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for i, ax in enumerate(axes.flat):
        if i >= len(test_ds.samples):
            break
        sample = test_ds.samples[i]
        pred = predictions[i]
        ax.plot(energy, sample.strength, 'b-', label='Reference', linewidth=2)
        ax.plot(energy, pred, 'r--', label='Emulator', linewidth=2)
        ax.set_xlabel('Energy (MeV)')
        ax.set_ylabel('Strength')
        ax.set_title(f'Sample {i+1}, L2 error = {errors[i]:.4f}')
        ax.legend()
        ax.set_xlim(0, 40)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)
    
    fig.suptitle(f'{n_params}D Parameter Space Emulation (PMM, {n_poles} poles)')
    fig.tight_layout()
    fig.savefig(out_dir / f"high_dim_{n_params}d_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    # Error distribution plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(errors, bins=15, edgecolor='black', alpha=0.7)
    ax.axvline(mean_err, color='r', linestyle='--', linewidth=2, label=f'Mean = {mean_err:.4f}')
    ax.set_xlabel('Normalized L2 Error')
    ax.set_ylabel('Count')
    ax.set_title(f'Error Distribution ({n_params}D, PMM {n_poles} poles)')
    ax.legend()
    fig.savefig(out_dir / f"high_dim_{n_params}d_errors.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    # Save results
    np.savez(
        out_dir / f"high_dim_{n_params}d_results.npz",
        n_params=n_params,
        n_train=n_train,
        n_test=n_test,
        mean_error=mean_err,
        errors=errors,
        backend=backend,
        n_poles=n_poles,
    )
    
    print(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="High-dimensional emulation demo")
    parser.add_argument("--n-params", type=int, default=5, help="Number of parameters (2-15)")
    parser.add_argument("--n-train", type=int, default=80, help="Training samples")
    parser.add_argument("--n-test", type=int, default=20, help="Test samples")
    parser.add_argument("--n-poles", type=int, default=21, help="Number of poles")
    parser.add_argument("--backend", type=str, default="pmm", 
                        choices=["pmm", "regression"],
                        help="Backend")
    parser.add_argument("--out", type=Path, default=Path("runs/high_dim_demo"))
    args = parser.parse_args()
    
    main(
        n_params=args.n_params,
        n_train=args.n_train,
        n_test=args.n_test,
        n_poles=args.n_poles,
        backend=args.backend,
        out_dir=args.out,
    )
