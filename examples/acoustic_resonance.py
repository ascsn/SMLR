"""Acoustic resonance emulation example.

This example demonstrates SMLR applied to acoustic/vibrational spectroscopy,
showing that the Lorentzian mixture approach works beyond nuclear physics.

Use cases:
- Room acoustics and modal analysis
- Musical instrument modeling
- Structural vibration analysis
- Ultrasound transducer characterization
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


def generate_acoustic_spectrum(
    params: np.ndarray,
    frequency: np.ndarray,
) -> np.ndarray:
    """Generate a synthetic acoustic response spectrum.
    
    Simulates the frequency response of an acoustic system (e.g., room modes,
    instrument body resonances) parameterized by physical properties.
    
    Parameters
    ----------
    params : array of shape (3,)
        - params[0]: Damping coefficient (affects peak widths)
        - params[1]: Material stiffness (affects harmonic spacing)
        - params[2]: Coupling strength (affects relative peak heights)
    frequency : array
        Frequency grid in Hz.
        
    Returns
    -------
    array
        Amplitude response (linear scale).
    """
    damping, stiffness, coupling = params
    
    # Fundamental frequency - direct linear mapping from stiffness
    f_fundamental = 150 + 200 * stiffness
    
    # Only 3 harmonics - exact integer multiples
    mode_freqs = np.array([
        f_fundamental,
        f_fundamental * 2.0,
        f_fundamental * 3.0,
    ])
    
    # Strength controlled only by coupling parameter
    mode_strengths = coupling * np.array([1.0, 0.5, 0.25])
    
    # Width controlled only by damping parameter
    widths = (1.5 + 2.0 * damping) * np.ones(3)
    
    return lorentzian_sum(frequency, mode_freqs, mode_strengths, widths)


def build_acoustic_dataset(
    n_samples: int = 50,
    seed: int = 42,
) -> StrengthDataset:
    """Build a dataset of acoustic response spectra.
    
    Parameters vary over physically reasonable ranges:
    - Length scale: 0.5 to 2.0 (normalized room dimension)
    - Damping: 0.1 to 0.8 (low to high absorption)
    - Stiffness: 0.2 to 1.5 (soft to stiff materials)
    - Coupling: 0.5 to 2.0 (source-receiver coupling)
    """
    rng = np.random.default_rng(seed)
    frequency = np.linspace(50, 1000, 300)  # 50 Hz to 1 kHz
    
    # Parameter ranges - tighter for better emulation
    length_range = (0.7, 1.4)
    damping_range = (0.2, 0.6)
    stiffness_range = (0.5, 1.2)
    coupling_range = (0.7, 1.4)
    
    samples = []
    for i in range(n_samples):
        params = np.array([
            rng.uniform(*damping_range),
            rng.uniform(*stiffness_range),
            rng.uniform(*coupling_range),
        ])
        spectrum = generate_acoustic_spectrum(params, frequency)
        samples.append(StrengthSample(
            params=params,
            energy=frequency,  # "energy" is frequency here
            strength=spectrum,
            label=f"acoustic_{i}",
        ))
    
    return StrengthDataset(samples)


def compute_acoustic_observables(
    frequency: np.ndarray,
    spectrum: np.ndarray,
) -> dict:
    """Compute acoustic observables from a spectrum.
    
    Returns
    -------
    dict with keys:
        - total_power: Integrated spectral power
        - centroid: Spectral centroid (brightness)
        - bandwidth: Spectral bandwidth
    """
    # Total integrated power
    total_power = np.trapz(spectrum, frequency)
    
    # Spectral centroid (frequency-weighted average)
    if total_power > 0:
        centroid = np.trapz(spectrum * frequency, frequency) / total_power
    else:
        centroid = 0.0
    
    # Spectral bandwidth (second moment)
    if total_power > 0:
        variance = np.trapz(spectrum * (frequency - centroid)**2, frequency) / total_power
        bandwidth = np.sqrt(variance)
    else:
        bandwidth = 0.0
    
    return {
        "total_power": total_power,
        "centroid": centroid,
        "bandwidth": bandwidth,
    }


def main(
    n_train: int = 150,
    n_test: int = 10,
    out_dir: Path = Path("runs/acoustic_demo"),
) -> None:
    """Run acoustic resonance emulation demo."""
    out_dir.mkdir(parents=True, exist_ok=True)
    print("Acoustic Resonance Emulation Demo")
    print("=" * 50)
    
    # Build dataset
    full_ds = build_acoustic_dataset(n_samples=n_train + n_test)
    train_ds, _, test_ds = full_ds.train_val_test_split(
        train=n_train / (n_train + n_test),
        val=0.0,
        seed=42,
    )
    
    # Define acoustic observables for evaluation
    def spectral_centroid(freq, spec, **kw):
        power = np.trapz(spec, freq)
        return np.trapz(spec * freq, freq) / (power + 1e-10)
    
    obs_set = ObservableSet([
        SumRule(k=0, name="total_power"),  # m_0 = integrated power
        CustomObservable(spectral_centroid, name="centroid"),
    ])
    
    # Fit emulator
    # Fit emulator with PMM backend
    from smlr.backends import get_emulator
    emulator = get_emulator("pmm", n_poles=21)
    print(f"Training on {len(train_ds)} samples...")
    emulator.fit(train_ds)
    
    # Evaluate
    frequency = train_ds.energy_grids()[0]
    errors = []
    observable_errors = {"centroid": [], "total_power": []}
    
    for sample in test_ds.samples:
        result = emulator.predict(sample.params, frequency)
        pred = result.spectrum if hasattr(result, 'spectrum') else result
        err = normalized_l2(pred, sample.strength, frequency)
        errors.append(err)
        
        # Compare observables
        true_obs = obs_set.to_dict(frequency, sample.strength)
        pred_obs = obs_set.to_dict(frequency, pred)
        
        for key in observable_errors:
            rel_err = abs(pred_obs[key] - true_obs[key]) / (abs(true_obs[key]) + 1e-10)
            observable_errors[key].append(rel_err)
    
    mean_err = np.mean(errors)
    print(f"\nTest Results:")
    print(f"  Mean normalized L2 error: {mean_err:.4f}")
    print(f"  Centroid relative error: {np.mean(observable_errors['centroid']):.4f}")
    print(f"  Total power relative error: {np.mean(observable_errors['total_power']):.4f}")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 4 test spectra
    for i, ax in enumerate(axes.flat):
        if i >= len(test_ds.samples):
            break
        sample = test_ds.samples[i]
        result = emulator.predict(sample.params, frequency)
        pred = result.spectrum if hasattr(result, 'spectrum') else result
        
        ax.plot(frequency, sample.strength, 'b-', label='Reference', linewidth=2)
        ax.plot(frequency, pred, 'r--', label='Emulator', linewidth=1.5)
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Amplitude')
        ax.set_title(f'D={sample.params[0]:.2f}, '
                     f'S={sample.params[1]:.2f}, C={sample.params[2]:.2f}\n'
                     f'L2 error = {errors[i]:.4f}')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(50, 1000)
        ax.set_ylim(bottom=0)
    
    fig.suptitle('Acoustic Response Emulation', fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "acoustic_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    # Parameter sensitivity plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    param_names = ['Damping', 'Stiffness', 'Coupling']
    baseline = np.array([0.4, 0.8, 1.0])
    
    for i, (ax, name) in enumerate(zip(axes.flat, param_names)):
        for val in np.linspace(0.2, 1.5, 5):
            params = baseline.copy()
            # Scale to appropriate range for each parameter
            if i == 0:  # Damping
                params[i] = 0.1 + 0.7 * val
            elif i == 1:  # Stiffness
                params[i] = 0.2 + 1.3 * val
            else:  # Coupling
                params[i] = 0.5 + 1.5 * val
            
            result = emulator.predict(params, frequency)
            pred = result.spectrum if hasattr(result, 'spectrum') else result
            ax.plot(frequency, pred, label=f'{name}={params[i]:.2f}')
        
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Amplitude')
        ax.set_title(f'Sensitivity to {name}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(50, 1000)
        ax.set_ylim(bottom=0)
    
    fig.suptitle('Parameter Sensitivity Analysis', fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "acoustic_sensitivity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    print(f"\nPlots saved to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Acoustic resonance emulation demo")
    parser.add_argument("--n-train", type=int, default=150)
    parser.add_argument("--n-test", type=int, default=10)
    parser.add_argument("--out", type=Path, default=Path("runs/acoustic_demo"))
    args = parser.parse_args()
    
    main(n_train=args.n_train, n_test=args.n_test, out_dir=args.out)
