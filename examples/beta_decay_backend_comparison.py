#!/usr/bin/env python3
"""
Beta Decay Backend Comparison: Comprehensive Evaluation
========================================================

This example provides a comprehensive comparison of SMLR emulation backends
using real nuclear beta decay strength function data. We compare:

1. Regression-based emulator (classic SMLR approach)
2. Parametric Matrix Model (PMM) emulator

The comparison evaluates:
- Interpolation accuracy (within training region)
- Extrapolation capability (outside training region)  
- Sum rule preservation (physical constraints)
- Computational efficiency (training and prediction times)

This uses synthetic data modeled after real β-decay strength calculations
for the Ni-80 nucleus.
"""

import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter
from pathlib import Path

# SMLR imports
from smlr.data import StrengthDataset, StrengthSample
from smlr.lorentz import lorentzian_sum
from smlr.backends import get_emulator, list_backends
from smlr.metrics import normalized_l2
from smlr.observables import SumRule


# =============================================================================
# Physical Model: Beta Decay Strength Function
# =============================================================================

def beta_strength(energy: np.ndarray, alpha: float, beta: float,
                  g_A: float = 1.2, Q_value: float = 10.0) -> np.ndarray:
    """
    Generate synthetic β-decay Gamow-Teller strength function.
    
    The strength function S(E) represents the probability distribution
    for β-decay transitions as a function of excitation energy.
    
    Parameters
    ----------
    energy : array
        Energy grid (MeV)
    alpha : float
        Isoscalar pairing strength parameter (0-2 range)
    beta : float  
        Isovector pairing strength parameter (0-1 range)
    g_A : float
        Axial coupling constant (typically 1.2)
    Q_value : float
        Q-value for the decay (MeV)
        
    Returns
    -------
    strength : array
        GT strength function S(E)
        
    Physics Notes
    -------------
    The GT strength encodes nuclear matrix elements:
    
        S(E) = Σ_f |⟨f|σ⃗τ±|i⟩|² δ(E - E_f)
        
    Pairing correlations affect both the centroid and fragmentation
    of the strength distribution.
    """
    # Giant resonance parameters depend on pairing
    E_GTR = 8.0 + 3.0 * alpha - 1.5 * beta      # GT resonance energy
    width_GTR = 3.0 + 0.8 * beta                 # Spreading width
    strength_GTR = 3.0 * (1 + 0.2 * alpha)       # Total strength (Ikeda sum rule)
    
    # Low-lying strength (sensitive to pairing)
    E_low = 2.0 + 1.5 * alpha - 0.5 * beta
    width_low = 0.8 + 0.3 * alpha
    strength_low = 0.5 * (1 - 0.3 * beta)
    
    # High-lying pygmy peak
    E_pygmy = 15.0 + 2.0 * alpha
    width_pygmy = 2.5
    strength_pygmy = 0.3 * (1 + 0.5 * beta)
    
    # Combine peaks
    S = lorentzian_sum(
        energy,
        centers=np.array([E_GTR, E_low, E_pygmy]),
        strengths=np.array([strength_GTR, strength_low, strength_pygmy]),
        widths=np.array([width_GTR, width_low, width_pygmy])
    )
    
    return S


def compute_half_life(energy: np.ndarray, strength: np.ndarray,
                      Q_value: float = 10.0, g_A: float = 1.2) -> float:
    """
    Compute β-decay half-life from strength function.
    
    Uses simplified phase-space integration:
    
        1/t₁/₂ ∝ g_A² ∫₀^Q S(E) f(Q-E) dE
        
    where f(E) is the Fermi phase-space factor.
    
    Parameters
    ----------
    energy : array
        Energy grid (MeV)
    strength : array
        GT strength function
    Q_value : float
        Q-value for decay (MeV)
    g_A : float
        Axial coupling constant
        
    Returns
    -------
    half_life : float
        Half-life in seconds (simplified units)
    """
    # Simplified Fermi phase space
    E_avail = Q_value - energy
    phase_space = np.maximum(E_avail, 0)**5  # ~(Q-E)^5 for allowed decays
    
    # Integrate S(E) * f(Q-E)
    integrand = strength * phase_space
    rate = g_A**2 * np.trapezoid(integrand, energy)
    
    # Convert to half-life (arbitrary units, scaled)
    if rate > 0:
        half_life = np.log(2) / (rate * 1e-6)  # Scale to ~seconds
    else:
        half_life = np.inf
        
    return half_life


# =============================================================================
# Dataset Generation
# =============================================================================

def generate_beta_decay_dataset(
    n_alpha: int = 8,
    n_beta: int = 8,
    alpha_range: tuple = (0.2, 1.8),
    beta_range: tuple = (0.1, 0.9),
    energy_range: tuple = (0, 25),
    n_energy: int = 200,
    seed: int = 42
) -> StrengthDataset:
    """
    Generate a grid of β-decay strength functions.
    
    Parameters
    ----------
    n_alpha : int
        Number of alpha grid points
    n_beta : int
        Number of beta grid points
    alpha_range : tuple
        (min, max) for alpha parameter
    beta_range : tuple
        (min, max) for beta parameter
    energy_range : tuple
        (min, max) for energy grid (MeV)
    n_energy : int
        Number of energy points
    seed : int
        Random seed for reproducibility
        
    Returns
    -------
    dataset : StrengthDataset
        Dataset of strength samples
    """
    rng = np.random.default_rng(seed)
    energy = np.linspace(energy_range[0], energy_range[1], n_energy)
    
    alphas = np.linspace(alpha_range[0], alpha_range[1], n_alpha)
    betas = np.linspace(beta_range[0], beta_range[1], n_beta)
    
    samples = []
    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            strength = beta_strength(energy, alpha, beta)
            # Add small noise to simulate numerical error
            noise = rng.normal(0, 0.01 * np.max(strength), size=len(energy))
            strength = np.maximum(strength + noise, 0)
            
            samples.append(StrengthSample(
                params=np.array([alpha, beta]),
                energy=energy,
                strength=strength,
                label=f"alpha={alpha:.2f}_beta={beta:.2f}"
            ))
    
    return StrengthDataset(samples)


def split_dataset(dataset: StrengthDataset, 
                  train_fraction: float = 0.7,
                  seed: int = 42) -> tuple:
    """Split dataset into training and test sets."""
    rng = np.random.default_rng(seed)
    n = len(dataset.samples)
    indices = rng.permutation(n)
    n_train = int(n * train_fraction)
    
    train_samples = [dataset.samples[i] for i in indices[:n_train]]
    test_samples = [dataset.samples[i] for i in indices[n_train:]]
    
    return StrengthDataset(train_samples), StrengthDataset(test_samples)


def generate_extrapolation_samples(
    n_samples: int = 20,
    energy_range: tuple = (0, 25),
    n_energy: int = 200,
    seed: int = 123
) -> StrengthDataset:
    """
    Generate samples outside the training region for extrapolation testing.
    
    These samples have parameters outside the normal training range:
    - alpha in (1.8, 2.2) -- beyond training max of 1.8
    - beta in (0.9, 1.1) -- beyond training max of 0.9
    """
    rng = np.random.default_rng(seed)
    energy = np.linspace(energy_range[0], energy_range[1], n_energy)

    # Create a grid of extrapolation points and add small jitter so points
    # are well-distributed instead of accidentally clustered.
    n_side = int(np.ceil(np.sqrt(n_samples)))
    alphas = np.linspace(1.81, 2.2, n_side)
    betas = np.linspace(0.91, 1.09, n_side)
    A, B = np.meshgrid(alphas, betas)
    coords = np.stack([A.ravel(), B.ravel()], axis=1)

    samples = []
    for i in range(min(n_samples, len(coords))):
        alpha, beta = coords[i]
        # small jitter to avoid perfect grid alignment
        alpha += rng.normal(0, 0.008)
        beta += rng.normal(0, 0.008)

        strength = beta_strength(energy, alpha, beta)

        samples.append(StrengthSample(
            params=np.array([alpha, beta]),
            energy=energy,
            strength=strength,
            label=f"extrap_alpha={alpha:.2f}_beta={beta:.2f}"
        ))

    # If n_samples is larger than grid, fill the remainder with random draws
    i = len(samples)
    while i < n_samples:
        alpha = rng.uniform(1.81, 2.2)
        beta = rng.uniform(0.91, 1.09)
        strength = beta_strength(energy, alpha, beta)
        samples.append(StrengthSample(
            params=np.array([alpha, beta]),
            energy=energy,
            strength=strength,
            label=f"extrap_alpha={alpha:.2f}_beta={beta:.2f}"
        ))
        i += 1

    return StrengthDataset(samples)


# =============================================================================
# Backend Comparison
# =============================================================================

def compare_backends(
    train_data: StrengthDataset,
    test_data: StrengthDataset,
    extrap_data: StrengthDataset,
    n_poles: int = 5,
    pmm_n: int = 8,
    verbose: bool = True
) -> dict:
    """
    Comprehensive comparison of emulation backends.
    
    Parameters
    ----------
    train_data : StrengthDataset
        Training data
    test_data : StrengthDataset
        Test data (interpolation)
    extrap_data : StrengthDataset
        Extrapolation test data
    n_poles : int
        Number of Lorentzian poles for regression backend
    pmm_n : int
        Matrix dimension for PMM backend
    verbose : bool
        Print progress
        
    Returns
    -------
    results : dict
        Comparison metrics for each backend
    """
    results = {}
    
    # Define sum rule for checking conservation
    m0_rule = SumRule(k=0, name="m0")  # Just compute, don't enforce
    
    # -----------------------------------------------------------------
    # Regression Backend
    # -----------------------------------------------------------------
    if verbose:
        print("\n" + "="*60)
        print("REGRESSION BACKEND")
        print("="*60)
    
    # Train
    t0 = perf_counter()
    reg_emulator = get_emulator("regression", n_poles=n_poles)
    reg_emulator.fit(train_data)
    reg_train_time = perf_counter() - t0
    
    # Predict on test set (interpolation)
    reg_interp_errors = []
    reg_interp_m0_true = []
    reg_interp_m0_pred = []
    
    t0 = perf_counter()
    for sample in test_data.samples:
        # Regression emulator returns (mixture, spectrum) when given energy
        _, pred_spectrum = reg_emulator.predict(sample.params, sample.energy)
        error = normalized_l2(pred_spectrum, sample.strength, sample.energy)
        reg_interp_errors.append(error)
        
        m0_true = m0_rule.compute(sample.energy, sample.strength)
        m0_pred = m0_rule.compute(sample.energy, pred_spectrum)
        reg_interp_m0_true.append(m0_true)
        reg_interp_m0_pred.append(m0_pred)
    reg_interp_time = perf_counter() - t0
    
    # Predict on extrapolation set
    reg_extrap_errors = []
    reg_extrap_m0_true = []
    reg_extrap_m0_pred = []
    
    for sample in extrap_data.samples:
        _, pred_spectrum = reg_emulator.predict(sample.params, sample.energy)
        error = normalized_l2(pred_spectrum, sample.strength, sample.energy)
        reg_extrap_errors.append(error)
        
        m0_true = m0_rule.compute(sample.energy, sample.strength)
        m0_pred = m0_rule.compute(sample.energy, pred_spectrum)
        reg_extrap_m0_true.append(m0_true)
        reg_extrap_m0_pred.append(m0_pred)
    
    results["regression"] = {
        "train_time": reg_train_time,
        "predict_time": reg_interp_time,
        "interp_errors": np.array(reg_interp_errors),
        "extrap_errors": np.array(reg_extrap_errors),
        "interp_m0_deviation": np.abs(np.array(reg_interp_m0_pred) - np.array(reg_interp_m0_true)) / np.array(reg_interp_m0_true),
        "extrap_m0_deviation": np.abs(np.array(reg_extrap_m0_pred) - np.array(reg_extrap_m0_true)) / np.array(reg_extrap_m0_true),
    }
    
    if verbose:
        print(f"\nTraining time: {reg_train_time:.3f} s")
        print(f"Prediction time ({len(test_data.samples)} samples): {reg_interp_time:.3f} s")
        print(f"Interpolation error: {np.mean(reg_interp_errors)*100:.2f}% ± {np.std(reg_interp_errors)*100:.2f}%")
        print(f"Extrapolation error: {np.mean(reg_extrap_errors)*100:.2f}% ± {np.std(reg_extrap_errors)*100:.2f}%")
        print(f"m₀ sum rule deviation (interp): {np.mean(results['regression']['interp_m0_deviation'])*100:.2f}%")
        print(f"m₀ sum rule deviation (extrap): {np.mean(results['regression']['extrap_m0_deviation'])*100:.2f}%")
    
    # -----------------------------------------------------------------
    # PMM Backend
    # -----------------------------------------------------------------
    if verbose:
        print("\n" + "="*60)
        print("PARAMETRIC MATRIX MODEL BACKEND")
        print("="*60)
    
    # Train
    t0 = perf_counter()
    pmm_emulator = get_emulator("pmm", n_poles=pmm_n)
    pmm_emulator.fit(train_data)
    pmm_train_time = perf_counter() - t0
    
    # Predict on test set (interpolation)
    pmm_interp_errors = []
    pmm_interp_m0_true = []
    pmm_interp_m0_pred = []
    
    t0 = perf_counter()
    for sample in test_data.samples:
        # PMM returns PMMResult with .spectrum attribute
        pred = pmm_emulator.predict(sample.params, sample.energy)
        error = normalized_l2(pred.spectrum, sample.strength, sample.energy)
        pmm_interp_errors.append(error)
        
        m0_true = m0_rule.compute(sample.energy, sample.strength)
        m0_pred = m0_rule.compute(sample.energy, pred.spectrum)
        pmm_interp_m0_true.append(m0_true)
        pmm_interp_m0_pred.append(m0_pred)
    pmm_interp_time = perf_counter() - t0
    
    # Predict on extrapolation set
    pmm_extrap_errors = []
    pmm_extrap_m0_true = []
    pmm_extrap_m0_pred = []
    
    for sample in extrap_data.samples:
        pred = pmm_emulator.predict(sample.params, sample.energy)
        error = normalized_l2(pred.spectrum, sample.strength, sample.energy)
        pmm_extrap_errors.append(error)
        
        m0_true = m0_rule.compute(sample.energy, sample.strength)
        m0_pred = m0_rule.compute(sample.energy, pred.spectrum)
        pmm_extrap_m0_true.append(m0_true)
        pmm_extrap_m0_pred.append(m0_pred)
    
    results["pmm"] = {
        "train_time": pmm_train_time,
        "predict_time": pmm_interp_time,
        "interp_errors": np.array(pmm_interp_errors),
        "extrap_errors": np.array(pmm_extrap_errors),
        "interp_m0_deviation": np.abs(np.array(pmm_interp_m0_pred) - np.array(pmm_interp_m0_true)) / np.array(pmm_interp_m0_true),
        "extrap_m0_deviation": np.abs(np.array(pmm_extrap_m0_pred) - np.array(pmm_extrap_m0_true)) / np.array(pmm_extrap_m0_true),
    }
    
    if verbose:
        print(f"\nTraining time: {pmm_train_time:.3f} s")
        print(f"Prediction time ({len(test_data.samples)} samples): {pmm_interp_time:.3f} s")
        print(f"Interpolation error: {np.mean(pmm_interp_errors)*100:.2f}% ± {np.std(pmm_interp_errors)*100:.2f}%")
        print(f"Extrapolation error: {np.mean(pmm_extrap_errors)*100:.2f}% ± {np.std(pmm_extrap_errors)*100:.2f}%")
        print(f"m₀ sum rule deviation (interp): {np.mean(results['pmm']['interp_m0_deviation'])*100:.2f}%")
        print(f"m₀ sum rule deviation (extrap): {np.mean(results['pmm']['extrap_m0_deviation'])*100:.2f}%")
    
    return results


# =============================================================================
# Visualization
# =============================================================================

def plot_comparison(
    train_data: StrengthDataset,
    test_data: StrengthDataset,
    extrap_data: StrengthDataset,
    results: dict,
    save_path: str = None
):
    """Create comprehensive comparison plots."""
    
    fig = plt.figure(figsize=(16, 12))
    
    # Get reference sample for spectrum comparison
    test_sample = test_data.samples[0]
    extrap_sample = extrap_data.samples[0]
    
    # Re-create emulators for plotting
    reg_emulator = get_emulator("regression", n_poles=5)
    reg_emulator.fit(train_data)
    pmm_emulator = get_emulator("pmm", n_poles=8)
    pmm_emulator.fit(train_data)
    
    # -----------------------------------------------------------------
    # Subplot 1: Interpolation spectrum comparison
    # -----------------------------------------------------------------
    ax1 = fig.add_subplot(2, 3, 1)
    
    _, reg_spec = reg_emulator.predict(test_sample.params, test_sample.energy)
    pmm_result = pmm_emulator.predict(test_sample.params, test_sample.energy)
    
    ax1.plot(test_sample.energy, test_sample.strength, 'k-', lw=2, label='True')
    ax1.plot(test_sample.energy, reg_spec, 'b--', lw=1.5, label='Regression')
    ax1.plot(test_sample.energy, pmm_result.spectrum, 'r:', lw=1.5, label='PMM')
    
    ax1.set_xlabel('Energy (MeV)')
    ax1.set_ylabel('Strength')
    ax1.set_title(f'Interpolation Test\nα={test_sample.params[0]:.2f}, β={test_sample.params[1]:.2f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    # tighten y-limits to relevant data range to avoid excess whitespace
    ymax = max(np.max(test_sample.strength), np.max(reg_spec), np.max(pmm_result.spectrum))
    ax1.set_ylim(0, ymax * 1.05)
    
    # -----------------------------------------------------------------
    # Subplot 2: Extrapolation spectrum comparison
    # -----------------------------------------------------------------
    ax2 = fig.add_subplot(2, 3, 2)
    
    _, reg_spec_ext = reg_emulator.predict(extrap_sample.params, extrap_sample.energy)
    pmm_result_ext = pmm_emulator.predict(extrap_sample.params, extrap_sample.energy)
    
    ax2.plot(extrap_sample.energy, extrap_sample.strength, 'k-', lw=2, label='True')
    ax2.plot(extrap_sample.energy, reg_spec_ext, 'b--', lw=1.5, label='Regression')
    ax2.plot(extrap_sample.energy, pmm_result_ext.spectrum, 'r:', lw=1.5, label='PMM')
    
    ax2.set_xlabel('Energy (MeV)')
    ax2.set_ylabel('Strength')
    ax2.set_title(f'Extrapolation Test\nα={extrap_sample.params[0]:.2f}, β={extrap_sample.params[1]:.2f}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ymax_ext = max(np.max(extrap_sample.strength), np.max(reg_spec_ext), np.max(pmm_result_ext.spectrum))
    ax2.set_ylim(0, ymax_ext * 1.05)
    
    # -----------------------------------------------------------------
    # Subplot 3: Error distribution comparison
    # -----------------------------------------------------------------
    ax3 = fig.add_subplot(2, 3, 3)
    
    x = ['Interp\n(Reg)', 'Interp\n(PMM)', 'Extrap\n(Reg)', 'Extrap\n(PMM)']
    means = [
        np.mean(results['regression']['interp_errors']) * 100,
        np.mean(results['pmm']['interp_errors']) * 100,
        np.mean(results['regression']['extrap_errors']) * 100,
        np.mean(results['pmm']['extrap_errors']) * 100,
    ]
    stds = [
        np.std(results['regression']['interp_errors']) * 100,
        np.std(results['pmm']['interp_errors']) * 100,
        np.std(results['regression']['extrap_errors']) * 100,
        np.std(results['pmm']['extrap_errors']) * 100,
    ]
    colors = ['blue', 'red', 'blue', 'red']
    
    bars = ax3.bar(x, means, yerr=stds, color=colors, alpha=0.7, capsize=5)
    ax3.set_ylabel('Normalized L² Error (%)')
    ax3.set_title('Error Comparison')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # -----------------------------------------------------------------
    # Subplot 4: Sum rule preservation
    # -----------------------------------------------------------------
    ax4 = fig.add_subplot(2, 3, 4)
    
    x = ['Interp\n(Reg)', 'Interp\n(PMM)', 'Extrap\n(Reg)', 'Extrap\n(PMM)']
    m0_devs = [
        np.mean(results['regression']['interp_m0_deviation']) * 100,
        np.mean(results['pmm']['interp_m0_deviation']) * 100,
        np.mean(results['regression']['extrap_m0_deviation']) * 100,
        np.mean(results['pmm']['extrap_m0_deviation']) * 100,
    ]
    colors = ['blue', 'red', 'blue', 'red']
    
    ax4.bar(x, m0_devs, color=colors, alpha=0.7)
    ax4.set_ylabel('m₀ Sum Rule Deviation (%)')
    ax4.set_title('Sum Rule Preservation')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # -----------------------------------------------------------------
    # Subplot 5: Training parameter space coverage
    # -----------------------------------------------------------------
    ax5 = fig.add_subplot(2, 3, 5)
    
    # Training points
    train_alphas = [s.params[0] for s in train_data.samples]
    train_betas = [s.params[1] for s in train_data.samples]
    ax5.scatter(train_alphas, train_betas, c='green', s=50, alpha=0.7, label='Training')
    
    # Test points
    test_alphas = [s.params[0] for s in test_data.samples]
    test_betas = [s.params[1] for s in test_data.samples]
    ax5.scatter(test_alphas, test_betas, c='blue', s=50, alpha=0.7, marker='s', label='Interp Test')
    
    # Extrapolation points
    ext_alphas = [s.params[0] for s in extrap_data.samples]
    ext_betas = [s.params[1] for s in extrap_data.samples]
    ax5.scatter(ext_alphas, ext_betas, c='red', s=50, alpha=0.7, marker='^', label='Extrap Test')
    
    ax5.set_xlabel('α (isoscalar pairing)')
    ax5.set_ylabel('β (isovector pairing)')
    ax5.set_title('Parameter Space Coverage')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    # set explicit axis limits to show extrapolation region clearly with padding
    all_alphas = train_alphas + test_alphas + ext_alphas
    all_betas = train_betas + test_betas + ext_betas
    pad_a = (max(all_alphas) - min(all_alphas)) * 0.08
    pad_b = (max(all_betas) - min(all_betas)) * 0.08
    ax5.set_xlim(min(all_alphas) - pad_a, max(all_alphas) + pad_a)
    ax5.set_ylim(min(all_betas) - pad_b, max(all_betas) + pad_b)
    
    # -----------------------------------------------------------------
    # Subplot 6: Timing comparison
    # -----------------------------------------------------------------
    ax6 = fig.add_subplot(2, 3, 6)
    
    x = ['Training', 'Prediction\n(per batch)']
    reg_times = [results['regression']['train_time'], results['regression']['predict_time']]
    pmm_times = [results['pmm']['train_time'], results['pmm']['predict_time']]
    
    width = 0.35
    x_pos = np.arange(len(x))
    
    ax6.bar(x_pos - width/2, reg_times, width, label='Regression', color='blue', alpha=0.7)
    ax6.bar(x_pos + width/2, pmm_times, width, label='PMM', color='red', alpha=0.7)
    
    ax6.set_ylabel('Time (seconds)')
    ax6.set_title('Computational Cost')
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(x)
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.35, wspace=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    plt.show()


def print_summary_table(results: dict):
    """Print a formatted summary table."""
    print("\n" + "="*70)
    print("SUMMARY: BACKEND COMPARISON FOR β-DECAY STRENGTH EMULATION")
    print("="*70)
    
    print("\n{:<25} {:>15} {:>15}".format("Metric", "Regression", "PMM"))
    print("-"*55)
    
    # Errors
    print("{:<25} {:>14.2f}% {:>14.2f}%".format(
        "Interpolation Error",
        np.mean(results['regression']['interp_errors']) * 100,
        np.mean(results['pmm']['interp_errors']) * 100
    ))
    print("{:<25} {:>14.2f}% {:>14.2f}%".format(
        "Extrapolation Error",
        np.mean(results['regression']['extrap_errors']) * 100,
        np.mean(results['pmm']['extrap_errors']) * 100
    ))
    
    # Sum rules
    print("{:<25} {:>14.2f}% {:>14.2f}%".format(
        "m₀ Deviation (interp)",
        np.mean(results['regression']['interp_m0_deviation']) * 100,
        np.mean(results['pmm']['interp_m0_deviation']) * 100
    ))
    print("{:<25} {:>14.2f}% {:>14.2f}%".format(
        "m₀ Deviation (extrap)",
        np.mean(results['regression']['extrap_m0_deviation']) * 100,
        np.mean(results['pmm']['extrap_m0_deviation']) * 100
    ))
    
    # Timing
    print("{:<25} {:>14.3f}s {:>14.3f}s".format(
        "Training Time",
        results['regression']['train_time'],
        results['pmm']['train_time']
    ))
    print("{:<25} {:>14.3f}s {:>14.3f}s".format(
        "Prediction Time",
        results['regression']['predict_time'],
        results['pmm']['predict_time']
    ))
    
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    
    reg_extrap = np.mean(results['regression']['extrap_errors'])
    pmm_extrap = np.mean(results['pmm']['extrap_errors'])
    
    if pmm_extrap < reg_extrap * 0.7:
        print("✓ PMM recommended for extrapolation-heavy applications")
    
    reg_m0 = np.mean(results['regression']['extrap_m0_deviation'])
    pmm_m0 = np.mean(results['pmm']['extrap_m0_deviation'])
    
    if pmm_m0 < reg_m0 * 0.5:
        print("✓ PMM recommended when sum rule preservation is critical")
    
    if results['regression']['train_time'] < results['pmm']['train_time'] * 0.5:
        print("✓ Regression recommended for rapid prototyping")
    
    print("="*70)


# =============================================================================
# Main
# =============================================================================

def main():
    print("="*70)
    print("β-DECAY STRENGTH FUNCTION EMULATION: BACKEND COMPARISON")
    print("="*70)
    print("\nAvailable backends:", list_backends())
    
    # Generate datasets
    print("\n[1/4] Generating training data...")
    full_data = generate_beta_decay_dataset(n_alpha=8, n_beta=8, seed=42)
    train_data, test_data = split_dataset(full_data, train_fraction=0.75, seed=42)
    print(f"      Training samples: {len(train_data.samples)}")
    print(f"      Test samples: {len(test_data.samples)}")
    
    print("\n[2/4] Generating extrapolation test data...")
    extrap_data = generate_extrapolation_samples(n_samples=20, seed=123)
    print(f"      Extrapolation samples: {len(extrap_data.samples)}")
    
    # Compare backends
    print("\n[3/4] Comparing backends...")
    results = compare_backends(
        train_data, test_data, extrap_data,
        n_poles=5, pmm_n=8, verbose=True
    )
    
    # Summary
    print_summary_table(results)
    
    # Visualization
    print("\n[4/4] Creating visualization...")
    output_dir = Path(__file__).parent.parent / "docs" / "examples" / "figs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plot_comparison(
        train_data, test_data, extrap_data, results,
        save_path=output_dir / "beta_decay_comparison.png"
    )
    
    print("\nDone!")


if __name__ == "__main__":
    main()
