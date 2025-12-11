#!/usr/bin/env python
"""Generate plots for the documentation examples.

This script runs the example scenarios and saves matplotlib figures
to docs/examples/figs/ for embedding in the documentation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from smlr.backends import get_emulator
from smlr.data import StrengthDataset, StrengthSample
from smlr.lorentz import lorentzian_sum
from smlr.metrics import normalized_l2
from smlr.pmm import ParametricMatrixModel

# Output directory
FIGS_DIR = Path(__file__).parent.parent / "docs" / "examples" / "figs"
FIGS_DIR.mkdir(parents=True, exist_ok=True)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (8, 5)
plt.rcParams['font.size'] = 12


# =============================================================================
# 1. High-Dimensional Nuclear Response
# =============================================================================

def generate_nuclear_spectrum(params, energy, n_poles=5):
    """Generate synthetic QRPA-like spectrum with N-dimensional parameters."""
    n_params = len(params)
    pole_centers = np.zeros(n_poles)
    pole_strengths = np.zeros(n_poles)
    
    for i in range(n_poles):
        weights = np.sin(np.arange(n_params) * (i + 1) * 0.5) / n_params
        pole_centers[i] = 5.0 + 10.0 * i / n_poles + 2.0 * np.dot(weights, params)
        strength_weights = np.cos(np.arange(n_params) * (i + 0.5)) / n_params
        pole_strengths[i] = 1.0 + 0.5 * np.dot(strength_weights, params)
    
    pole_strengths = np.maximum(pole_strengths, 0.1)
    width = 0.5 + 0.2 * np.mean(params)
    return lorentzian_sum(energy, pole_centers, pole_strengths, np.full(n_poles, width))


def build_nuclear_dataset(n_params=10, n_samples=100, seed=42):
    rng = np.random.default_rng(seed)
    energy = np.linspace(0, 40, 200)
    samples = []
    for i in range(n_samples):
        params = rng.uniform(-1, 1, n_params)
        spectrum = generate_nuclear_spectrum(params, energy)
        samples.append(StrengthSample(params=params, energy=energy, 
                                      strength=spectrum, label=f"sample_{i}"))
    return StrengthDataset(samples)


def plot_nuclear_example():
    """Generate plots for nuclear response example."""
    print("Generating nuclear response plots...")
    
    # Build dataset
    n_params = 10
    dataset = build_nuclear_dataset(n_params=n_params, n_samples=150)
    energy = np.linspace(0, 40, 200)
    
    # Train emulator
    emu = get_emulator("regression", n_components=5, regression_method="ridge")
    emu.fit(dataset)
    
    # Test on new points
    rng = np.random.default_rng(999)
    test_params = rng.uniform(-1, 1, (5, n_params))
    
    # Plot comparisons
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    errors = []
    for idx, (ax, params) in enumerate(zip(axes.flat[:4], test_params[:4])):
        true = generate_nuclear_spectrum(params, energy)
        pred_mix = emu.predict_mixture(params)
        pred = lorentzian_sum(energy, pred_mix.energies, pred_mix.strengths, pred_mix.widths)
        error = normalized_l2(true, pred, energy)
        errors.append(error)
        
        ax.plot(energy, true, 'b-', linewidth=2, label='True (QRPA-like)')
        ax.plot(energy, pred, 'r--', linewidth=2, label=f'Emulator (L²={error:.1%})')
        ax.fill_between(energy, true, pred, alpha=0.2, color='gray')
        ax.set_xlabel('Energy (MeV)')
        ax.set_ylabel('S(E)')
        ax.set_title(f'Test point {idx+1} (10D parameter)')
        ax.legend(loc='upper right')
    
    plt.suptitle(f'High-Dimensional Emulation: Mean Error = {np.mean(errors):.1%}', fontsize=14)
    plt.tight_layout()
    plt.savefig(FIGS_DIR / "nuclear_response_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Scaling plot
    fig, ax = plt.subplots(figsize=(8, 5))
    dims = [3, 5, 7, 10]
    samples_per_dim = [30, 50, 80, 150]
    mean_errors = []
    
    for d, n_samp in zip(dims, samples_per_dim):
        ds = build_nuclear_dataset(n_params=d, n_samples=n_samp)
        emu = get_emulator("regression", n_components=5, regression_method="ridge")
        emu.fit(ds)
        
        errs = []
        for _ in range(10):
            p = rng.uniform(-1, 1, d)
            true = generate_nuclear_spectrum(p, energy)
            pred_mix = emu.predict_mixture(p)
            pred = lorentzian_sum(energy, pred_mix.energies, pred_mix.strengths, pred_mix.widths)
            errs.append(normalized_l2(true, pred, energy))
        mean_errors.append(np.mean(errs))
    
    ax.bar(range(len(dims)), [e*100 for e in mean_errors], color='steelblue', alpha=0.8)
    ax.set_xticks(range(len(dims)))
    ax.set_xticklabels([f'{d}D\n({n}s)' for d, n in zip(dims, samples_per_dim)])
    ax.set_ylabel('Mean L² Error (%)')
    ax.set_xlabel('Parameter Dimension (samples)')
    ax.set_title('Emulator Accuracy vs Parameter Dimension')
    ax.set_ylim(0, max(mean_errors)*100 * 1.3)
    for i, e in enumerate(mean_errors):
        ax.text(i, e*100 + 0.3, f'{e:.1%}', ha='center', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(FIGS_DIR / "nuclear_response_scaling.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved nuclear_response_comparison.png and nuclear_response_scaling.png")


# =============================================================================
# 2. Acoustic Resonance
# =============================================================================

def generate_acoustic_spectrum(params, frequency):
    """Generate synthetic acoustic response spectrum."""
    length_scale, damping, stiffness, coupling = params
    f_fundamental = 100 * (1 + stiffness) / (1 + 0.5 * length_scale)
    
    n_modes = 6
    mode_freqs = []
    mode_strengths = []
    
    for n in range(1, n_modes + 1):
        inharmonicity = 1 + 0.002 * n**2 * stiffness
        freq = f_fundamental * n * inharmonicity
        strength = coupling / (n ** (0.5 + 0.3 * damping))
        mode_freqs.append(freq)
        mode_strengths.append(strength)
    
    widths = 5 * (1 + 2 * damping) * np.ones(n_modes)
    return lorentzian_sum(frequency, mode_freqs, mode_strengths, widths)


def build_acoustic_dataset(n_samples=50, seed=42):
    rng = np.random.default_rng(seed)
    frequency = np.linspace(50, 1000, 300)
    samples = []
    for i in range(n_samples):
        params = np.array([
            rng.uniform(0.5, 2.0),   # length
            rng.uniform(0.1, 0.8),   # damping
            rng.uniform(0.2, 1.5),   # stiffness
            rng.uniform(0.5, 2.0),   # coupling
        ])
        spectrum = generate_acoustic_spectrum(params, frequency)
        samples.append(StrengthSample(params=params, energy=frequency,
                                      strength=spectrum, label=f"acoustic_{i}"))
    return StrengthDataset(samples)


def plot_acoustic_example():
    """Generate plots for acoustic resonance example."""
    print("Generating acoustic resonance plots...")
    
    dataset = build_acoustic_dataset(n_samples=60)
    frequency = np.linspace(50, 1000, 300)
    
    # Train emulator
    emu = get_emulator("regression", n_components=6)
    emu.fit(dataset)
    
    # Show damping variation
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    damping_vals = [0.15, 0.45, 0.75]
    base_params = [1.0, 0.0, 0.8, 1.2]
    
    for ax, damp in zip(axes, damping_vals):
        params = np.array([base_params[0], damp, base_params[2], base_params[3]])
        true = generate_acoustic_spectrum(params, frequency)
        pred_mix = emu.predict_mixture(params)
        pred = lorentzian_sum(frequency, pred_mix.energies, pred_mix.strengths, pred_mix.widths)
        error = normalized_l2(true, pred, frequency)
        
        ax.plot(frequency, true, 'b-', linewidth=2, label='True')
        ax.plot(frequency, pred, 'r--', linewidth=2, label=f'Emulator')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Amplitude')
        ax.set_title(f'Damping = {damp:.2f} (error: {error:.1%})')
        ax.legend()
    
    plt.suptitle('Acoustic Response: Effect of Damping', fontsize=14)
    plt.tight_layout()
    plt.savefig(FIGS_DIR / "acoustic_damping_variation.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Show harmonic structure
    fig, ax = plt.subplots(figsize=(10, 5))
    
    params = np.array([1.2, 0.3, 0.8, 1.5])
    true = generate_acoustic_spectrum(params, frequency)
    pred_mix = emu.predict_mixture(params)
    pred = lorentzian_sum(frequency, pred_mix.energies, pred_mix.strengths, pred_mix.widths)
    
    ax.plot(frequency, true, 'b-', linewidth=2.5, label='True spectrum')
    ax.plot(frequency, pred, 'r--', linewidth=2, label='Emulator prediction')
    
    # Mark peaks
    for i, (e, s) in enumerate(zip(pred_mix.energies, pred_mix.strengths)):
        if e > 50 and e < 1000:
            ax.axvline(e, color='orange', alpha=0.5, linestyle=':')
            ax.annotate(f'Mode {i+1}', (e, max(pred)*0.9 - i*0.08*max(pred)), 
                       fontsize=9, ha='center')
    
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Amplitude Response')
    ax.set_title('Acoustic Resonance: Harmonic Mode Structure')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(FIGS_DIR / "acoustic_harmonic_structure.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved acoustic_damping_variation.png and acoustic_harmonic_structure.png")


# =============================================================================
# 3. Materials Spectroscopy
# =============================================================================

def generate_materials_spectrum(params, energy):
    """Generate synthetic XAS/optical absorption spectrum."""
    band_gap, spin_orbit, strain, temperature, doping = params
    
    effective_gap = band_gap * (1 - 5 * strain)
    thermal_width = 0.026 * temperature / 300
    
    n_transitions = 5
    positions = []
    strengths = []
    
    # Excitonic peak
    exciton_binding = 0.1 * band_gap
    positions.append(effective_gap - exciton_binding)
    strengths.append(0.3 * (1 - doping * 5))
    
    # Band edge
    positions.append(effective_gap)
    strengths.append(1.0)
    
    # Spin-orbit split peak
    positions.append(effective_gap + spin_orbit)
    strengths.append(0.5)
    
    # Higher transitions
    positions.append(effective_gap + 2 * spin_orbit)
    strengths.append(0.2)
    positions.append(effective_gap + 3 * spin_orbit + 0.5)
    strengths.append(0.1)
    
    positions = np.array(positions)
    strengths = np.array(strengths)
    
    base_width = 0.1 + thermal_width
    widths = base_width * (1 + 0.2 * np.arange(n_transitions))
    
    return lorentzian_sum(energy, positions, strengths, widths)


def build_materials_dataset(n_samples=80, seed=42):
    rng = np.random.default_rng(seed)
    energy = np.linspace(0.5, 5.0, 250)
    samples = []
    for i in range(n_samples):
        params = np.array([
            rng.uniform(1.0, 2.5),      # band_gap
            rng.uniform(0.2, 0.6),      # spin_orbit
            rng.uniform(-0.02, 0.02),   # strain
            rng.uniform(150, 400),      # temperature
            rng.uniform(0, 0.05),       # doping
        ])
        spectrum = generate_materials_spectrum(params, energy)
        samples.append(StrengthSample(params=params, energy=energy,
                                      strength=spectrum, label=f"xas_{i}"))
    return StrengthDataset(samples)


def plot_materials_example():
    """Generate plots for materials spectroscopy example."""
    print("Generating materials spectroscopy plots...")
    
    dataset = build_materials_dataset(n_samples=80)
    energy = np.linspace(0.5, 5.0, 250)
    
    # Train emulator
    emu = get_emulator("regression", n_components=5, regression_method="polynomial", poly_degree=2)
    emu.fit(dataset)
    
    # Show strain variation
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    strains = [-0.015, 0.0, 0.015]
    base_params = [1.8, 0.4, 0.0, 300, 0.02]
    
    for ax, strain in zip(axes, strains):
        params = np.array([base_params[0], base_params[1], strain, base_params[3], base_params[4]])
        true = generate_materials_spectrum(params, energy)
        pred_mix = emu.predict_mixture(params)
        pred = lorentzian_sum(energy, pred_mix.energies, pred_mix.strengths, pred_mix.widths)
        error = normalized_l2(true, pred, energy)
        
        ax.plot(energy, true, 'b-', linewidth=2, label='True')
        ax.plot(energy, pred, 'r--', linewidth=2, label='Emulator')
        ax.set_xlabel('Energy (eV)')
        ax.set_ylabel('Absorption')
        strain_pct = strain * 100
        ax.set_title(f'Strain = {strain_pct:+.1f}% (error: {error:.1%})')
        ax.legend()
    
    plt.suptitle('XAS Spectrum: Effect of Biaxial Strain', fontsize=14)
    plt.tight_layout()
    plt.savefig(FIGS_DIR / "materials_strain_variation.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Show temperature broadening
    fig, ax = plt.subplots(figsize=(10, 5))
    
    temps = [100, 200, 300, 400]
    colors = plt.cm.coolwarm(np.linspace(0.2, 0.8, len(temps)))
    
    for temp, color in zip(temps, colors):
        params = np.array([1.8, 0.4, 0.0, temp, 0.02])
        spectrum = generate_materials_spectrum(params, energy)
        ax.plot(energy, spectrum, color=color, linewidth=2, label=f'T = {temp} K')
    
    ax.set_xlabel('Energy (eV)')
    ax.set_ylabel('Absorption')
    ax.set_title('Temperature-Dependent Spectral Broadening')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(FIGS_DIR / "materials_temperature_broadening.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved materials_strain_variation.png and materials_temperature_broadening.png")


# =============================================================================
# 4. Backend Comparison (PMM vs Regression)
# =============================================================================

def generate_simple_spectrum(alpha, beta, energy):
    """Simple 2-parameter spectrum for backend comparison."""
    center = 15 + 3 * alpha - 2 * beta
    width = 1.5 + 0.5 * beta
    strength = 1.0 + 0.3 * alpha
    return lorentzian_sum(energy, [center], [strength], [width])


def build_comparison_dataset(n_samples=40, seed=42):
    rng = np.random.default_rng(seed)
    energy = np.linspace(0, 35, 150)
    samples = []
    for i in range(n_samples):
        alpha = rng.uniform(0, 1)
        beta = rng.uniform(0, 1)
        spectrum = generate_simple_spectrum(alpha, beta, energy)
        samples.append(StrengthSample(
            params=np.array([alpha, beta]),
            energy=energy,
            strength=spectrum,
            label=f"sample_{i}"
        ))
    return StrengthDataset(samples)


def plot_backend_comparison():
    """Generate plots comparing PMM and regression backends."""
    print("Generating backend comparison plots...")
    
    dataset = build_comparison_dataset(n_samples=40)
    energy = np.linspace(0, 35, 150)
    
    # Train both backends
    reg_emu = get_emulator("regression", n_components=3)
    reg_emu.fit(dataset)
    
    pmm_emu = ParametricMatrixModel(n_poles=5, max_iterations=500, verbose=False)
    pmm_emu.fit(dataset, reference_point=np.array([0.5, 0.5]))
    
    # Interpolation test
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    
    rng = np.random.default_rng(123)
    
    # Row 1: Interpolation
    interp_points = [[0.3, 0.3], [0.5, 0.7], [0.7, 0.4]]
    reg_interp_errors = []
    pmm_interp_errors = []
    
    for ax, (a, b) in zip(axes[0], interp_points):
        params = np.array([a, b])
        true = generate_simple_spectrum(a, b, energy)
        
        reg_pred = reg_emu.predict_mixture(params)
        reg_spectrum = lorentzian_sum(energy, reg_pred.energies, reg_pred.strengths, reg_pred.widths)
        reg_err = normalized_l2(true, reg_spectrum, energy)
        reg_interp_errors.append(reg_err)
        
        pmm_result = pmm_emu.predict(params, energy)
        pmm_err = normalized_l2(true, pmm_result.spectrum, energy)
        pmm_interp_errors.append(pmm_err)
        
        ax.plot(energy, true, 'k-', linewidth=2.5, label='True')
        ax.plot(energy, reg_spectrum, 'b--', linewidth=2, label=f'Regression ({reg_err:.1%})')
        ax.plot(energy, pmm_result.spectrum, 'r:', linewidth=2, label=f'PMM ({pmm_err:.1%})')
        ax.set_xlabel('Energy')
        ax.set_ylabel('S(E)')
        ax.set_title(f'Interpolation: α={a:.1f}, β={b:.1f}')
        ax.legend(fontsize=9)
    
    # Row 2: Extrapolation
    extrap_points = [[1.3, 0.5], [0.5, 1.4], [1.2, 1.3]]
    reg_extrap_errors = []
    pmm_extrap_errors = []
    
    for ax, (a, b) in zip(axes[1], extrap_points):
        params = np.array([a, b])
        true = generate_simple_spectrum(a, b, energy)
        
        reg_pred = reg_emu.predict_mixture(params)
        reg_spectrum = lorentzian_sum(energy, reg_pred.energies, reg_pred.strengths, reg_pred.widths)
        reg_err = normalized_l2(true, reg_spectrum, energy)
        reg_extrap_errors.append(reg_err)
        
        pmm_result = pmm_emu.predict(params, energy)
        pmm_err = normalized_l2(true, pmm_result.spectrum, energy)
        pmm_extrap_errors.append(pmm_err)
        
        ax.plot(energy, true, 'k-', linewidth=2.5, label='True')
        ax.plot(energy, reg_spectrum, 'b--', linewidth=2, label=f'Regression ({reg_err:.1%})')
        ax.plot(energy, pmm_result.spectrum, 'r:', linewidth=2, label=f'PMM ({pmm_err:.1%})')
        ax.set_xlabel('Energy')
        ax.set_ylabel('S(E)')
        ax.set_title(f'Extrapolation: α={a:.1f}, β={b:.1f}')
        ax.legend(fontsize=9)
    
    axes[0, 0].set_ylabel('S(E)\n(Interpolation)', fontsize=12)
    axes[1, 0].set_ylabel('S(E)\n(Extrapolation)', fontsize=12)
    
    plt.suptitle('Backend Comparison: Regression vs PMM', fontsize=14)
    plt.tight_layout()
    plt.savefig(FIGS_DIR / "backend_comparison_spectra.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Summary bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    
    x = np.arange(2)
    width = 0.35
    
    reg_means = [np.mean(reg_interp_errors)*100, np.mean(reg_extrap_errors)*100]
    pmm_means = [np.mean(pmm_interp_errors)*100, np.mean(pmm_extrap_errors)*100]
    
    bars1 = ax.bar(x - width/2, reg_means, width, label='Regression', color='steelblue')
    bars2 = ax.bar(x + width/2, pmm_means, width, label='PMM', color='coral')
    
    ax.set_ylabel('Mean L² Error (%)')
    ax.set_title('Interpolation vs Extrapolation Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(['Interpolation\n(within training)', 'Extrapolation\n(outside training)'])
    ax.legend()
    
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5, f'{height:.1f}%',
                ha='center', va='bottom', fontsize=10)
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5, f'{height:.1f}%',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(FIGS_DIR / "backend_comparison_summary.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved backend_comparison_spectra.png and backend_comparison_summary.png")


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("Generating documentation example figures")
    print("=" * 60)
    print(f"Output directory: {FIGS_DIR}")
    print()
    
    plot_nuclear_example()
    plot_acoustic_example()
    plot_materials_example()
    plot_backend_comparison()
    
    print()
    print("=" * 60)
    print("All figures generated successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
