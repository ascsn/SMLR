# Theory and Mathematical Background

This document explains the mathematical foundations and physics motivation behind SMLR.

## Table of Contents

- [Linear Response Theory](#linear-response-theory)
- [Strength Functions](#strength-functions)
- [Lorentzian Representation](#lorentzian-representation)
- [Surrogate Modeling Approach](#surrogate-modeling-approach)
- [References](#references)

---

## Linear Response Theory

### Overview

Linear response theory describes how a quantum system responds to an external perturbation. For a system with Hamiltonian $H_0$ perturbed by an operator $\hat{O}$:

$$H = H_0 + \lambda \hat{O}(t)$$

The response function $R(E)$ characterizes the strength of transitions at energy $E$.

### Applications in Nuclear Physics

**Beta Decay**: Describes weak-force transitions between neutrons and protons
- **Gamow-Teller (GT)**: Spin-flip transitions ($\Delta L = 0, \Delta S = 1$)
- **Fermi**: Non-spin-flip transitions ($\Delta L = 0, \Delta S = 0$)

**Dipole Response**: Describes electromagnetic excitations
- **E1 transitions**: Electric dipole ($\Delta L = 1$, parity change)
- **M1 transitions**: Magnetic dipole ($\Delta L = 1$, no parity change)

**Giant Resonances**: Collective nuclear excitations involving coherent motion of many nucleons

---

## Strength Functions

### Definition

The strength function $S(E)$ quantifies the transition probability as a function of excitation energy:

$$S(E) = \sum_n |\langle n | \hat{O} | 0 \rangle|^2 \delta(E - E_n)$$

where $|0\rangle$ is the ground state and $|n\rangle$ are excited states at energies $E_n$.

### Sum Rules

Strength functions satisfy energy-weighted sum rules that provide model-independent constraints:

$$m_k = \int_{-\infty}^{\infty} E^k S(E) \, dE$$

For example, the $m_1$ sum rule (Thomas-Reiche-Kuhn) for dipole strength:

$$m_1 = \frac{NZ}{A} \cdot 60 \cdot A^{-1/3} \text{ (MeV·fm}^2\text{)}$$

### Practical Calculations

Ab initio calculations (QRPA, shell model) discretize $S(E)$ on an energy grid:

$$S(E_i) \approx \sum_{n \in \text{window}} |\langle n | \hat{O} | 0 \rangle|^2$$

Computing a single spectrum can take **hours to days** depending on the nucleus and model space.

---

## Lorentzian Representation

### Physical Motivation

Individual nuclear resonances have Lorentzian (Breit-Wigner) lineshapes due to finite lifetimes:

$$L(E; E_0, \Gamma, A) = A \cdot \frac{\Gamma/(2\pi)}{(E - E_0)^2 + \Gamma^2/4}$$

where:
- $E_0$ = resonance energy (peak position)
- $\Gamma$ = width (related to lifetime by $\tau = \hbar/\Gamma$)
- $A$ = strength (integrated area)

### Mixture Model

SMLR represents strength functions as sums of Lorentzians:

$$S(E) \approx \sum_{i=1}^{K} A_i \cdot \frac{\Gamma_i/(2\pi)}{(E - E_i)^2 + \Gamma_i^2/4}$$

**Advantages:**
1. **Dimensionality reduction**: $3K$ parameters vs. 100s of energy grid points
2. **Physical interpretation**: Each component represents a resonance
3. **Smooth interpolation**: Natural for parameter-space emulation
4. **Uncertainty quantification**: Width encodes resonance uncertainty

### Fitting Strategy

SMLR uses nonlinear least-squares optimization with soft constraints:

**Objective Function:**
$$\min_{\theta} \sum_j \left( S(E_j) - \sum_{i=1}^K L(E_j; \theta_i) \right)^2 + \lambda \|\theta\|^2$$

**Constraints:**
- Ordered resonance energies: $E_1 < E_2 < \ldots < E_K$
- Positive strengths: $A_i > 0$
- Positive widths: $\Gamma_i > 0$

**Implementation:**
- Parameterization using inverse softplus: $x = \log(\exp(z) - 1)^{-1}$ ensures positivity
- Sequential energy constraints with minimum spacing
- L2 regularization prevents overfitting

---

## Surrogate Modeling Approach

SMLR offers **two distinct emulation backends**, each with different trade-offs:

1. **Regression-based** (`StrengthEmulator`): Fast, simple, good for smooth dependence
2. **Parametric Matrix Model** (`ParametricMatrixModel`): Physics-based, better extrapolation

### Problem Statement

Given strength function evaluations at parameter points $\{\mathbf{p}_1, \ldots, \mathbf{p}_N\}$:

$$S(\mathbf{p}_i, E) = \text{expensive ab initio calculation}$$

**Goal**: Learn a fast approximation $\tilde{S}(\mathbf{p}, E)$ that predicts spectra at new $\mathbf{p}$ in milliseconds.

---

## Backend 1: Regression-Based Emulation

### Two-Stage Approach

**Stage 1: Compression**

For each training spectrum, fit a Lorentzian mixture:

$$S(\mathbf{p}_i, E) \approx \sum_{k=1}^K L(E; E_k^{(i)}, \Gamma_k^{(i)}, A_k^{(i)})$$

This gives mixture parameters $\Theta_i = \{E_k^{(i)}, A_k^{(i)}, \Gamma_k^{(i)}\}_{k=1}^K$.

**Stage 2: Emulation**

Learn mappings from parameters $\mathbf{p}$ to mixture parameters:

$$E_k = f_E^{(k)}(\mathbf{p}), \quad A_k = f_A^{(k)}(\mathbf{p}), \quad \Gamma_k = f_\Gamma^{(k)}(\mathbf{p})$$

SMLR uses **linear regression** with feature scaling:

$$f(\mathbf{p}) = \mathbf{W} \cdot \text{scale}(\mathbf{p}) + \mathbf{b}$$

### Prediction Workflow

Given new parameter vector $\mathbf{p}^*$:

1. **Predict mixture**: $\hat{\Theta} = \{f_E^{(k)}(\mathbf{p}^*), f_A^{(k)}(\mathbf{p}^*), f_\Gamma^{(k)}(\mathbf{p}^*)\}$
2. **Reconstruct spectrum**: $\tilde{S}(\mathbf{p}^*, E) = \sum_k L(E; \hat{\Theta}_k)$
3. **Evaluate**: Compute on arbitrary energy grid in $O(K \cdot N_E)$ time

### Width Strategies

**Global Width** (`width_mode="global"`):
$$\Gamma = f_\Gamma(\mathbf{p}) \quad \text{(shared across all resonances)}$$

Use when resonances have similar widths (e.g., thermal broadening dominates).

**Per-Component Width** (`width_mode="per_component"`):
$$\Gamma_k = f_\Gamma^{(k)}(\mathbf{p}) \quad \text{(each resonance has independent width)}$$

Use for multi-scale physics (e.g., narrow valence states + broad giant resonances).

### Normalization

Optionally normalize strengths to unit integral:

$$\tilde{S}(E) \leftarrow \frac{S(E)}{\int S(E') \, dE'}$$

**Benefits:**
- Removes arbitrary scale factors
- Improves regression conditioning
- Focuses emulator on spectral shape

**When to use:**
- Strengths vary by orders of magnitude
- Only relative distributions matter
- Absolute normalization can be restored post-prediction

---

## Backend 2: Parametric Matrix Model (PMM)

### Motivation

The regression approach treats pole parameters as independent variables learned
via regression. This ignores the underlying physics: strength functions arise
from the eigenvalue structure of a response matrix (Hamiltonian).

The **Parametric Matrix Model** directly learns this matrix structure, leading to:

- Better sum rule preservation
- More physically motivated extrapolation
- Interpretable learned parameters

### Mathematical Foundation

In linear response theory, the strength function can be written as:

$$S(E; \mathbf{p}) = \sum_n |\langle n(\mathbf{p}) | \hat{O} | 0 \rangle|^2 \cdot L(E; E_n(\mathbf{p}), \Gamma)$$

where $|n(\mathbf{p})\rangle$ and $E_n(\mathbf{p})$ are eigenstates and eigenvalues of a
parameter-dependent Hamiltonian or response matrix $M(\mathbf{p})$.

### The PMM Ansatz

We parameterize the response matrix as a **linear function of parameters**:

$$M(\mathbf{p}) = D + \sum_{i=1}^{d} (p_i - p_i^{(0)}) \cdot S_i$$

where:

- $D$ is a diagonal matrix containing base eigenvalues
- $S_i$ are symmetric perturbation matrices (one per parameter)
- $\mathbf{p}^{(0)}$ is a reference point (typically the center of training data)
- $d$ is the parameter dimension

This is a **reduced-order model**: instead of working with the full many-body
Hamiltonian, we learn an effective $n \times n$ matrix that reproduces the
relevant spectral features.

### Strength Function Reconstruction

Given parameters $\mathbf{p}$:

1. **Build matrix**: $M(\mathbf{p}) = D + \sum_i (p_i - p_i^{(0)}) S_i$
2. **Diagonalize**: $(E_n, |\psi_n\rangle) = \text{eigh}(M)$
3. **Compute strengths**: $B_n = |\langle \psi_n | \mathbf{v}_0 \rangle|^2$
4. **Reconstruct**: $S(E) = \sum_n B_n \cdot L(E; E_n, \Gamma)$

where $\mathbf{v}_0$ is an "external field" vector that determines transition amplitudes.

### Training

PMM training optimizes the learned parameters $(D, \{S_i\}, \mathbf{v}_0, \Gamma)$
to minimize the reconstruction error over training spectra:

$$\mathcal{L} = \sum_{j=1}^{N} \frac{\|\tilde{S}(\mathbf{p}_j, E) - S_j(E)\|_2^2}{\|S_j(E)\|_2^2} + \lambda \|\theta\|_2^2$$

where $\theta$ represents all learnable parameters and $\lambda$ is regularization.

### Advantages of PMM

**Sum Rule Preservation**:

Energy-weighted sum rules are automatically preserved:

$$m_k = \int E^k S(E) dE = \sum_n B_n E_n^k$$

The total strength $m_0 = \sum_n B_n = \|\mathbf{v}_0\|^2$ is constant.

**Extrapolation**:

Because the parameter dependence is encoded in the matrix structure,
extrapolation follows the physics rather than naive polynomial extension.

**Interpretability**:

- $D$ gives the "unperturbed" spectrum at the reference point
- $S_i$ shows how each parameter affects eigenvalue structure
- $\mathbf{v}_0$ encodes the transition operator

### Parametric Width

Optionally, the width can also depend on parameters:

$$\Gamma(\mathbf{p}) = \eta_0 + \sum_i \eta_i p_i$$

This captures broadening mechanisms that depend on physics parameters
(e.g., temperature, coupling strength).

### When to Use PMM

| Use PMM when... | Use Regression when... |
|-----------------|------------------------|
| Sum rules must be preserved | Fast training needed |
| Extrapolation is important | Many training samples available |
| Fewer training samples | Smooth parameter dependence |
| Physics interpretability matters | High-dimensional parameters (10+) |
| Response matrix structure known | Uncertainty quantification (GP) |

### Implementation

```python
from smlr.backends import get_emulator
import numpy as np

# Create PMM emulator
pmm = get_emulator("pmm", n_poles=10)

# Fit with reference point
pmm.fit(dataset, reference_point=np.array([0.5, 0.5]))

# Predict
result = pmm.predict(new_params, energy_grid)
print(f"Eigenvalues: {result.eigenvalues}")
print(f"Strengths: {result.strengths}")
```

---

## Error Quantification

### Normalized L² Metric

$$\epsilon_{\text{L}^2} = \frac{\| S_{\text{pred}} - S_{\text{true}} \|_{\text{L}^2}}{\| S_{\text{true}} \|_{\text{L}^2}}$$

where:

$$\| f \|_{\text{L}^2}^2 = \int f(E)^2 \, dE \approx \sum_j f(E_j)^2 \Delta E_j$$

**Interpretation:**
- $\epsilon < 0.01$: Excellent (< 1% error)
- $0.01 < \epsilon < 0.1$: Good (1-10% error)
- $0.1 < \epsilon < 0.5$: Acceptable for many applications
- $\epsilon > 0.5$: Poor, more training data or components needed

### Validation Strategy

1. **Train/Test Split**: Hold out 10-30% of data
2. **Cross-Validation**: K-fold CV for small datasets
3. **Extrapolation Tests**: Evaluate on parameter region boundaries
4. **Convergence Studies**: Increase $K$ (components) until error plateaus

---

## Computational Complexity

| **Operation** | **QRPA/Shell Model** | **SMLR Emulator** | **Speedup** |
|--------------|---------------------|-------------------|-------------|
| Single spectrum | $O(10^3 - 10^6)$ s | $O(10^{-3})$ s | $10^6 - 10^9 \times$ |
| Training cost | N/A | $O(N \cdot T_{\text{fit}})$ | One-time |
| Parameter scan (1000 points) | ~months | ~seconds | $10^6 \times$ |

**Key Insight**: Training overhead is amortized over thousands of predictions, making emulation essential for:
- Bayesian parameter estimation
- Uncertainty quantification
- Real-time optimization
- Interactive exploration

---

## Extensions and Variants

### Gaussian Process Emulation

Replace linear regression with GP for:
- Nonlinear parameter dependence
- Automatic uncertainty estimates
- Small data regime ($N < 50$)

**Trade-offs:**
- Higher computational cost ($O(N^3)$)
- Better uncertainty quantification
- Requires careful kernel selection

### Neural Network Emulation

Use MLPs or CNNs for:
- Complex, high-dimensional parameter spaces
- Large datasets ($N > 1000$)
- End-to-end spectrum prediction (skip Lorentzian stage)

**Trade-offs:**
- Requires more training data
- Less interpretable
- GPU acceleration beneficial

### Adaptive Component Selection

Automatically determine $K$ using:
- Bayesian information criterion (BIC)
- Cross-validation error
- Residual analysis

**Implementation idea:**
```python
for K in range(1, K_max):
    emu = StrengthEmulator(n_components=K)
    error = cross_validate(emu, dataset)
    if error < threshold:
        break
```

---

## References

### Nuclear Physics Background

1. **QRPA Theory**: Rowe, D. J. (1970). *Nuclear Collective Motion*
2. **Sum Rules**: Blaizot, J. P., & Ripka, G. (1986). *Quantum Theory of Finite Systems*
3. **Beta Decay**: Behrens, H., & Bühring, W. (1982). *Electron Radial Wave Functions and Nuclear Beta-decay*

### Surrogate Modeling

4. **Gaussian Processes**: Rasmussen & Williams (2006). *Gaussian Processes for Machine Learning*
5. **Lorentzian Fitting**: Bevington & Robinson (2003). *Data Reduction and Error Analysis*
6. **Emulation in Physics**: SURMISE package (2020). Argonne National Laboratory

### Related Work

7. Machine learning for nuclear physics: Niu et al., *Phys. Rev. C* (2019)
8. Bayesian model calibration: Kennedy & O'Hagan, *J. R. Stat. Soc. B* (2001)

---

## Summary

SMLR combines:
- **Physics**: Lorentzian lineshapes from resonance theory
- **Statistics**: Linear regression with feature scaling
- **Numerics**: Robust optimization with constraints

The result is a practical tool that preserves physical interpretability while achieving dramatic computational speedups.

**Next**: See the [Usage Guide](usage.md) for practical applications, or the [API Reference](api.md) for implementation details.
