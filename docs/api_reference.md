# API Reference

Complete auto-generated documentation for all public SMLR classes and functions.

## Core Classes

### Surrogate

The unified interface for all emulation backends.

::: smlr.base.Surrogate
    options:
      members:
        - __init__
        - fit
        - predict
        - backend_name

### EmulatorResult

Container for emulator predictions.

::: smlr.base.EmulatorResult

---

## Data Management

### StrengthSample

::: smlr.data.StrengthSample
    options:
      members:
        - normalized

### StrengthDataset

::: smlr.data.StrengthDataset
    options:
      members:
        - from_arrays
        - from_folder
        - parameters
        - energy_grids
        - strength_arrays
        - subset
        - normalized
        - __len__
        - __getitem__

---

## Lorentzian Fitting

### LorentzianMixture

::: smlr.lorentz.LorentzianMixture
    options:
      members:
        - __call__
        - n_components

### fit_lorentzian_mixture

::: smlr.lorentz.fit_lorentzian_mixture

---

## Emulator Backends

### StrengthEmulator

Regression-based emulator backend.

::: smlr.emulator.StrengthEmulator
    options:
      members:
        - __init__
        - fit
        - predict

### ParametricMatrixModel

Physics-based PMM backend.

::: smlr.pmm.ParametricMatrixModel
    options:
      members:
        - __init__
        - fit
        - predict

### Backend Factory

::: smlr.backends.get_emulator

---

## Metrics

::: smlr.metrics.normalized_l2

::: smlr.metrics.mean_absolute_relative_error

---

## Plotting

::: smlr.plotting.plot_spectrum

::: smlr.plotting.plot_comparison

::: smlr.plotting.plot_mixture_components

---

## Observables

### Observable Classes

::: smlr.observables.Observable
    options:
      show_bases: false

::: smlr.observables.SumRule

::: smlr.observables.DipolePolarizability

::: smlr.observables.BetaDecayHalfLife

### Factory Functions

::: smlr.observables.create_trk_sum_rule

::: smlr.observables.create_polarizability
