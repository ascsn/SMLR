# H2 2D Strength-Function Emulator

This directory packages the H2 two-parameter strength-function emulation work into a self-contained snapshot for GitHub.

The package uses the temporary emulator copy, not the repository package source. The main package files under `Dipole_polarizability/src` are not required to run this snapshot.

## Contents

- `data/total_strength/`: 50 folded H2 strength spectra, one `.out` file per `(q, theta)` grid point.
- `emulator_src/`: copied emulator source used for this H2 run, including the strength-only and strength-normalization changes.
- `scripts/h2_10model_sweep.py`: hyperparameter sweep driver.
- `scripts/plot_best_h2_model.py`: summary plots for the sweep best model.
- `scripts/make_h2_original_style_diagnostics.py`: H2 strength-only analog of the original diagnostics.
- `results/sweep_runs/`: completed sweep outputs, fitted parameters, summaries, logs, and diagnostic plots.
- `logs/`: top-level diagnostic regeneration logs from the temporary run.

## Data Layout

The strength files are named

```text
strength_<q>_<theta>.out
```

where `q` is the probe spatial-scale parameter and `theta` is the probe direction in radians. Each file has two columns:

```text
excitation_energy_eV  strength
```

## Main Results

The original 10-model unnormalized sweep is summarized in:

```text
results/sweep_runs/sweep_summary.csv
```

The best model from that sweep was:

```text
m06_n16_ret1_fold1_linear_exp_affine_lr5e-3
strength_cost = 0.017020845785737038
```

After adding H2-appropriate median-L2 strength normalization in the temporary emulator copy, the same best hyperparameter setting improved to:

```text
normalized_median_l2_best_config
strength_scale = 3.4967087283078895
strength_cost = 0.005408762488514185
global_best_cost = 0.00540892081335187
```

The normalized physical-unit diagnostics are in:

```text
results/sweep_runs/best_plots/normalized_median_l2_strength_diagnostics_xlim_0_40/
```

Key files there:

- `strength_true_vs_prediction.png`
- `parameter_error_map.png`
- `parameter_relative_l2_error_map.png`
- `index_selection_map.png`
- `detail_spectrum_selected_grid.png`
- `per_sample_strength_metrics.csv`
- `diagnostic_summary.json`

## Rerun

From the repository root:

```bash
python H2_2d/scripts/h2_10model_sweep.py
```

This writes new results under:

```text
H2_2d/results/sweep_runs/
```

To regenerate plots for the current sweep best:

```bash
python H2_2d/scripts/plot_best_h2_model.py
```

To regenerate original-style H2 strength diagnostics:

```bash
python H2_2d/scripts/make_h2_original_style_diagnostics.py
```

For the normalized retrain diagnostics specifically:

```bash
H2_BEST_MODEL_JSON=H2_2d/results/sweep_runs/best_plots/normalized_median_l2_best_model.json \
H2_DIAG_OUTDIR=H2_2d/results/sweep_runs/best_plots/normalized_median_l2_strength_diagnostics_xlim_0_40 \
python H2_2d/scripts/make_h2_original_style_diagnostics.py
```

## Notes

The strength loss is relative-L2 per spectrum, so there was no fixed paper scalar directly entering the strength-only cost. The H2 spectra are much smaller in amplitude than the dipole-polarizability paper data, however, so the temporary copy adds `--strength-normalization` and records the resulting `strength_scale` in each normalized run summary.

The high-energy ghost-pole behavior remains a separate modeling issue from strength normalization. The normalized run improves the fit but still leaves one pole above the sampled spectrum interval, so future runs should use a pole-window penalty or a bounded-pole parameterization if that behavior needs to be suppressed.
