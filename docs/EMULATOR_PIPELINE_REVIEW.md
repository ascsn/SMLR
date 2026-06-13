# Emulator Pipeline Review Checklist

This is a working checklist for cleaning the legacy beta-decay and dipole
polarizability workflows into a reproducible baseline. Check items off as each
file or task is kept, discarded, reorganized, or reimplemented.

## Canonical Data And Results

- [x] Keep `data/beta_decay_80Ni/` as canonical beta-decay paper data.
- [x] Confirm beta strength files are currently under `data/beta_decay_80Ni/total_strength/` with `lorm_*.out` names.
- [ ] Decide whether the beta strength directory should be named `total_strength/` or `total_lorm/` permanently.
- [x] Confirm `data/beta_decay_80Ni/total_excm/` is the discrete excitation-strength input for half-life calculations.
- [x] Confirm `data/beta_decay_80Ni/total_half_life/` is the scalar half-life target set.
- [x] Keep `data/dipole_polarizability_160Yb/` as canonical dipole paper data.
- [x] Confirm `data/dipole_polarizability_160Yb/total_strength/` is the dipole strength-function input.
- [x] Confirm `data/dipole_polarizability_160Yb/total_alphaD/` is the scalar polarizability target set.
- [ ] Review `results/data_beta/` and choose gold reference parameter/result files for tests.
- [ ] Review `results/data_dipole/` and choose gold reference parameter/result files for tests.

## Core Code To Keep And Harden

- [ ] Implement reusable data discovery/loading for strength grids and scalar observables.
- [ ] Keep and review `src/smlr/core/ansatz.py`.
  Owns packed trainable parameters, feature construction, matrix/vector construction, and random initialization.
- [ ] Make parameter packing/unpacking a tested reusable core capability.
- [ ] Check `src/smlr/core/ansatz.py` against both legacy beta and dipole parameter layouts.
- [ ] Keep and review `src/smlr/core/numerics.py`.
  Owns Lorentzian evaluation, batched Lorentzian evaluation, and centered spectrum initialization.
- [ ] Keep Lorentzian evaluation in reusable core code.
- [ ] Implement eigensolver-based prediction as a reusable core path.
- [ ] Keep and review `src/smlr/core/fitting.py`.
  Owns central-spectrum Lorentzian fitting used to initialize emulator poles and strengths.
- [ ] Keep and review `src/smlr/core/retention.py`.
  Owns retained-mode policy.
- [ ] Keep and extend `src/smlr/core/objectives.py`.
  Baseline for shared strength-plus-observable loss composition.
- [ ] Keep loss-function composition in reusable core code.
- [ ] Keep generic artifact saving in shared training or utility code, not in paper notebooks.
- [ ] Add paper-scaling options to `src/smlr/core/objectives.py` if exact legacy reproduction requires fixed denominators.
- [ ] Keep `src/smlr/core/splitting.py` if training/test split logic remains code-driven.
- [ ] Move split definitions into explicit config files if splits should be reproducibility artifacts.
- [ ] Keep `src/smlr/core/training.py` for seed setting, moving averages, and TensorFlow Adam construction.
- [ ] Implement training/restart driver as shared code.
- [ ] Decide whether restart bookkeeping and early stopping should move into `src/smlr/core/training.py`.

## Metrics And Validation

- [ ] Keep `src/smlr/metrics.py` for now.
  It contains generic post-training evaluation helpers: absolute/relative errors, observable summaries, integrated strength errors, weighted spectral loss, and RMSE.
- [ ] Decide later whether `src/smlr/metrics.py` should move to `scripts/diagnostics/metrics.py`.
  It is diagnostic/evaluation code, not core emulator math.
- [x] Move package-style `src/smlr/validation.py` to `scripts/data_validation/validation.py`.
- [ ] Use `scripts/data_validation/validation.py beta-paper` as a reproducibility check for beta data.
- [ ] Use `scripts/data_validation/validation.py dipole-paper` as a reproducibility check for dipole data.
- [ ] Decide whether validation should stay script-only or eventually become reusable library code again.

## Domain Code To Review

- [ ] Keep and simplify `src/smlr/domains/dipole.py`.
  It should own alphaD calculation, dipole constants, dipole file naming, and dipole-specific initialization choices.
- [ ] Keep observable calculation formulas in domain code when they are physics-specific.
- [ ] Remove package-adapter language from `src/smlr/domains/dipole.py` once the native trainer exists.
- [ ] Keep and simplify `src/smlr/domains/beta_decay.py`.
  It should own beta-decay constants, phase-space/half-life calculation, beta file naming, and paper-specific EM1/EM2 unpacking.
- [ ] Remove package-adapter language from `src/smlr/domains/beta_decay.py` once the native trainer exists.
- [ ] Keep `src/smlr/domains/common.py` only if `MatrixAnsatzConfig` remains shared.
- [ ] Consider moving `MatrixAnsatzConfig` from `domains/common.py` into `src/smlr/core/ansatz.py`.

## Training Code To Review

- [ ] Review `src/smlr/training/dipole_em1.py`.
  It is currently a compatibility entry point into legacy dipole training.
- [ ] Replace `src/smlr/training/dipole_em1.py` with a native trainer.
- [ ] Review `src/smlr/training/beta_em1.py`.
  It is currently a compatibility entry point into legacy beta EM1 training.
- [ ] Replace `src/smlr/training/beta_em1.py` with a native trainer.
- [ ] Review `src/smlr/training/beta_em2.py`.
  It is currently a compatibility entry point into legacy beta EM2 training.
- [ ] Replace `src/smlr/training/beta_em2.py` with a native trainer or discard if EM2 is not in scope.
- [ ] Review `src/smlr/training/strength_only.py`.
  It now stages a simple plan without depending on deleted package specs.
- [ ] Decide whether `strength_only.py` should become the generic trainer or be removed until needed.

## Package-Style Infrastructure Removed From Baseline

- [x] Remove `src/smlr/cli.py`.
- [x] Remove `src/smlr/specs.py`.
- [x] Remove `src/smlr/serialization.py`.
- [x] Remove the `smlr` console script entry from `pyproject.toml`.
- [x] Remove root `uv.lock` from the baseline if present.
- [ ] Decide later whether config dataclasses should return as a lightweight `configs/` format after paper configs are known.
- [ ] Decide later whether emulator serialization should return after the native trainers are stable.

## Config To Externalize

- [ ] Externalize data paths for each emulator run.
- [ ] Externalize parameter-grid filters and train/test split rules.
- [ ] Externalize central-point selection or explicitly pin the central point.
- [ ] Externalize physics constants used by domain formulas.
- [ ] Externalize loss weights for strength and scalar observables.
- [ ] Externalize minimum iterations and early-stopping policy.
- [ ] Externalize optimizer settings, including optimizer name and learning rate.
- [ ] Externalize output filenames and output directory conventions.
- [ ] Keep these configs outside source code so paper reproduction runs are auditable.

## Backends

- [x] Remove `src/smlr/backends/jax.py`.
- [x] Remove `src/smlr/backends/pytorch.py`.
- [x] Remove `jax` and `torch` optional dependency extras from `pyproject.toml`.
- [x] Keep only TensorFlow Adam optimizer support in `src/smlr/backends/optimizers.py`.
- [ ] Decide whether `src/smlr/backends/tensorflow.py` is still useful or whether TensorFlow calls should stay direct.

## Legacy Beta-Decay Files

- [ ] Review `Beta_decay/main.py` as original EM1 source of truth.
- [ ] Extract data loading from `Beta_decay/main.py`.
- [ ] Extract cost terms from `Beta_decay/main.py`.
- [ ] Extract restart/training-loop behavior from `Beta_decay/main.py`.
- [ ] Review `Beta_decay/main_only_HL.py` as original EM2/half-life-only source of truth.
- [ ] Review `Beta_decay/helper.py` carefully.
- [ ] Extract phase-space functions from `Beta_decay/helper.py`.
- [ ] Extract half-life formula from `Beta_decay/helper.py`.
- [ ] Extract beta Lorentzian/file-reader helpers from `Beta_decay/helper.py`.
- [ ] Reorganize `Beta_decay/check_results.py` into `scripts/diagnostics/` or discard.
- [ ] Reorganize `Beta_decay/check_results_only_HL.py` into `scripts/diagnostics/` or discard.
- [ ] Reorganize `Beta_decay/animate_strength_evolution.py` into `scripts/diagnostics/` or notebooks if still needed.
- [ ] Convert `Beta_decay/test_emulator_strength.py` into a real regression test or discard.
- [ ] Convert `Beta_decay/test_strength_outside_train_region.py` into diagnostics or discard.

## Legacy Beta Package Files

- [ ] Review `Beta_decay_package/src/main_gpt2.py` as current compatibility target for beta EM1.
- [ ] Use `Beta_decay_package/src/main_gpt2.py` as a migration source, then discard.
- [ ] Review `Beta_decay_package/src/main_only_HL_gpt.py` as current compatibility target for beta EM2.
- [ ] Use `Beta_decay_package/src/main_only_HL_gpt.py` as a migration source, then discard.
- [ ] Review `Beta_decay_package/src/helper_gpt.py` as the main beta migration source.
- [ ] Extract only still-needed physics/data/cost pieces from `Beta_decay_package/src/helper_gpt.py`.
- [ ] Reorganize `Beta_decay_package/src/diagnostics_general_gpt.py` into `scripts/diagnostics/` if paper diagnostics still depend on it.
- [ ] Reorganize `Beta_decay_package/scripts/run_beta_hyperparameter_sweep.py` into `scripts/sweeps/`.
- [ ] Reorganize `Beta_decay_package/scripts/check_results*.py` into `scripts/diagnostics/`.

## Legacy Dipole Files

- [ ] Review `Dipole_polarizability/src/main_gpt2.py` as current compatibility target for dipole EM1.
- [ ] Use `Dipole_polarizability/src/main_gpt2.py` as a migration source, then discard.
- [ ] Review `Dipole_polarizability/src/main_only_alphaD_gpt.py` for alphaD-only EM2.
- [ ] Decide whether dipole alphaD-only EM2 belongs in baseline reproduction.
- [ ] Review `Dipole_polarizability/src/helper_gpt.py` as the main dipole migration source.
- [ ] Extract dipole data loading from `Dipole_polarizability/src/helper_gpt.py`.
- [ ] Extract alphaD calculation from `Dipole_polarizability/src/helper_gpt.py`.
- [ ] Extract dipole cost terms from `Dipole_polarizability/src/helper_gpt.py`.
- [ ] Extract dipole ansatz/initialization behavior from `Dipole_polarizability/src/helper_gpt.py`.
- [ ] Reorganize `Dipole_polarizability/src/diagnostics_general_gpt.py` into `scripts/diagnostics/`.
- [ ] Reorganize or discard `Dipole_polarizability/src/diagnostics_general_gpt_styled.py`.
- [ ] Reorganize `Dipole_polarizability/src/diagnostics_em2_gpt.py` only if alphaD-only EM2 remains in scope.
- [ ] Compare `Dipole_polarizability/src/main_gpt.py` with `main_gpt2.py`, then discard if superseded.
- [ ] Move or discard `Dipole_polarizability/src/YbPerformance_revised.ipynb`.
- [ ] Sort `Dipole_polarizability/scripts/*.py` into `scripts/diagnostics/`, `scripts/observables/`, or discard.
- [ ] Archive or discard `Dipole_polarizability/scrap/` after checking for unique animation/diagnostic logic.

## Paper Reproduction And Notebook Files

- [ ] Keep `paper_reproduction/dipole_em1_paper_reproduction.ipynb` as a reproduction target.
- [ ] Replace hidden assumptions in `paper_reproduction/dipole_em1_paper_reproduction.ipynb` with explicit config.
- [ ] Keep `paper_reproduction/beta_em2_paper_reproduction.ipynb` if beta EM2 remains in paper scope.
- [ ] Review `paper_reproduction/specs/bd_n13_paper.json` as an example config.
- [ ] Normalize `paper_reproduction/specs/bd_n13_paper.json` with future paper result configs.
- [ ] Review which `notebooks/*.ipynb` files produce final paper figures.
- [ ] Move exploratory notebooks into `notebooks/exploratory/` or discard.
- [ ] Keep `interactive/*.html` only if they are docs or paper supplements.

## New Implementations Needed

- [ ] Implement native dipole EM1 trainer in `src/smlr/training/`.
- [ ] Implement native beta EM1 trainer in `src/smlr/training/`.
- [ ] Implement or discard native beta EM2 trainer depending on paper reproduction scope.
- [ ] Implement or discard dipole alphaD-only trainer depending on paper reproduction scope.
- [ ] Implement shared data loaders for strength grids and scalar observables.
- [ ] Extend shared objective builder for strength-only loss.
- [ ] Extend shared objective builder for strength plus alphaD loss.
- [ ] Extend shared objective builder for strength plus half-life loss.
- [ ] Extend shared objective builder for observable-only loss.
- [ ] Add paper-specific fixed scaling constants when exact reproduction needs them.
- [ ] Implement shared restart/early-stopping runner.
- [ ] Create reproducibility configs for each paper result.
- [ ] Include data paths in each reproducibility config.
- [ ] Include parameter grid/filter in each reproducibility config.
- [ ] Include model dimension and retain fraction in each reproducibility config.
- [ ] Include random seeds/restarts in each reproducibility config.
- [ ] Include optimizer settings in each reproducibility config.
- [ ] Include output directory in each reproducibility config.
- [ ] Include expected reference parameter/result file in each reproducibility config.
- [ ] Add regression tests comparing cleaned-code outputs to selected files in `results/`.

## Reproducibility Additions

- [ ] Create one shared config file per emulator.
- [ ] Pin the environment used for paper reproduction.
- [ ] Make seed handling deterministic across Python, NumPy, and TensorFlow.
- [ ] Save a manifest with every run.
- [ ] Include git commit hash in the run manifest.
- [ ] Include resolved config contents in the run manifest.
- [ ] Include resolved data paths in the run manifest.
- [ ] Include output filenames in the run manifest.
- [ ] Add tests for parameter-vector packing/unpacking.
- [ ] Add one tiny synthetic training-pass test.

## Suggested Migration Order

- [ ] Freeze one known-good paper config for dipole EM1.
- [ ] Freeze one known-good paper config for beta EM1 or beta EM2.
- [ ] Extract domain formulas from legacy helpers into `src/smlr/domains/`.
- [ ] Extract reusable training loop into `src/smlr/training/`.
- [ ] Replace compatibility entry points one by one.
- [ ] Convert diagnostics scripts to read cleaned output directories.
- [ ] Add regression checks against selected `results/` files.
- [ ] Remove or archive legacy folders only after the cleaned pipeline reproduces the chosen reference configs.
