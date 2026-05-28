# SMLR Release Plan

This plan scopes the first solid release of SMLR around four goals:

- Reproduce the paper results.
- Support 2D spectral emulation workflows.
- Provide the basic diagnostic tools used in the paper.
- Ship accompanying documentation that lets a new user install, validate data, train, diagnose, and reproduce results.

## Release Target

Target version: `0.1.0`

Release status: alpha research package

Primary audience:

- Paper readers who want to reproduce the published figures and benchmark results.
- Users with general two-parameter linear-response strength data.
- Developers extending the current emulators to new observables or domains.

Supported observables for the first release:

- Dipole polarizability.
- Beta-decay half-life.

Other observables are out of scope for `0.1.0` unless they are needed directly for the paper-reproduction workflows.

Out of scope for the first release:

- Full general-purpose N-dimensional emulator support.
- Polished graphical interfaces.
- Large-scale data hosting inside the package wheel.
- Backend parity across TensorFlow, PyTorch, and JAX. The first release supports TensorFlow only.
- Stable public API guarantees beyond the documented first-release commands.

## 1. Define The Release Boundary

- [x] Decide which workflows are officially supported in `0.1.0`.
- [x] Mark beta-decay EM1 paper reproduction as supported.
- [x] Mark beta-decay EM2 half-life-only paper reproduction as supported.
- [x] Mark dipole EM1 paper reproduction as supported.
- [x] Mark generic 2D strength-only spectral emulation as supported.
- [x] Mark paper-style diagnostics as supported.
- [ ] Decide which old scripts should be left only in commit history.
- [ ] Decide whether any archival code needs to be moved to an `archive` branch before release.
- [x] Add a clear "supported vs archival" note to `README.md`.
- [x] Add the same boundary statement to the documentation home page.

## 2. Consolidate Package Entry Points

- [ ] Make `src/smlr/` the canonical implementation location.
- [ ] Ensure `smlr validate` covers all first-release data layouts.
- [ ] Ensure `smlr spec` prints all first-release built-in specs.
- [ ] Ensure `smlr train beta-paper-em1` reproduces the intended beta EM1 workflow.
- [ ] Ensure `smlr train beta-paper-em2` reproduces the intended beta EM2 workflow.
- [ ] Ensure `smlr train dipole-paper-em1` reproduces the intended dipole EM1 workflow.
- [ ] Ensure `smlr train strength-only` supports a documented 2D spectral-emulation example.
- [ ] Ensure `smlr diagnose beta-paper` runs the beta paper diagnostics.
- [ ] Ensure `smlr diagnose dipole-paper-em1` runs the dipole paper diagnostics.
- [ ] Ensure `smlr predict` works from serialized emulator records for the supported domains.
- [ ] Remove old package scripts from the supported path; leave them in commit history or move them to an `archive` branch.

## 3. Paper Reproduction

- [ ] Define the exact paper reproduction commands in `docs/Reproducibility.md`.
- [ ] Record expected input data locations for beta decay and dipole polarizability.
- [ ] Record expected output directories and filenames.
- [ ] Record expected reference metrics for each reproduced run.
- [ ] Add checks that trained parameter files are serialized with enough metadata to reproduce the run.
- [ ] Add checks that diagnostic summaries match expected reference bounds.
- [ ] Add a compact "fast smoke test" reproduction path with reduced iterations.
- [ ] Add a full reproduction path for paper-quality results.
- [ ] Make clear which generated figures correspond to which paper figures.
- [ ] Add a troubleshooting section for missing data, bad filenames, and failed TensorFlow imports.

## 4. 2D Spectral Emulation

- [x] Define the accepted generic 2D strength-file format.
- [x] Document filename regex requirements with named parameter groups.
- [x] Document required strength table columns.
- [x] Document assumptions about rectangular parameter grids.
- [x] Add or confirm validation for non-rectangular grids.
- [x] Add or confirm validation for malformed numeric files.
- [ ] Provide one small example dataset that can be shipped with the repo.
- [ ] Provide one example spec for the small 2D spectral-emulation dataset.
- [ ] Provide one training command for that example.
- [ ] Provide one prediction command for a new 2D parameter point.
- [ ] Provide one diagnostics command for the example.
- [ ] Explain how users adapt the example to their own two-parameter strength data.

## 5. Diagnostics

- [ ] Define the required diagnostic outputs for beta EM1.
- [ ] Define the required diagnostic outputs for beta EM2.
- [ ] Define the required diagnostic outputs for dipole EM1.
- [ ] Standardize diagnostic output filenames where possible.
- [ ] Ensure each diagnostic run writes a machine-readable CSV or JSON summary.
- [ ] Ensure each diagnostic run writes human-readable plots.
- [ ] Ensure diagnostics fail with useful messages when required params, specs, or data are missing.
- [ ] Add tests that diagnostic PNGs are created and nonempty.
- [ ] Add tests that diagnostic CSV schemas remain stable.
- [ ] Document how to interpret the main diagnostic plots.

## 6. Serialization And Reuse

- [ ] Ensure every supported training workflow writes an `emulator.json`.
- [ ] Ensure serialized records include model name, adapter, backend, package version, parameter file, run spec, and metadata.
- [ ] Ensure serialized records include retained-mode policy when relevant.
- [ ] Ensure `smlr serialize` can package existing paper parameter files.
- [ ] Ensure `smlr predict` can load all first-release serialized emulator records.
- [ ] Add examples showing prediction from an already trained emulator.
- [ ] Add tests for loading beta EM1, beta EM2, dipole EM1, and generic 2D spectral emulators.

## 7. Tests And Quality Gates

- [ ] Keep core numerical regression tests passing.
- [ ] Keep data-validation tests passing.
- [ ] Keep spec serialization tests passing.
- [ ] Keep paper-run reference tests passing.
- [x] Add beta EM1 short-training parity smoke test against the project legacy code.
- [ ] Add CLI smoke tests for each supported command group.
- [ ] Add tests for graceful missing-optional-dependency errors.
- [ ] Add tests for package import without TensorFlow installed, if TensorFlow remains optional.
- [ ] Add a build test that creates both wheel and source distribution.
- [ ] Add a clean-install test from the built wheel.
- [ ] Add a release checklist item to run `uv run python -m unittest`.
- [ ] Add a release checklist item to run `uv build`.

## 8. Dependency Management

- [x] Document that users should install and set up the package with `uv`.
- [x] Document that `uv` manages the project virtual environment and dependency resolution.
- [ ] Decide the minimal base dependency set.
- [ ] Confirm `numpy` is the only required base dependency, or move additional required imports into base dependencies.
- [ ] Confirm TensorFlow is installed by the documented first-release setup path.
- [x] Make clear that TensorFlow is the only supported backend for `0.1.0`.
- [ ] Confirm the `paper` extra installs everything needed for paper reproduction.
- [ ] Confirm the `examples` extra installs everything needed for notebooks and example plots.
- [ ] Confirm the `docs` extra builds the documentation.
- [x] Document `uv` install commands for base, paper, examples, and docs use cases.
- [ ] Add clear error messages when users call TensorFlow-backed trainers without TensorFlow installed.

## 9. Documentation

- [ ] Make `README.md` a short, accurate landing page.
- [x] Fix any broken README image or file references.
- [x] Add a first-install quick start.
- [x] Add a first validation command.
- [ ] Add a first training command.
- [ ] Add a first diagnostics command.
- [ ] Add a first prediction command.
- [ ] Expand `docs/Getting Started.md` for users who have never installed a Python package.
- [ ] Expand `docs/Reproducibility.md` with paper reproduction commands and expected outputs.
- [ ] Expand `docs/Development.md` with package structure, tests, build, and release steps.
- [ ] Keep extra documentation features out of `0.1.0` unless they are required for install, setup, paper reproduction, diagnostics, or the documented 2D spectral-emulation workflow.

## To Be Released

These are useful additions, but they are not required for the `0.1.0` release.

- [ ] Expanded `docs/Examples.md` with additional worked examples.
- [ ] API reference pages for public modules and commands.
- [ ] Glossary for EM1, EM2, strength function, half-life, retained modes, and central point.
- [ ] Additional observables beyond dipole polarizability and beta-decay half-life.
- [ ] PyTorch backend support.
- [ ] JAX backend support.
- [ ] General N-dimensional emulator support.

## 10. Repository Hygiene

- [ ] Add a root `LICENSE` file matching the `pyproject.toml` license.
- [ ] Add `CITATION.cff` or `CITATION.bib`.
- [ ] Add `CHANGELOG.md`.
- [ ] Add `CONTRIBUTING.md`.
- [ ] Review `.gitignore` for generated run outputs, caches, docs builds, and local scratch files.
- [ ] Remove committed `__pycache__` files from source control.
- [ ] Decide whether built distributions under `dist/` should be committed.
- [ ] Decide which generated diagnostics and figures are reference artifacts.
- [ ] Remove archival scripts from the first-release branch, leaving them in commit history or moving them to an `archive` branch.
- [ ] Move paper-only generated outputs into a clearly named reproducibility area.
- [ ] Keep tiny example data in the repo; keep large data external or documented separately.

## 11. Continuous Integration

- [ ] Add a GitHub Actions workflow for tests.
- [ ] Run tests on supported Python versions.
- [ ] Add a workflow job for package build.
- [ ] Add a workflow job for docs build.
- [ ] Add dependency caching for faster CI.
- [ ] Add a manual workflow for full paper reproduction if it is too expensive for every pull request.
- [ ] Add status badges to `README.md` after CI is working.

## 12. Versioning And Release Artifacts

- [ ] Confirm version number in `pyproject.toml`.
- [ ] Confirm version number in `src/smlr/__init__.py`.
- [ ] Decide whether versioning should be single-sourced.
- [ ] Build a clean wheel and source distribution.
- [ ] Install the wheel in a clean environment.
- [ ] Run import, validation, and CLI smoke tests from the installed wheel.
- [ ] Generate release notes from the completed checklist.
- [ ] Tag the release in git.
- [ ] Archive or publish data needed for paper reproduction.
- [ ] Link the release to the paper citation and data archive.

## 13. First-Release Acceptance Criteria

The release is ready when all of the following are true:

- [ ] A new user can install the package from the repo instructions.
- [ ] A new user can validate the paper data layout.
- [ ] A new user can run at least one fast paper-reproduction smoke test.
- [ ] A new user can train a documented 2D spectral emulator.
- [ ] A new user can run paper-style diagnostics.
- [ ] A new user can load a serialized emulator and make a prediction.
- [ ] The test suite passes.
- [ ] The package builds into a wheel and source distribution.
- [ ] The documentation explains the supported scope and the known limitations.
- [ ] Archival scripts and generated artifacts are clearly separated from supported package code.
