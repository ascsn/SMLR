# Changelog

All notable changes to SMLR will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive documentation overhaul
- Professional README with badges and feature highlights
- Contributing guidelines (CONTRIBUTING.md)
- MIT License file
- Enhanced MkDocs documentation site
- API reference with detailed parameter descriptions
- Usage tutorials and examples
- Paper reproduction workflows

## [0.1.0] - 2025-01-XX

### Added
- Initial release of generalized SMLR package
- `StrengthDataset` class for flexible data management
  - Support for arbitrary parameter dimensions
  - CSV-based data loading
  - Array-based construction
  - Train/validation/test splitting
- `LorentzianMixture` model for strength function compression
  - Physics-informed fitting with SciPy
  - Global and per-component width strategies
  - Evaluation on arbitrary energy grids
- `StrengthEmulator` for parameter-space emulation
  - Linear regression-based surrogate modeling
  - Automatic feature scaling
  - Mixture parameter prediction
  - Spectrum reconstruction
- Validation metrics
  - Normalized L² error
  - Mean absolute relative error
- Plotting utilities
  - Headless matplotlib backend (Agg)
  - Spectrum visualization
  - Comparison plots
  - Mixture component decomposition
- Demonstration scripts
  - Synthetic data generator
  - Paper reproduction workflows
- Comprehensive test suite
  - Unit tests for all core modules
  - Integration tests
  - >90% code coverage
- Type hints throughout codebase
- Ruff-based code formatting and linting

### Legacy
- Preserved `Beta_decay/` and `Dipole_polarizability/` directories for reproducibility
- Original QRPA calculation scripts remain unchanged

## Release Notes

### Version 0.1.0

This is the first official release of SMLR as a generalized Python package. Previous versions existed as domain-specific scripts for beta-decay and dipole polarizability calculations.

**Key Features:**
- Domain-agnostic strength function emulation
- Production-ready code quality
- Professional documentation
- Reproducible workflows

**Migration from Legacy Scripts:**
If you were using the beta-decay or dipole-polarizability scripts directly, the new package API offers:
- Cleaner, more maintainable code
- Better error handling
- Type safety
- Comprehensive tests
- Extensibility to new domains

See the [Usage Guide](docs/usage.md) for migration examples.

---

**Note:** Dates will be filled in upon actual releases. This changelog will be updated with each version.
