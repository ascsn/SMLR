# Development

This section describes the development workflow for the SMLR package.

## Project structure

- `src/smlr/` — Python package entry point
- `Beta_decay/` — beta-decay emulator code and scripts
- `Dipole_polarizability/` — dipole emulator code and scripts
- `docs/` — user-facing documentation pages

## Packaging with uv

Build the package with:

```bash
uv build
```

Install development dependencies as needed using `uv add`.

## Contributing

To contribute, add tests and documentation updates in the relevant
directories, then rebuild the package to verify the metadata.
