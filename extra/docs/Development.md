# Development

This section describes the development workflow for the SMLR package.

## Project structure

- `src/smlr/` — Python package entry point
- `Beta_decay/` — beta-decay emulator code and scripts
- `Dipole_polarizability/` — dipole emulator code and scripts
- `docs/` — user-facing documentation pages

## Packaging with uv

Install the development environment with `uv`:

```bash
uv sync --extra paper --extra docs
```

Run commands through the managed virtual environment:

```bash
uv run smlr --help
uv run python -m unittest
```

Build the package with:

```bash
uv build
```

Add dependencies with `uv add` so `pyproject.toml` and `uv.lock` stay in sync.
For `0.1.0`, TensorFlow is the only supported training backend.

## Contributing

To contribute, add tests and documentation updates in the relevant
directories, then rebuild the package to verify the metadata.
