# Getting Started

## Install the package

Use `uv` to build and install the project locally:

```bash
cd /Users/laurenjin/Documents/projects/SMLR-package/SMLR-jingy
uv build
``` 

If you want to install the package into a virtual environment:

```bash
uv venv create
uv venv activate
uv install
```

## Run the examples

Each emulator module includes scripts in the `Beta_decay/` and
`Dipole_polarizability/` directories. For example:

```bash
python -m Beta_decay.main
python Dipole_polarizability/main.py
```

## Project layout

- `Beta_decay/` — beta-decay emulator workflows
- `Dipole_polarizability/` — dipole strength emulator workflows
- `docs/` — documentation pages with a structured table of contents
- `src/smlr/` — package metadata and Python package entry point
