# Getting Started

## Install the package

Use `uv` for local setup. `uv` creates and manages the project virtual
environment from `pyproject.toml` and `uv.lock`.

```bash
cd /Users/laurenjin/Documents/projects/SMLR-package/SMLR-jingy
uv sync --extra paper
``` 

For a smaller install that only includes the base package dependencies:

```bash
uv sync
```

For documentation work:

```bash
uv sync --extra docs
```

You do not need to activate the virtual environment manually. Run package
commands with `uv run`.

## Validate data

Check the paper data layout before training:

```bash
uv run smlr validate beta-paper --data-dir beta_decay_80Ni
uv run smlr validate dipole-paper --strength-dir dipole_polarizability_160Yb/total_strength
```

For general two-parameter linear-response strength data, use the strength-grid
validator. The default accepted format is `strength_<p1>_<p2>.out`; each file
must contain exactly two numeric columns, with column 0 equal to omega and
column 1 equal to B strength:

```bash
uv run smlr validate strength-grid path/to/strengths
```

## Run supported workflows

Use the `smlr` command for first-release workflows:

```bash
uv run smlr spec beta-paper-em1
uv run smlr train beta-paper-em1 -- --help
uv run smlr diagnose beta-paper -- --help
```

## Project layout

- `src/smlr/` — supported package code and command-line entry points
- `Beta_decay/` — research and paper workflow code from the development history
- `Dipole_polarizability/` — research and paper workflow code from the development history
- `docs/` — documentation pages with a structured table of contents

For `0.1.0`, code outside the documented `smlr` package commands should be
treated as research or archival code unless a workflow explicitly points to it.
