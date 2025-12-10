# Paper reproduction guide

This page redoes the beta-decay and dipole strength workflows **using the new `smlr` package** and
shows the exact outputs and figures from a fresh run.

## Environment (uv recommended)
```bash
curl -Ls https://astral.sh/uv/install.sh | sh
uv sync --group dev --group docs   # installs into .venv with docs + dev extras
```

## Re-run the packaged workflow
```bash
uv run python examples/paper_repro.py --mode paper --max-files 6 --plots-dir docs/figs
```
- Uses `beta_decay_data_Ni_80/` and `dipoles_data_all/` if present; otherwise the script prints a
  fallback note and runs the synthetic demo.
- Saves comparison plots under `docs/figs/` for inclusion in the site.

### Current reproduced numbers (this repo state)
- Beta decay (Ni-80) emulator vs nearest reference: normalized L2 = **0.377**
- Dipole (Yb) emulator vs nearest reference: normalized L2 = **0.039**

![Beta decay emulator](figs/beta_decay.png)

![Dipole emulator](figs/dipole.png)

### What the script does
1) Loads a small subset of spectra (default `--max-files 6`) for each dataset.
2) Fits a 3-pole `StrengthEmulator` with global width.
3) Evaluates at the mean parameter vector and compares to the nearest neighbor spectrum.
4) Reports normalized L2 and saves headless matplotlib plots.

## Notebook & script entry points
- Notebook: `docs/paper_repro_notebook.ipynb` (rendered here; set `RUN_HEAVY=true` to use more files).
- Script: `examples/paper_repro.py`
  - Synthetic smoke test: `uv run python examples/paper_repro.py --mode synthetic --out runs/demo-script`
  - Paper subset: `uv run python examples/paper_repro.py --mode paper --max-files 6 --plots-dir docs/figs`

## Adapting to new data locations
- Point the script/notebook loaders to your directories (see loader logic inside `paper_repro.py`).
- Keep the metadata convention: two-column spectra files plus parameter vectors supplied in the file
  names or a CSV.
- All plotting is headless (Agg) and saved to disk for reproducibility.
