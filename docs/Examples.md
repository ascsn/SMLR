# Examples

This page uses the dipole-polarizability dataset as the first-release example
for general two-parameter spectral emulation. The beta-decay dataset is covered
in `docs/Reproducibility.md` because it is the primary paper-reproduction path.

## Shipping Example Data

Recommended release layout:

```text
examples/
  data/
    dipole_polarizability_160Yb/
      total_strength/
      total_alphaD/
  specs/
    dp_2d_spectral.json
```

Keep user-playground data under `examples/data/`, not at package import time
inside `src/smlr/`. The Python package should stay small and importable without
large research data files. If the DP dataset is too large for normal git, use
Git LFS, a release artifact, or a small downloader script that verifies a
checksum. The spec can then point to either the checked-in example path or the
downloaded data path.

For the current source tree, the DP files are expected at:

```text
dipole_polarizability_160Yb/
  total_strength/strength_<beta>_<alpha>.out
  total_alphaD/alphaD_<beta>_<alpha>.out
```

Each strength file has two numeric columns: column 0 is the energy coordinate
and column 1 is the strength.

## DP 2D Spectral Spec

The example spec is:

```text
examples/specs/dp_2d_spectral.json
```

It treats the DP dataset as a generic two-parameter LRT problem:

- `p1` is the physical `alpha` parameter.
- `p2` is the physical `beta` parameter.
- Training uses `strength_(?P<p2>...)_(?P<p1>...).out` because the current DP
  filenames store beta first and alpha second.
- The model uses the generic `DipoleAdapter`, not `PaperDipoleAdapter`.

Validate the example data and spec with:

```bash
uv run smlr validate dipole-paper \
  --strength-dir dipole_polarizability_160Yb/total_strength

uv run smlr spec file examples/specs/dp_2d_spectral.json
```

## Train

Full example run:

```bash
uv run smlr train dipole-2d-example -- \
  --n 10 \
  --retain 0.5 \
  --fold 2.0 \
  --w-strength 1.0 \
  --w-alphaD 1.0 \
  --w-m1 0.0 \
  --n-restarts 1 \
  --seed0 42 \
  --num-iter 30000 \
  --print-every 1000 \
  --plots save \
  --save-dir runs/dp_2d_spectral
```

Fast smoke test:

```bash
uv run smlr train dipole-2d-example -- \
  --n 4 \
  --retain 0.5 \
  --fold 2.0 \
  --n-restarts 1 \
  --seed0 42 \
  --num-iter 3 \
  --print-every 1 \
  --plots none \
  --save-dir /private/tmp/smlr-dp-2d-smoke
```

The command writes `emulator.json`, `params.txt`, train/test parameter tables,
per-seed cost history, and metrics into the chosen output directory.

## Predict

Predict at one point:

```bash
uv run smlr predict \
  --emulator runs/dp_2d_spectral \
  --domain dipole \
  --points '[[1.10, 2.75]]' \
  --central-point '[1.10, 2.75]' \
  --energy '[0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0]' \
  --output runs/dp_2d_spectral/prediction_1p.json \
  --json
```

Predict at several points:

```bash
uv run smlr predict \
  --emulator runs/dp_2d_spectral \
  --domain dipole \
  --points '[[0.714, 2.50], [1.10, 2.75], [1.486, 3.25]]' \
  --central-point '[1.10, 2.75]' \
  --energy '[0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0]' \
  --output runs/dp_2d_spectral/predictions.json \
  --json
```

The point order is `[p1, p2]`, which is `[alpha, beta]` for this DP example.

## Adapt To Your Data

To adapt the example to a new two-parameter strength dataset:

1. Put files in one directory using `strength_<p1>_<p2>.out`, or write a regex
   with named groups that matches your existing filenames.
2. Ensure every strength file has exactly two numeric columns: energy and
   strength.
3. Use a rectangular parameter grid for the first release.
4. Copy `examples/specs/dp_2d_spectral.json` and update `data_dir`,
   `filename_regex`, `parameter_names`, `train_filter_ranges`, and
   `central_point`.
5. Run `smlr validate strength-grid` before training.

For the release-supported generic filename convention:

```bash
uv run smlr validate strength-grid path/to/total_strength
```

For a custom beta-first, alpha-second naming convention like the DP example:

```bash
uv run smlr validate strength-grid path/to/total_strength \
  --regex 'strength_(?P<p2>[0-9.]+)_(?P<p1>[0-9.]+)\.out' \
  --parameter p1 \
  --parameter p2
```
