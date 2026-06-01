# Reproducibility

This page records the first-release paper reproduction path for the beta-decay
case. Use `uv` for setup and run package commands with `uv run`.

## Input Data

Beta decay, Ni-80:

- Root: `beta_decay_80Ni/`
- Strength files: `beta_decay_80Ni/total_lorm/lorm_Ni_80_<beta>_<alpha>.out`
- Excitation files: `beta_decay_80Ni/total_excm/excm_Ni_80_<beta>_<alpha>.out`
- Half-life files: `beta_decay_80Ni/total_half_life/half_life_Ni_80_<beta>_<alpha>.txt`

Dipole polarizability, Yb-160:

- Root: `dipole_polarizability_160Yb/`
- Strength files: `dipole_polarizability_160Yb/total_strength/strength_<beta>_<alpha>.out`
- Polarizability files: `dipole_polarizability_160Yb/total_alphaD/alphaD_<beta>_<alpha>.out`

For strength-like files, column 0 is the energy coordinate and column 1 is the
strength.

## Validate Data

```bash
uv sync --extra paper
uv run smlr validate beta-paper --data-dir beta_decay_80Ni
uv run smlr validate dipole-paper --strength-dir dipole_polarizability_160Yb/total_strength
```

## Beta-Decay EM1 Paper Run

The paper n=13 beta-decay example is represented by
`examples/specs/bd_n13_paper.json`. The training command is:

```bash
uv run smlr train beta-paper-em1 -- \
  --data-dir beta_decay_80Ni \
  --n 13 \
  --retain 0.9 \
  --weight 1.0 \
  --n-restarts 1 \
  --seed0 42 \
  --num-iter 30000 \
  --print-every 1000 \
  --plots save \
  --save-dir runs/bd_n13_paper
```

For a quick smoke test of the command path:

```bash
uv run smlr train beta-paper-em1 -- \
  --data-dir beta_decay_80Ni \
  --n 13 \
  --retain 0.9 \
  --weight 1.0 \
  --n-restarts 1 \
  --seed0 42 \
  --num-iter 3 \
  --min-iter 0 \
  --print-every 1 \
  --plots none \
  --save-dir /private/tmp/smlr-bd-n13-smoke
```

## Expected Outputs

For the beta-decay command above with `--save-dir runs/bd_n13_paper`, expected
outputs include:

- `runs/bd_n13_paper/params_best_n13_retain0.9.txt`
- `runs/bd_n13_paper/train_set.txt`
- `runs/bd_n13_paper/emulator.json`
- `runs/bd_n13_paper/params.txt`
- `runs/bd_n13_paper/seed_42/params_n13_retain0.9_seed42.txt`
- `runs/bd_n13_paper/seed_42/cost_history_seed42.txt`
- `runs/bd_n13_paper/seed_42/meta.txt`
- optional plot files under `runs/bd_n13_paper/seed_42/` when `--plots save`

For the dipole EM1 paper workflow, expected package outputs include:

- `runs/dp_em1_paper/best_params_global.txt`
- `runs/dp_em1_paper/train_param_values.txt`
- `runs/dp_em1_paper/test_param_values.txt`
- `runs/dp_em1_paper/run_summary.json`
- `runs/dp_em1_paper/emulator.json`
- `runs/dp_em1_paper/params.txt`
- `runs/dp_em1_paper/seed_42/best_params.txt`
- `runs/dp_em1_paper/seed_42/cost_history.txt`
- `runs/dp_em1_paper/seed_42/meta.json`
- `runs/dp_em1_paper/seed_42/metrics.json`
