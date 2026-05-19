# Beta_decay_package

Package-style beta-decay QRPA emulator workflows, structured to match the
current `Dipole_polarizability/src` workflow while preserving the beta-decay
EM1 and EM2 model definitions.

Main entry points:

```bash
python -m Beta_decay_package.src.main_gpt2 --help
python -m Beta_decay_package.src.main_only_HL_gpt --help
```

The corresponding legacy/original workflows are:

```bash
python -m Beta_decay.main --help
python -m Beta_decay.main_only_HL --help
```

Run artifacts default to:

- `Beta_decay_package/runs_em1/`
- `Beta_decay_package/runs_em2/`

The package helper resolves `beta_decay_data_Ni_80/` relative to the repository
root, matching the cleaned legacy workflow.

Matched original/package sweep commands:

```bash
python Beta_decay_package/scripts/run_beta_hyperparameter_sweep.py --dry-run
python Beta_decay_package/scripts/run_beta_hyperparameter_sweep.py --which both --plots none
```

Use `--python /path/to/python` if the active interpreter does not have the
scientific runtime dependencies installed.
