# Beta_decay Legacy Workflows

This directory contains the beta-decay QRPA research workflows used while
preparing the reusable `smlr` package. The code is intentionally kept separate
from `src/smlr` until the shared emulator API is extracted.

Run from the repository root:

```bash
python -m Beta_decay.main --help
python -m Beta_decay.main_only_HL --help
```

Generated run artifacts should go under `runs_em1/` or `runs_em2/` via
`--save-dir`. Example media lives in `docs/assets/beta_decay/` instead of this
source directory.

