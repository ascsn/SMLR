#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

mkdir -p Dipole_polarizability/aaron_H2/sweeps/h2_strength_sweep
exec /Users/laurenjin/envs/smlr-frib/bin/python -u \
  Dipole_polarizability/aaron_H2/run_h2_hyperparam_sweep.py \
  --resume
