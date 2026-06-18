#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/Users/laurenjin/envs/smlr-frib/bin/python}"
NUM_ITER="${NUM_ITER:-20000}"
MIN_ITER="${MIN_ITER:-$NUM_ITER}"
WIDTH="${WIDTH:-0.5}"
S_INIT="${S_INIT:-0.1}"
DATA_DIR="${DATA_DIR:-data/gamow_teller_48Ca_4d/total_strength_K0}"
NS=(20 22 24 26 28 30)

for N in "${NS[@]}"; do
  WLABEL="${WIDTH//./p}"
  SAVE_DIR="runs_em1/sweeps/n/gt48ca_K0_n${N}_w${WLABEL}"
  echo "=== n=${N} width=${WIDTH} save=${SAVE_DIR} ==="
  MPLCONFIGDIR=/private/tmp/mplconfig "$PYTHON_BIN" -B Beta_decay_package/src/main_gpt2.py \
    --data-dir "$DATA_DIR" \
    --strength-window 0 30 \
    --n "$N" \
    --retain 1.0 \
    --weight 0.0 \
    --fixed-width "$WIDTH" \
    --s-init-scale "$S_INIT" \
    --reference-index 0 \
    --diag-index 1 \
    --n-restarts 1 \
    --seed0 42 \
    --num-iter "$NUM_ITER" \
    --min-iter "$MIN_ITER" \
    --print-every 2000 \
    --plots save \
    --save-dir "$SAVE_DIR"
done
