#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 16:31:28 2025

@author: anteravlic + ChatGPT formatting
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dipole Polarizability Emulator — EM2 (alphaD-only)

User-facing script to train a half-cost emulator for dipole polarizability.
- Headless: never opens GUI windows.
- Figures are NEVER shown; use `--plots save` to write PNG/PDFs.
- Per-seed artifacts are saved under `<save-dir>/seed_<SEED>/`.
- Enforces a minimum of 20,000 iterations per restart.
- Saves ONLY the global-best params to: params_<n>_only_alphaD.txt

Examples
--------
# 1) Fast default (headless, no figures)
python dipole_em2_only_alphaD.py

# 2) Customized run with saved figures
python dipole_em2_only_alphaD.py --n 9 --n-restarts 5 --seed0 7 \
    --num-iter 40000 --print-every 500 --plots save --save-dir runs_dipole_em2
"""

import os, re, argparse, random as rn
import numpy as np
import tensorflow as tf

# Headless plotting backend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import helper


# ------------------------------ CLI (user options) ------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Train the dipole polarizability emulator (EM2, alphaD-only)."
    )
    p.add_argument("--n",            type=int,   default=9,        help="Model dimension (default: 9)")
    p.add_argument("--n-restarts",   type=int,   default=5,        help="Number of restarts (different seeds)")
    p.add_argument("--seed0",        type=int,   default=42,       help="Base seed (seeds=seed0..seed0+n_restarts-1)")
    p.add_argument("--num-iter",     type=int,   default=30000,    help="Iterations per restart (must be ≥ 20000)")
    p.add_argument("--print-every",  type=int,   default=1000,     help="Logging cadence (iterations)")
    p.add_argument("--save-dir",     type=str,   default="runs_em2", help="Directory for run artifacts")
    p.add_argument("--plots",        choices=["none","save"], default="save",
                   help="Skip plots (none) or save figures (save).")
    return p.parse_args()


args = parse_args()

n              = args.n
N_RESTARTS     = args.n_restarts
SEED0          = args.seed0
NUM_ITERATIONS = args.num_iter
PRINT_EVERY    = args.print_every
SAVE_DIR       = args.save_dir
PLOTS_MODE     = args.plots  # "none" | "save"

os.makedirs(SAVE_DIR, exist_ok=True)

# Policy: enforce a hard minimum number of iterations per restart
MIN_ITERATIONS = 20000
if NUM_ITERATIONS < MIN_ITERATIONS:
    raise ValueError(f"--num-iter ({NUM_ITERATIONS}) must be >= {MIN_ITERATIONS}.")


# ------------------------------ Utilities --------------------------------------
def set_all_seeds(seed: int):
    rn.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def moving_average(arr, k):
    if len(arr) < k:
        return None
    return np.convolve(arr, np.ones(k)/k, mode="valid")


# ------------------------------ Dataset build ----------------------------------
# User note: adjust these folders if dataset lives elsewhere.
rn.seed(495)
np.random.seed(58)

strength_dir = "../dipoles_data_all/total_strength/"
alphaD_dir   = "../dipoles_data_all/total_alphaD/"

# Files look like: strength_<beta>_<alpha>.out
pattern = re.compile(r"strength_([0-9.]+)_([0-9.]+)\.out")

formatted_alpha_values, formatted_beta_values, all_points = [], [], []
for fname in os.listdir(strength_dir):
    m = pattern.match(fname)
    if not m:
        continue
    beta_val  = m.group(1)
    alpha_val = m.group(2)
    all_points.append((alpha_val, beta_val))
    # Keep a rectangular subgrid (tweak if needed)
    if (1.5 <= float(beta_val) <= 4.0) and (0.4 <= float(alpha_val) <= 1.8):
        formatted_alpha_values.append(alpha_val)
        formatted_beta_values.append(beta_val)

alpha = formatted_alpha_values
beta  = formatted_beta_values
combined = [(alpha[i], beta[i]) for i in range(len(alpha))]

if not combined:
    raise RuntimeError(f"No training points found in {strength_dir}. Check file names and filters.")

# Train split only (consistent with original)
train_set = combined
cv_set, test_set = [], []

# Central point by bounding-box center
Amin = min(float(a) for a, b in combined); Amax = max(float(a) for a, b in combined)
Bmin = min(float(b) for a, b in combined); Bmax = max(float(b) for a, b in combined)
cx, cy = (Amin + Amax) / 2.0, (Bmin + Bmax) / 2.0
central_point = min(combined, key=lambda t: (float(t[0]) - cx)**2 + (float(t[1]) - cy)**2)
print("Central data point in train set:", central_point)

# Model matrices (shape info is often useful for debugging)
D, S1, S2 = helper.nec_mat(n)
print("Matrix shapes:", D.shape, S1.shape, S2.shape)

# Data table (alphaD targets)
strength, alphaD = helper.data_table(train_set)
alphaD = np.vstack(alphaD)                    # shape (N, >= 3)
alphaD_list = [float(a[2]) for a in alphaD]   # target alphaD values


# ------------------------------ Param count ------------------------------------
# Matches the original EM2 alphaD-only parameterization
nec_num_param = n + 3 * int(n * (n + 1) / 2) + 1
print("Number of parameters:", nec_num_param)


# ------------------------------ One restart ------------------------------------
def run_single_restart(seed: int, run_dir: str):
    """Run one restart; return (best_cost, best_params, best_iter, cost_history)."""
    set_all_seeds(seed)

    # Random init per restart (same range as original)
    random_initial_guess = np.random.uniform(0, 2, nec_num_param)
    params    = tf.Variable(random_initial_guess, dtype=tf.float32)
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.05)

    @tf.function
    def optimization_step():
        with tf.GradientTape() as tape:
            cost, alphaD_train = helper.cost_function_only_alphaD_batched(
                params, n, train_set, alphaD_list, central_point
            )
        grads = tape.gradient(cost, [params])
        optimizer.apply_gradients(zip(grads, [params]))
        return cost, alphaD_train

    cost_loop        = []
    train_cost_min   = np.inf
    params_min_train = None
    min_train_iter   = -1

    # Simple MA display on saved plots (when enabled)
    MA_WINDOW = max(1, PRINT_EVERY // 2)

    for i in range(NUM_ITERATIONS):
        cost, alphaD_pred = optimization_step()

        # LR readout (fallback safe)
        try:
            current_lr = float(getattr(optimizer, "lr", optimizer._decayed_lr(tf.float32)).numpy())
        except Exception:
            current_lr = 0.05

        c = float(cost.numpy())
        cost_loop.append(c)

        rel = np.abs(np.array(alphaD_pred) - np.array(alphaD[:, 2])) / np.array(alphaD[:, 2])

        if c < train_cost_min:
            train_cost_min   = c
            params_min_train = params.numpy().copy()
            min_train_iter   = i

        if (i % PRINT_EVERY) == 0:
            print(f"[seed {seed}] iter {i:6d} | cost={c:.6e} | lr={current_lr:.3e} "
                  f"| mean rel={float(np.mean(rel)):.6e}")

            # Save training comparison figure (never shown)
            if PLOTS_MODE == "save":
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.plot(range(len(train_set)), alphaD_pred, marker='.', ls='--', color='red',  label='Emu')
                ax.plot(range(len(train_set)), alphaD[:, 2],  marker='.', ls='--', color='black',label='QRPA')
                ax.set_yscale('log')
                ax.set_title(f'iter = {i} (seed={seed})')
                ax.legend()
                fig.savefig(os.path.join(run_dir, f"train_alphaD_compare_iter{i}.png"),
                            bbox_inches='tight', dpi=130)
                plt.close(fig)

    # Cost curve (save only)
    if PLOTS_MODE == "save":
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(range(len(cost_loop)), cost_loop, label='Training set')
        # Moving average overlay in iteration space
        if len(cost_loop) >= MA_WINDOW:
            ma = moving_average(cost_loop, MA_WINDOW)
            ax.plot(range(MA_WINDOW-1, MA_WINDOW-1+len(ma)), ma, ls='--', label='moving avg')
        ax.set_yscale('log')
        ax.set_xlabel('Iteration', size=12)
        ax.set_ylabel('Cost', size=12)
        ax.set_title(f'Cost history (seed={seed}) | best={train_cost_min:.3e} @ iter {min_train_iter}')
        ax.legend()
        fig.savefig(os.path.join(run_dir, f"cost_history_seed{seed}.png"), bbox_inches='tight', dpi=150)
        fig.savefig(os.path.join(run_dir, f"cost_history_seed{seed}.pdf"), bbox_inches='tight')
        plt.close(fig)

    return train_cost_min, params_min_train, min_train_iter, cost_loop


# ------------------------------ Multi-start driver -----------------------------
global_best_cost   = np.inf
global_best_params = None
global_best_meta   = {"seed": None, "iter": None}

for r in range(N_RESTARTS):
    seed = SEED0 + r
    run_dir = os.path.join(SAVE_DIR, f"seed_{seed}")
    os.makedirs(run_dir, exist_ok=True)

    print(f"\n========== RESTART {r+1}/{N_RESTARTS} (seed={seed}) ==========")
    print(f"Policy: MIN_ITERATIONS = {MIN_ITERATIONS}, user max = {NUM_ITERATIONS}")

    best_cost, best_params, best_iter, _ = run_single_restart(seed, run_dir)

    print(f"[seed {seed}] Best train cost in this restart: {best_cost:.6e} at iter {best_iter}")

    # Save per-restart metadata & params
    np.savetxt(os.path.join(run_dir, f"params_n{n}_seed{seed}.txt"), best_params)
    with open(os.path.join(run_dir, "meta.txt"), "w") as f:
        f.write(f"best_cost={best_cost}\n")
        f.write(f"best_iter={best_iter}\n")
        f.write(f"seed={seed}\n")
        f.write(f"n={n}\n")
        f.write(f"MIN_ITERATIONS={MIN_ITERATIONS}\n")
        f.write(f"NUM_ITERATIONS={NUM_ITERATIONS}\n")
        f.write(f"PRINT_EVERY={PRINT_EVERY}\n")

    # Track global best
    if best_cost < global_best_cost:
        global_best_cost   = best_cost
        global_best_params = best_params.copy()
        global_best_meta   = {"seed": seed, "iter": best_iter}

# ------------------------------ Save global best -------------------------------
print('Final (global best) train cost: {:.6e}'.format(global_best_cost))
print('Best at iteration: ', global_best_meta["iter"], ' seed: ', global_best_meta["seed"])

np.savetxt(f'params_{n}_only_alphaD.txt', global_best_params)

# Persist the training set for reference
with open('train_set.txt', "w") as f:
    for tup in train_set:
        f.write(",".join(map(str, tup)) + "\n")
