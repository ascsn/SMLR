#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Beta-decay emulator main (restart-enabled + early stopping + per-restart plots)
"""

import os
import re
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import Polynomial
import random as rn

import helper

# ============================
# Problem setup / constants
# ============================
A = 80
Z = 28
g_A = 1.2
nucnam = 'Ni_80'

# ============================
# Phase-space polynomial (for HL)
# ============================
poly = helper.fit_phase_space(0, Z, A, 15)
coeffs = Polynomial(poly).coef

coefficients = tf.constant(coeffs[::-1], dtype=tf.float64)  # reversed for Horner
x_values = np.linspace(0.611, 15, 100, dtype=float)
x_tensor = tf.constant(x_values, dtype=tf.float64)
y_values = helper.evaluate_polynomial_tf(coefficients, x_tensor).numpy()

# Quick visualization (optional)
plt.figure(figsize=(8, 5))
plt.plot(x_values, y_values, label='phase-space poly')
plt.xlabel('x', fontsize=12)
plt.ylabel('f(x)', fontsize=12)
plt.title('Polynomial Plot', fontsize=14)
plt.axhline(0, color='gray', lw=0.8)
plt.axvline(0, color='gray', lw=0.8)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.yscale('log')
for i, xv in enumerate(x_values):
    if (i % 10 == 0):
        plt.scatter(xv, helper.phase_factor(0, Z, A, xv/helper.emass), color='red', s=10)
plt.show()

# ============================
# Build dataset from files
# ============================
rn.seed(20)

strength_dir = '../beta_decay_data_' + nucnam + '/'
pattern = re.compile(r'lorm_' + re.escape(nucnam) + r'_([0-9.]+)_([0-9.]+)\.out')

formatted_alpha_values = []
formatted_beta_values = []
all_points = []

for fname in os.listdir(strength_dir):
    match = pattern.match(fname)
    if match:
        beta_val = match.group(1)
        alpha_val = match.group(2)
        all_points.append((alpha_val, beta_val))

        # filtering region
        if ((0.1 <= float(beta_val) <= 0.9) and (0.2 <= float(alpha_val) <= 1.8)):
            formatted_alpha_values.append(alpha_val)
            formatted_beta_values.append(beta_val)

# pairs (alpha, beta)
alpha = formatted_alpha_values
beta = formatted_beta_values
combined = [(alpha[i], beta[i]) for i in range(len(alpha))]

# split ratios
train_ratio = 1.0
cv_ratio = 0.0
test_ratio = 0.0

n_total = len(combined)
n_train = int(n_total * train_ratio)
n_cv = int(n_total * cv_ratio)
n_test = n_total - n_train - n_cv

train_set = combined[:n_train]
cv_set = combined[n_train:n_train + n_cv]
test_set = combined[n_train + n_cv:]

# central point (centroid)
combined_ar = np.array(combined, dtype=float)
centroid = combined_ar.mean(axis=0)
distances = np.linalg.norm(combined_ar - centroid, axis=1)
central_index = np.argmin(distances)
central_point = tuple(combined[central_index])
print('Central data point in train set:', central_point)
print('Sizes -> test:', len(test_set), 'cv:', len(cv_set), 'train:', len(train_set))

# ============================
# Model dimensions & data table
# ============================
np.random.seed(42)  # base seed (dataset is fixed independent of restarts)

n = 17       # model dimension
retain = 0.9    # fraction of eigenmodes kept (centered)
weight = 1.0    # initial HL weight (consider calibrating)
width = 2.0     # (unused unless you re-enable the init fit section)

D, S1, S2 = helper.nec_mat(n)
print('Matrix shapes:', D.shape, S1.shape, S2.shape)

# Build tables once (train only here)
Lors, HLs = helper.data_table(train_set, coeffs, g_A, nucnam)

# Parameter count:
# n - diagonal (D)
# 2 * n(n+1)/2 - two symmetric matrices (S1, S2)
# n - v0
# 4 - eta, x1, x2, x3
nec_num_param = n + 2 * int(n * (n + 1) / 2) + n + 4
print('Number of parameters:', nec_num_param)

# ============================
# Restart utilities & config
# ============================
def set_all_seeds(seed: int):
    rn.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def make_initial_guess(nec_num_param: int, seed: int):
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 1.0, size=nec_num_param)

# Multi-start config
N_RESTARTS     = 1              # number of different seeds
SEED0          = 42             # seeds: SEED0..SEED0+N_RESTARTS-1
NUM_ITERATIONS = 30000          # max iterations per restart
PRINT_EVERY    = 300            # log / plotting cadence
SHOW_PLOTS     = True           # set False to suppress iterative plots
SAVE_DIR       = "runs_beta"
os.makedirs(SAVE_DIR, exist_ok=True)

# Early-stopping config
PRINT_EVERY         = 200     # check a bit more often
PATIENCE_BEST       = 120     # was 60  (≈ 24k iters of tolerance)
MIN_DELTA_REL_BEST  = 2e-4    # was 5e-4 (easier to count as an improvement)

PATIENCE_PLATEAU    = 300     # was 50  (≈ 24k iters before plateau stop)
MA_WINDOW           = 10      # was 5   (smoother trend; less false plateaus)
MIN_DELTA_REL       = 3e-4    # was 1e-3 (accept slower MA improvements)

def moving_average(arr, k):
    if len(arr) < k:
        return None
    return np.convolve(arr, np.ones(k)/k, mode='valid')

# ============================
# Multi-start training loop
# ============================
global_best_cost = np.inf
global_best_params = None
global_best_meta = {"seed": None, "iter": None}

last_cost_history = None  # for final plot

for r in range(N_RESTARTS):
    seed = SEED0 + r
    print(f"\n========== RESTART {r+1}/{N_RESTARTS} (seed={seed}) ==========")
    set_all_seeds(seed)

    init_vec = make_initial_guess(nec_num_param, seed)
    params = tf.Variable(init_vec, dtype=tf.float64)
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.01)

    # compile a fresh step bound to THIS optimizer/params (avoids TF variable-creation error)
    @tf.function
    def optimization_step():
        with tf.GradientTape() as tape:
            cost, Lor, Lor_true, x, HLs_calc, B_last, E_last = helper.cost_function(
                params, n, train_set, Lors, HLs, coeffs, g_A, weight, central_point, retain
            )
        grads = tape.gradient(cost, [params])
        optimizer.apply_gradients(zip(grads, [params]))
        return cost, Lor, Lor_true, x, HLs_calc, B_last, E_last

    cost_history = []
    best_cost_this = np.inf
    best_params_this = None
    best_iter_this = -1

    # early stopping trackers
    no_improve_cnt = 0
    plateau_cnt    = 0
    last_ma        = None

    for i in range(NUM_ITERATIONS):
        cost, Lor, Lor_true, x, HLs_check_train, B_last, E_last = optimization_step()
        cost_val = float(cost.numpy())
        cost_history.append(cost_val)

        # track best of this restart
        #improved = (best_cost_this - cost_val) >= MIN_DELTA_ABS
        improved = (best_cost_this - cost_val) >= MIN_DELTA_REL_BEST * max(abs(best_cost_this), 1e-12)
        if improved or (best_cost_this == np.inf):
            best_cost_this = cost_val
            best_params_this = params.numpy().copy()
            best_iter_this = i
            no_improve_cnt = 0
        else:
            no_improve_cnt += 1

        # logging & (optional) plotting at cadence
        if (i % PRINT_EVERY) == 0:
            try:
                current_lr = float(optimizer._decayed_lr(tf.float64).numpy())
            except Exception:
                current_lr = 0.01
            rel = np.abs(np.array(HLs_check_train) - np.array(HLs)) / np.array(HLs)
            print(f"[seed {seed}] iter {i:6d} | cost={cost_val:.6e} | lr={current_lr:.3e} "
                  f"| mean r={np.mean(rel):.3e} | no_improve={no_improve_cnt}")

            if SHOW_PLOTS:
                plt.figure(figsize=(10, 4))
                plt.subplot(1, 2, 1)
                plt.plot(np.arange(len(rel)), rel, marker='.', ls='--')
                plt.axhline(np.mean(rel), color='k', ls='--')
                plt.yscale('log')
                plt.title(f'Half-life rel. err. (iter={i})')

                plt.subplot(1, 2, 2)
                plt.stem(E_last, B_last, use_line_collection=True)
                plt.plot(x, Lor, label='pred')
                plt.plot(x, Lor_true, label='true')
                plt.ylim(0)
                plt.xlim(-10, np.max(x))
                plt.axvline(0.8, ls='--')
                plt.legend()
                plt.tight_layout()
                plt.show()

            # ----- moving-average plateau check (computed at cadence) -----
            ds_costs = cost_history[::PRINT_EVERY] if PRINT_EVERY > 0 else cost_history[:]
            ma = moving_average(ds_costs, MA_WINDOW)
            if ma is not None:
                current_ma = ma[-1]
                if last_ma is None:
                    last_ma = current_ma
                    plateau_cnt = 0
                else:
                    rel_impr = (last_ma - current_ma) / max(abs(last_ma), 1e-12)
                    if rel_impr >= MIN_DELTA_REL:
                        plateau_cnt = 0
                        last_ma = current_ma
                    else:
                        plateau_cnt += 1
            # --------------------------------------------------------------

        # ----- early stopping conditions -----
        stop_best = (no_improve_cnt >= PATIENCE_BEST)
        stop_plateau = (plateau_cnt >= PATIENCE_PLATEAU)
        if stop_best or stop_plateau:
            reason = "no best improvement" if stop_best else "moving-average plateau"
            print(f"[seed {seed}] Early stopping at iter {i} due to {reason}. "
                  f"best_cost={best_cost_this:.6e} (iter={best_iter_this})")
            break
        # -------------------------------------

    # save per-restart artifacts
    run_dir = os.path.join(SAVE_DIR, f"seed_{seed}")
    os.makedirs(run_dir, exist_ok=True)

    np.savetxt(os.path.join(run_dir, f"params_n{n}_retain{retain}_seed{seed}.txt"),
               best_params_this)
    np.savetxt(os.path.join(run_dir, f"cost_history_seed{seed}.txt"),
               np.array(cost_history))
    with open(os.path.join(run_dir, "meta.txt"), "w") as f:
        f.write(f"best_cost={best_cost_this}\n")
        f.write(f"best_iter={best_iter_this}\n")
        f.write(f"seed={seed}\n")
        f.write(f"n={n}\nretain={retain}\nweight={weight}\n")
        f.write(f"stopped_at={len(cost_history)-1}\n")

    # ----- NEW: per-restart convergence plot (saved + optional show) -----
    plt.figure()
    plt.plot(range(len(cost_history)), cost_history, label=f'seed {seed}')
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.title(f'Convergence (seed={seed}) | best={best_cost_this:.3e} @ iter {best_iter_this}')
    plt.legend()
    png_path = os.path.join(run_dir, f'cost_curve_seed{seed}.png')
    pdf_path = os.path.join(run_dir, f'cost_curve_seed{seed}.pdf')
    plt.savefig(png_path, bbox_inches='tight', dpi=150)
    plt.savefig(pdf_path, bbox_inches='tight')
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()
    # ---------------------------------------------------------------------

    print(f"[seed {seed}] best_iter={best_iter_this} best_cost={best_cost_this:.6e}")

    # update global best
    if best_cost_this < global_best_cost:
        global_best_cost = best_cost_this
        global_best_params = best_params_this.copy()
        global_best_meta = {"seed": seed, "iter": best_iter_this}

    last_cost_history = cost_history  # keep for final plot

# ============================
# Save global best & summary
# ============================
np.savetxt(f'params_best_n{n}_retain{retain}.txt', global_best_params)

with open("train_set.txt", "w") as f:
    for tup in train_set:
        f.write(",".join(map(str, tup)) + "\n")

# final quick plot (last restart history)
if last_cost_history is not None:
    plt.figure()
    plt.plot(range(len(last_cost_history)), last_cost_history, label='Last restart cost')
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.legend()
    plt.title(f"Global best cost={global_best_cost:.3e} "
              f"(seed={global_best_meta['seed']}, iter={global_best_meta['iter']})")
    plt.show()

print(f"\n*** GLOBAL BEST *** cost={global_best_cost:.6e} "
      f"(seed={global_best_meta['seed']}, iter={global_best_meta['iter']})")
print(f"Saved: params_best_n{n}_retain{retain}.txt and per-seed artifacts in {SAVE_DIR}/")
