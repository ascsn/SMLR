#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 15 11:44:53 2025

@author: anteravlic
"""


import os
import re
import argparse
import random as rn
import numpy as np
import tensorflow as tf


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from numpy.polynomial.polynomial import Polynomial

import helper


# -----------------------------
# CLI
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="EM2 training (half-lives only), headless & parsed")
    p.add_argument("--n",            type=int,   default=9,        help="Model dimension (default: 9)")
    p.add_argument("--n-restarts",   type=int,   default=4,        help="Number of different seeds")
    p.add_argument("--seed0",        type=int,   default=42,       help="Base seed (seeds = seed0..seed0+n_restarts-1)")
    p.add_argument("--num-iter",     type=int,   default=30000,    help="Iterations per restart (must be >= 20000)")
    p.add_argument("--print-every",  type=int,   default=1000,      help="Logging/plotting cadence (iterations)")
    p.add_argument("--save-dir",     type=str,   default="runs_em2", help="Output directory for run artifacts")
    p.add_argument("--plots",        choices=["none","save"], default="save",
                   help="Training plots: none (default) or save images to disk")
    p.add_argument("--phase-plot",   action="store_true", default=False,
                   help="If set, save the phase-space plot (per seed) when --plots save.")
    return p.parse_args()


args = parse_args()

# Bind parsed values
n              = args.n
N_RESTARTS     = args.n_restarts
SEED0          = args.seed0
NUM_ITERATIONS = args.num_iter
PRINT_EVERY    = args.print_every
SAVE_DIR       = args.save_dir
PLOTS_MODE     = args.plots             # "none" | "save"
DO_PHASE_PLOT  = args.phase_plot

os.makedirs(SAVE_DIR, exist_ok=True)

# -----------------------------
# Policy: minimum iterations
# -----------------------------
MIN_ITERATIONS = 20000
if NUM_ITERATIONS < MIN_ITERATIONS:
    raise ValueError(
        f"--num-iter ({NUM_ITERATIONS}) must be >= {MIN_ITERATIONS}. "
        f"Increase --num-iter or change MIN_ITERATIONS in the script."
    )

# -----------------------------
# Problem setup
# -----------------------------
A = 80
Z = 28
g_A = 1.2
nucnam = 'Ni_80'

def set_all_seeds(seed: int):
    rn.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

# -----------------------------
# Phase-space polynomial (for HL)
# -----------------------------
poly = helper.fit_phase_space(0, Z, A, 15)
coeffs = Polynomial(poly).coef

coefficients = tf.constant(coeffs[::-1], dtype=tf.float64)
x_values = np.linspace(0.611, 15, 100)
x_tensor = tf.constant(x_values, dtype=tf.float64)
y_values = helper.evaluate_polynomial_tf(coefficients, x_tensor).numpy()

def save_phase_plot(out_dir: str):
    if not (DO_PHASE_PLOT and PLOTS_MODE == "save"):
        return
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    ax.plot(x_values, y_values, label='phase-space poly')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('f(x)', fontsize=12)
    ax.set_title('Polynomial Plot', fontsize=14)
    ax.axhline(0, color='gray', lw=0.8)
    ax.axvline(0, color='gray', lw=0.8)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    ax.set_yscale('log')
    for i, xv in enumerate(x_values):
        if (i % 10 == 0):
            ax.scatter(xv, helper.phase_factor(0, Z, A, xv/helper.emass), color='red', s=10)
    fig.savefig(os.path.join(out_dir, "phase_space_poly.png"), bbox_inches="tight", dpi=150)
    plt.close(fig)

# -----------------------------
# Construct the data set
# -----------------------------
rn.seed(20)

strength_dir = '../beta_decay_data_' + nucnam + '/'
pattern = re.compile(r'lorm_' + re.escape(nucnam) + r'_([0-9.]+)_([0-9.]+)\.out')

formatted_alpha_values = []
formatted_beta_values  = []
all_points = []

for fname in os.listdir(strength_dir):
    m = pattern.match(fname)
    if m:
        beta_val  = m.group(1)
        alpha_val = m.group(2)
        all_points.append((alpha_val, beta_val))
        # filter region
        if (0.1 <= float(beta_val) <= 0.9) and (0.2 <= float(alpha_val) <= 1.8):
            formatted_alpha_values.append(alpha_val)
            formatted_beta_values.append(beta_val)

alpha = formatted_alpha_values
beta  = formatted_beta_values
combined = [(alpha[i], beta[i]) for i in range(len(alpha))]

# splits 
train_ratio = 1.0
cv_ratio    = 0.0
test_ratio  = 0.0

n_total = len(combined)
n_train = int(n_total * train_ratio)
n_cv    = int(n_total * cv_ratio)
n_test  = n_total - n_train - n_cv

train_set = combined[:n_train]
cv_set    = combined[n_train:n_train + n_cv]
test_set  = combined[n_train + n_cv:]

print(len(test_set), len(cv_set), len(train_set))

# -----------------------------
# Central point (closest to centroid)
# -----------------------------
combined_ar   = np.array(combined, dtype=float)
centroid      = combined_ar.mean(axis=0)
distances     = np.linalg.norm(combined_ar - centroid, axis=1)
central_index = np.argmin(distances)
central_point = tuple(combined[central_index])
print('Central data point in train set:', central_point)

# -----------------------------
# Model & data table (HL only)
# -----------------------------
D, S1, S2 = helper.nec_mat(n)
print(D.shape, S1.shape, S2.shape)

print('Reading data ...')
HLs = helper.data_table_only_HL(train_set, coeffs, g_A, nucnam)
print('Data read ...')

# Parameters: n (D) + 2 * n(n+1)/2 (S1,S2)  
nec_num_param = n + 2 * int(n * (n + 1) / 2)
print('Number of parameters:', nec_num_param)

params_shape = [D.shape, S1.shape, S2.shape]  # kept for reference/debug


# -----------------------------
# One restart training routine
# -----------------------------
def run_single_restart(seed: int, run_dir: str):
    """Runs one training restart; returns (best_cost, best_params, best_iter, full_cost_history)."""
    set_all_seeds(seed)

    # Random init per restart
    random_initial_guess = np.random.uniform(-1, 1, nec_num_param)
    params = tf.Variable(random_initial_guess, dtype=tf.float64)

    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.01)

    @tf.function
    def optimization_step():
        with tf.GradientTape() as tape:
            cost, HLs_calc = helper.cost_function_only_HL(params, n, train_set, HLs, central_point)
        gradients = tape.gradient(cost, [params])
        optimizer.apply_gradients(zip(gradients, [params]))
        return cost, HLs_calc

    cost_loop = []
    train_cost_min   = np.inf
    params_min_train = None
    min_train_iter   = -1

    for i in range(NUM_ITERATIONS):
        cost, HLs_train = optimization_step()
        try:
            current_learning_rate = float(optimizer._decayed_lr(tf.float32).numpy())
        except Exception:
            current_learning_rate = 0.01

        cost_val = float(cost.numpy())
        cost_loop.append(cost_val)

        rel = np.abs(np.array(HLs_train) - np.array(HLs)) / np.array(HLs)

        if cost_val < train_cost_min:
            train_cost_min   = cost_val
            params_min_train = params.numpy().copy()
            min_train_iter   = i

        if i % PRINT_EVERY == 0:
            print(f"[seed {seed}] Iter {i:6d} | Cost={cost_val:.6e} | LR={current_learning_rate:.3e} "
                  f"| mean r={np.mean(rel):.3e}")

            # Save HL comparison plot (never shown)
            if PLOTS_MODE == "save":
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.plot(range(len(train_set)), HLs_train, marker='.', label='Emu', ls='--', color='red')
                ax.plot(range(len(train_set)), HLs,        marker='.', label='QRPA', ls='--', color='black')
                ax.set_yscale('log')
                ax.set_title(f'iter = {i} (seed={seed})')
                ax.legend()
                fig.savefig(os.path.join(run_dir, f"train_hl_compare_iter{i}.png"), bbox_inches='tight', dpi=130)
                plt.close(fig)

    # Per-restart summary plot of the cost curve (save only)
    if PLOTS_MODE == "save":
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(range(len(cost_loop)), cost_loop, label='Training set')
        ax.set_yscale('log')
        ax.set_xlabel('Number of training iterations', size=12)
        ax.set_ylabel('Cost function', size=12)
        ax.set_title(f'Cost history (seed={seed}) | best={train_cost_min:.3e} @ iter {min_train_iter}')
        ax.legend()
        fig.savefig(os.path.join(run_dir, f"cost_history_seed{seed}.png"), bbox_inches='tight', dpi=150)
        fig.savefig(os.path.join(run_dir, f"cost_history_seed{seed}.pdf"), bbox_inches='tight')
        plt.close(fig)

    return train_cost_min, params_min_train, min_train_iter, cost_loop


# -----------------------------
# Multi-start driver (N_RESTARTS)
# -----------------------------
global_best_cost   = np.inf
global_best_params = None
global_best_meta   = {"seed": None, "iter": None}

for r in range(N_RESTARTS):
    seed = SEED0 + r
    run_dir = os.path.join(SAVE_DIR, f"seed_{seed}")
    os.makedirs(run_dir, exist_ok=True)

    print(f"\n========== RESTART {r+1}/{N_RESTARTS} (seed={seed}) ==========")
    print(f"Policy: MIN_ITERATIONS = {MIN_ITERATIONS}, user max = {NUM_ITERATIONS}")

    # Save per-seed phase plot if requested
    save_phase_plot(run_dir)

    best_cost, best_params, best_iter, _ = run_single_restart(seed, run_dir)

    print(f"[seed {seed}] Best train cost in this restart: {best_cost:.6e} at iter {best_iter}")

    # Save per-restart artifacts (params & meta for the restart)
    np.savetxt(os.path.join(run_dir, f"params_n{n}_seed{seed}.txt"), best_params)
    with open(os.path.join(run_dir, "meta.txt"), "w") as f:
        f.write(f"best_cost={best_cost}\n")
        f.write(f"best_iter={best_iter}\n")
        f.write(f"seed={seed}\n")
        f.write(f"n={n}\n")
        f.write(f"min_iterations_enforced={MIN_ITERATIONS}\n")
        f.write(f"num_iterations={NUM_ITERATIONS}\n")

    # Track global best
    if best_cost < global_best_cost:
        global_best_cost   = best_cost
        global_best_params = best_params.copy()
        global_best_meta   = {"seed": seed, "iter": best_iter}

# -----------------------------
# Save the global best 
# -----------------------------
print('Final (global best) train cost: {:.6e}'.format(global_best_cost))
print('Best at iteration: ', global_best_meta["iter"], ' seed: ', global_best_meta["seed"])

# Save parameters in a file at top-level 
np.savetxt(f'params_{n}_only_HL.txt', global_best_params)

# Persist the train set for reference
with open("train_set.txt", "w") as f:
    for tup in train_set:
        f.write(",".join(map(str, tup)) + "\n")
