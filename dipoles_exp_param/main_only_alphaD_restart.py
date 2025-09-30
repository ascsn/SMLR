#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dipole polarizability emulator (alphaD-only) with multi-start restarts (no argparse).
- Runs N_RESTARTS trainings with seeds SEED0..SEED0+N_RESTARTS-1
- Saves ONLY the global-best params to: params_{n}_only_alphaD.txt
- Keeps plotting and output structure similar to your original script.
"""

import helper
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random as rn
import re
import os

# -----------------------------
# Fixed restart settings (no CLI parsing)
# -----------------------------
N_RESTARTS = 10
SEED0 = 42

def set_all_seeds(seed: int):
    rn.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

# -----------------------------
# Build dataset (unchanged logic)
# -----------------------------
rn.seed(495)
np.random.seed(58)

strength_dir = '../dipoles_data_all/total_strength/'
alphaD_dir   = '../dipoles_data_all/total_alphaD/'

pattern = re.compile(r'strength_([0-9.]+)_([0-9.]+)\.out')

formatted_alpha_values = []
formatted_beta_values  = []
all_points = []

for fname in os.listdir(strength_dir):
    match = pattern.match(fname)
    if match:
        beta_val  = match.group(1)
        alpha_val = match.group(2)
        all_points.append((alpha_val, beta_val))
        if (1.5 <= float(beta_val) <= 4.0) and (0.4 <= float(alpha_val) <= 1.8):
            formatted_alpha_values.append(alpha_val)
            formatted_beta_values.append(beta_val)

alpha = formatted_alpha_values
beta  = formatted_beta_values

combined = []
for i in range(len(alpha)):
    combined.append((alpha[i], beta[i]))

train_ratio, cv_ratio, test_ratio = 1.0, 0.0, 0.0
n_all   = len(combined)
n_train = int(n_all * train_ratio)
n_cv    = int(n_all * cv_ratio)
n_test  = n_all - n_train - n_cv

train_set = combined[:n_train]
cv_set    = combined[n_train:n_train+n_cv]
test_set  = combined[n_train+n_cv:]

# Central point by bounding-box center (as in your beta/dipole scripts)
Amin = min(float(a) for a, b in combined); Amax = max(float(a) for a, b in combined)
Bmin = min(float(b) for a, b in combined); Bmax = max(float(b) for a, b in combined)
cx, cy = (Amin + Amax) / 2.0, (Bmin + Bmax) / 2.0
central_point = min(combined, key=lambda t: (float(t[0]) - cx)**2 + (float(t[1]) - cy)**2)
print('Central data point in train set:', central_point)

# -----------------------------
# Model size and matrices
# -----------------------------
n = 9 # your chosen dimension

D, S1, S2 = helper.nec_mat(n)
print(D.shape, S1.shape, S2.shape)

# initialize the external field vector (kept from your original)
v0 = np.random.rand(n)

# -----------------------------
# Data table (alphaD)
# -----------------------------
strength, alphaD = helper.data_table(train_set)
alphaD = np.vstack(alphaD)                   # shape (N, >=3)
alphaD_list = [float(a[2]) for a in alphaD]  # target alphaD values

# Parameter count (as in your original comment)
nec_num_param = n + 3*int(n * (n + 1) / 2) + 1
print('Number of parameters:', nec_num_param)

# -----------------------------
# One restart training routine
# -----------------------------
def run_single_restart(seed: int):
    """Run one training restart; return (best_cost, best_params, best_iter, cost_history)."""
    set_all_seeds(seed)

    # Random init per restart (same range as your original)
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

    num_iterations = 30000
    num_check      = 1000
    cost_loop      = []

    # Track best within this restart
    train_cost_min   = np.inf
    params_min_train = None
    min_train_iter   = -1

    for i in range(num_iterations):
        cost, alphaD_check_train = optimization_step()
        # Learning rate readout (legacy.Adam exposes .lr)
        try:
            current_learning_rate = float(optimizer.lr.numpy())
        except Exception:
            current_learning_rate = 0.05

        c = float(cost.numpy())
        cost_loop.append(c)

        rel = np.abs(np.array(alphaD_check_train) - np.array(alphaD[:,2])) / np.array(alphaD[:,2])

        if c < train_cost_min:
            train_cost_min   = c
            params_min_train = params.numpy().copy()
            min_train_iter   = i

        if i % num_check == 0:
            print(f"[seed {seed}] Iteration {i}, Cost: {c:.6e}, Rate: {current_learning_rate:.3e}")
            print(f"[seed {seed}] r (mean rel): {float(np.mean(rel)):.6e}")
            # Plot training alphaD comparison (same look as your script)
            plt.figure(i)
            plt.plot([j for j in range(len(train_set))], alphaD_check_train,
                     marker='.', label='Emu', ls='--', color='red')
            plt.plot([j for j in range(len(train_set))], np.array(alphaD[:,2]),
                     marker='.', label='QRPA', ls='--', color='black')
            plt.yscale('log')
            plt.title('iter = ' + str(i) + f' (seed={seed})')
            plt.legend()
            plt.savefig('it_out.png')  # you had this in your script
            plt.show()

    # Per-restart cost curve
    plt.figure()
    plt.plot(range(len(cost_loop)), cost_loop, label='Training set')
    plt.yscale('log')
    plt.xlabel('Number of training iterations', size=16)
    plt.ylabel('Cost function', size=16)
    plt.title(f'Cost history (seed={seed}) | best={train_cost_min:.3e} @ iter {min_train_iter}')
    plt.legend()
    plt.show()

    return train_cost_min, params_min_train, min_train_iter, cost_loop

# -----------------------------
# Multi-start driver (N_RESTARTS)
# -----------------------------
global_best_cost   = np.inf
global_best_params = None
global_best_meta   = {"seed": None, "iter": None}

for r in range(N_RESTARTS):
    seed = SEED0 + r
    print(f"\n========== RESTART {r+1}/{N_RESTARTS} (seed={seed}) ==========")
    best_cost, best_params, best_iter, _ = run_single_restart(seed)

    print(f"[seed {seed}] Best train cost in this restart: {best_cost:.6e} at iter {best_iter}")

    if best_cost < global_best_cost:
        global_best_cost   = best_cost
        global_best_params = best_params.copy()
        global_best_meta   = {"seed": seed, "iter": best_iter}

# -----------------------------
# Save ONLY the global best (same filename pattern you used)
# -----------------------------
print('Final (global best) train cost: {:.6e}'.format(global_best_cost))
print('Best at iteration: ', global_best_meta["iter"], ' seed: ', global_best_meta["seed"])

np.savetxt('params_'+str(n)+'_only_alphaD.txt', global_best_params)

with open('train_set.txt', "w") as f:
    for tup in train_set:
        f.write(",".join(map(str, tup)) + "\n")
