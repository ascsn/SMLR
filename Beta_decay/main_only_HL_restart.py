#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 15:54:29 2025

@author: anteravlic
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Emulator 2 (half-lives only) with multi-start restarts (no argparse).
- Runs N_RESTARTS trainings with seeds SEED0..SEED0+N_RESTARTS-1
- Saves ONLY the global-best params to: params_{n}_only_HL.txt
- Keeps plotting and output structure as in your original script.
"""

import helper
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import Polynomial
import random as rn
import os
import re

# -----------------------------
# Fixed restart settings (no CLI parsing)
# -----------------------------
N_RESTARTS = 10
SEED0 = 42

def set_all_seeds(seed: int):
    rn.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

'''
Main data regarding the decay
'''
A = 80
Z = 28
g_A = 1.2
nucnam='Ni_80'

'''
Construction of phase space integrals
'''
poly = helper.fit_phase_space(0, Z, A, 15)
coeffs = Polynomial(poly).coef

# Coefficients for Horner eval (reverse order)
coefficients = tf.constant(coeffs[::-1], dtype=tf.float64)

# Range of x values for plotting
x_values = np.linspace(0.611, 15, 100)
x_tensor = tf.constant(x_values, dtype=tf.float64)

# Evaluate the polynomial at all x values
y_values = helper.evaluate_polynomial_tf(coefficients, x_tensor).numpy()

# Plotting (unchanged)
x = np.linspace(0.611, 15, 100)
plt.figure(figsize=(8, 5))
plt.plot(x_values, y_values)
plt.xlabel('x', fontsize=12)
plt.ylabel('f(x)', fontsize=12)
plt.title('Polynomial Plot', fontsize=14)
plt.axhline(0, color='gray', lw=0.8)
plt.axvline(0, color='gray', lw=0.8)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.yscale('log')

for i in range(len(x)):
    if (i % 10 == 0):
        plt.scatter(x[i], helper.phase_factor(0, Z, A, x[i]/helper.emass), color='red')

plt.show()

'''
Construct the data set
'''
rn.seed(20)

strength_dir = '../beta_decay_data_' + nucnam + '/'

# Pattern for strength files: lorm_{nucnam}_{beta}_{alpha}.out
pattern = re.compile(r'lorm_'+re.escape(nucnam)+r'_([0-9.]+)_([0-9.]+)\.out')

formatted_alpha_values = []
formatted_beta_values = []
all_points = []

for fname in os.listdir(strength_dir):
    match = pattern.match(fname)
    if match:
        beta_val = match.group(1)
        alpha_val = match.group(2)
        all_points.append((alpha_val, beta_val))

        if (0.1 <= float(beta_val) <= 0.9) and (0.2 <= float(alpha_val) <= 1.8):
            formatted_alpha_values.append(alpha_val)
            formatted_beta_values.append(beta_val)

# Combine into pairs
alpha = formatted_alpha_values
beta = formatted_beta_values
combined = [(alpha[i], beta[i]) for i in range(len(alpha))]

# Split ratios (unchanged logic)
train_ratio = 1.0
cv_ratio = 0.0
test_ratio = 0.0

n_total = len(combined)
n_train = int(n_total * train_ratio)
n_cv = int(n_total * cv_ratio)
n_test = n_total - n_train - n_cv  # Ensure all elements are used

train_set = combined[:n_train]
cv_set = combined[n_train:n_train + n_cv]
test_set = combined[n_train + n_cv:]

print(len(test_set), len(cv_set), len(train_set))

'''
Central data point (closest to centroid)
'''
combined_ar = np.array(combined, dtype=float)
centroid = combined_ar.mean(axis=0)

distances = np.linalg.norm(combined_ar - centroid, axis=1)
central_index = np.argmin(distances)
central_point = tuple(combined[central_index])
print('Central data point in train set:', central_point)

'''
Model dimension
'''
n = 9

D, S1, S2 = helper.nec_mat(n)
print(D.shape, S1.shape, S2.shape)

'''
Data table for half-lives (train set)
'''
print('Reading data ...')
HLs = helper.data_table_only_HL(train_set, coeffs, g_A, nucnam)
print('Data read ...')

nec_num_param = n + 2*int(n * (n + 1) / 2)
print('Number of parameters: ', nec_num_param)

params_shape = [D.shape, S1.shape, S2.shape]

# -----------------------------
# One restart training routine
# -----------------------------
def run_single_restart(seed: int):
    """Runs one training restart, returns (best_cost, best_params, best_iter, full_cost_history)."""
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

    # Training loop (unchanged semantics)
    num_iterations = 30000  # Total iterations
    num_check = 300         # Print/plot cadence
    cost_loop = []

    # Track best within this restart
    train_cost_min = np.inf
    params_min_train = None
    min_train_iter = -1

    for i in range(num_iterations):
        cost, HLs_train = optimization_step()
        current_learning_rate = optimizer._decayed_lr(tf.float32).numpy()
        cost_val = float(cost.numpy())
        cost_loop.append(cost_val)

        rel = np.abs(np.array(HLs_train) - np.array(HLs)) / np.array(HLs)

        if cost_val < train_cost_min:
            train_cost_min = cost_val
            params_min_train = params.numpy().copy()
            min_train_iter = i

        if i % num_check == 0:
            print(f"[seed {seed}] Iteration {i}, Cost: {cost_val:.6e}, Rate: {current_learning_rate:.3e}")
            print('[seed {}] Train cost: {:.6e}'.format(seed, cost_val))
            print('[seed {}] r: {:.6e}'.format(seed, float(np.mean(rel))))

            # Plot train HL comparison (same as your original)
            plt.figure(i)
            plt.plot(range(len(train_set)), HLs_train, marker='.', label='Emu', ls='--', color='red')
            plt.plot(range(len(train_set)), HLs,        marker='.', label='QRPA', ls='--', color='black')
            plt.yscale('log')
            plt.title('iter = ' + str(i) + f' (seed={seed})')
            plt.legend()
            plt.show()

    # Per-restart summary plot of the cost curve
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
global_best_cost = np.inf
global_best_params = None
global_best_meta = {"seed": None, "iter": None}

for r in range(N_RESTARTS):
    seed = SEED0 + r
    print(f"\n========== RESTART {r+1}/{N_RESTARTS} (seed={seed}) ==========")
    best_cost, best_params, best_iter, _ = run_single_restart(seed)

    print(f"[seed {seed}] Best train cost in this restart: {best_cost:.6e} at iter {best_iter}")

    if best_cost < global_best_cost:
        global_best_cost = best_cost
        global_best_params = best_params.copy()
        global_best_meta = {"seed": seed, "iter": best_iter}

# -----------------------------
# Save ONLY the global best (same filename pattern)
# -----------------------------
print('Final (global best) train cost: {:.6e}'.format(global_best_cost))
print('Best at iteration: ', global_best_meta["iter"], ' seed: ', global_best_meta["seed"])

# save parameters in a file
np.savetxt('params_'+str(n)+'_only_HL.txt', global_best_params)

# persist the train set for reference
with open("train_set.txt", "w") as f:
    for tup in train_set:
        f.write(",".join(map(str, tup)) + "\n")
