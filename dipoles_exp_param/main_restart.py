#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 27 19:48:17 2025

@author: anteravlic
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dipole emulator main (multi-start restarts + early stopping + per-restart plots)
"""

import os, re, sys, random as rn
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import Polynomial  # (kept in case your helper needs it)

import helper

# =========================================================
# Dataset construction (unchanged logic, minor tidying)
# =========================================================
rn.seed(142)
np.random.seed(30)

strength_dir = '../dipoles_data_all/total_strength/'
alphaD_dir   = '../dipoles_data_all/total_alphaD/'

pattern = re.compile(r'strength_([0-9.]+)_([0-9.]+)\.out')

formatted_alpha_values, formatted_beta_values, all_points = [], [], []
for fname in os.listdir(strength_dir):
    m = pattern.match(fname)
    if not m:
        continue
    beta_val  = m.group(1)
    alpha_val = m.group(2)
    all_points.append((alpha_val, beta_val))
    if (1.5 <= float(beta_val) <= 4.0) and (0.4 <= float(alpha_val) <= 1.8):
        formatted_alpha_values.append(alpha_val)
        formatted_beta_values.append(beta_val)

alpha = formatted_alpha_values
beta  = formatted_beta_values
combined = [(alpha[i], beta[i]) for i in range(len(alpha))]

# central point by bounding-box center
Amin = min(float(a) for a, b in combined); Amax = max(float(a) for a, b in combined)
Bmin = min(float(b) for a, b in combined); Bmax = max(float(b) for a, b in combined)
cx, cy = (Amin + Amax) / 2.0, (Bmin + Bmax) / 2.0
central_point = min(combined, key=lambda t: (float(t[0]) - cx)**2 + (float(t[1]) - cy)**2)

train_ratio, cv_ratio, test_ratio = 1.0, 0.0, 0.0
n_all = len(combined)
n_train = int(n_all * train_ratio)
n_cv    = int(n_all * cv_ratio)
n_test  = n_all - n_train - n_cv
train_set = combined[:n_train]
cv_set    = combined[n_train:n_train+n_cv]
test_set  = combined[n_train+n_cv:]

print('Central point:', central_point)
print('Sizes -> test:', len(test_set), 'cv:', len(cv_set), 'train:', len(train_set))

# =========================================================
# Model/setup (kept as in your script)
# =========================================================
n = 16
retain = 0.5
fold = 2.0

D, S1, S2 = helper.nec_mat(n)
print(D.shape, S1.shape, S2.shape)

strength, alphaD = helper.data_table(train_set)
alphaD = np.vstack(alphaD)
alphaD_list = [float(a[2]) for a in alphaD]

# Param count (kept from your code)
nec_num_param = 1 + 3*n + n + 4 * int(n * (n + 1) / 2) + 4
print('Number of parameters:', nec_num_param)

# Initialize some params more meaningfully (as you did)
def make_initial_guess(seed: int):
    rng = np.random.default_rng(seed)
    init = rng.uniform(0.0, 1.0, size=nec_num_param).astype(np.float32)
    init[0]   = fold
    init[-4:] = np.array([1, 1, 1, 1], dtype=np.float32)  # x1, x1, x2, x3 (as you wrote)
    return init

# Re-initialize M0 & v0 from central spectrum (unchanged)
keep = round(retain * n)
data = np.loadtxt(os.path.join(strength_dir, f"strength_{central_point[1]}_{central_point[0]}.out"))
omega = data[:, 0].astype(np.float32)
y     = data[:, 1].astype(np.float32)
omega_tf = tf.convert_to_tensor(omega, dtype=tf.float32)
y_tf     = tf.convert_to_tensor(y,     dtype=tf.float32)
eta_tf   = tf.convert_to_tensor(fold,  dtype=tf.float32)
E_hat, B_hat, y_hat = helper.fit_strength_with_tf_lorentzian(omega_tf, y_tf, keep, eta_tf, min_spacing=0.01)

plt.figure()
plt.plot(data[:,0], y_hat, label='fit')
plt.stem(E_hat, B_hat, use_line_collection=True, label='poles')
plt.plot(data[:,0], data[:,1], label='true')
plt.legend(); plt.title('Central spectrum fit'); plt.show()

def encode_init_with_fit(random_init):
    return helper.encode_initial_guess(random_init, E_hat, B_hat, n, retain).astype(np.float32)

# =========================================================
# Multi-start + Early stopping config (same semantics as beta)
# =========================================================
def set_all_seeds(seed: int):
    rn.seed(seed); np.random.seed(seed); tf.random.set_seed(seed)

# Multi-start
N_RESTARTS    = 30
SEED0         = 42
SAVE_DIR      = "runs_dipole"
os.makedirs(SAVE_DIR, exist_ok=True)

# Training limits & logging cadence
NUM_ITERATIONS = 30000
PRINT_EVERY    = 200        # check a bit more often

# Early stopping (two criteria)
PATIENCE_BEST        = 120      # checks without meaningful new best
MIN_DELTA_REL_BEST   = 2e-4     # relative improvement needed to reset best patience

PATIENCE_PLATEAU     = 120      # checks without MA improvement
MA_WINDOW            = 10       # MA length (in checks)
MIN_DELTA_REL        = 3e-4     # relative MA improvement threshold

WARMUP_CHECKS        = 12       # skip early-stop during first WARMUP_CHECKS logs
REQUIRE_BOTH_TO_STOP = True    # True => require (best stop AND plateau stop)

# Optional hard guard (don’t stop before X raw iterations)
MIN_ITERS_BEFORE_STOP = 30000       # set e.g. 20000 if you want a floor

def moving_average(arr, k):
    if len(arr) < k: return None
    return np.convolve(arr, np.ones(k)/k, mode='valid')

# =========================================================
# Training with restarts
# =========================================================
global_best_cost   = np.inf
global_best_params = None
global_best_meta   = {"seed": None, "iter": None}

for r in range(N_RESTARTS):
    seed = SEED0 + r
    print(f"\n========== RESTART {r+1}/{N_RESTARTS} (seed={seed}) ==========")
    set_all_seeds(seed)

    # params & optimizer
    init_vec   = make_initial_guess(seed)
    init_vec   = encode_init_with_fit(init_vec)
    params     = tf.Variable(init_vec, dtype=tf.float32)
    optimizer  = tf.keras.optimizers.legacy.Adam(learning_rate=0.01)

    # compile a fresh step bound to THIS optimizer/params
    @tf.function
    def optimization_step():
        with tf.GradientTape() as tape:
            cost, strength_cost, alphaD_cost, m1_cost, Lor, Lor_true, x, alphaD_train, B, eigenvalues = \
                helper.cost_function_batched_mixed(
                    params, n, train_set, strength, alphaD_list,
                    central_point, retain,
                    1, 2, 0, 875.0, 1e-8   # your original weights/args
                )
        grads = tape.gradient(cost, [params])
        optimizer.apply_gradients(zip(grads, [params]))
        return cost, strength_cost, alphaD_cost, m1_cost, Lor, Lor_true, x, alphaD_train, B, eigenvalues

    cost_history = []
    best_cost_this   = np.inf
    best_params_this = None
    best_iter_this   = -1

    # early stopping trackers
    checks_done    = 0
    no_improve_cnt = 0
    plateau_cnt    = 0
    last_ma        = None

    for i in range(NUM_ITERATIONS):
        (cost, strength_cost, alphaD_cost, m1_cost,
         Lor, Lor_true, x, alphaD_train, B, eigs) = optimization_step()
        c = float(cost.numpy())
        cost_history.append(c)

        # track best
        rel_impr_best = (best_cost_this - c) / max(abs(best_cost_this), 1e-12) if best_cost_this < np.inf else np.inf
        improved = (rel_impr_best >= MIN_DELTA_REL_BEST) or (best_cost_this == np.inf)
        if improved:
            best_cost_this   = c
            best_params_this = params.numpy().copy()
            best_iter_this   = i
            no_improve_cnt   = 0
        else:
            no_improve_cnt  += 1

        # logging & checks
        if (i % PRINT_EVERY) == 0:
            checks_done += 1
            try:
                lr = float(optimizer._decayed_lr(tf.float32).numpy())
            except Exception:
                lr = 0.01

            rel = np.abs(np.array(alphaD_train) - np.array(alphaD[:,2])) / np.array(alphaD[:,2])
            print(f"[seed {seed}] iter {i:6d} | cost={c:.6e} | lr={lr:.3e} | "
                  f"no_improve={no_improve_cnt} | mean αD rel={np.mean(rel):.3e} | "
                  f"str%={float(strength_cost/c):.3f} αD%={float(alphaD_cost/c):.3f} m1%={float(m1_cost/c):.3f}")

            # Warm-up: don’t consume patience during the first few logs
            if checks_done <= WARMUP_CHECKS or i < MIN_ITERS_BEFORE_STOP:
                no_improve_cnt = 0; plateau_cnt = 0; last_ma = None
            else:
                # plateau check via moving average on downsampled costs
                ds = cost_history[::PRINT_EVERY] if PRINT_EVERY > 0 else cost_history[:]
                ma = moving_average(ds, MA_WINDOW)
                if ma is not None:
                    current_ma = ma[-1]
                    if last_ma is None:  # first MA value after warm-up
                        last_ma = current_ma
                        plateau_cnt = 0
                    else:
                        rel_impr_ma = (last_ma - current_ma) / max(abs(last_ma), 1e-12)
                        if rel_impr_ma >= MIN_DELTA_REL:
                            plateau_cnt = 0
                            last_ma = current_ma
                        else:
                            plateau_cnt += 1

            # optional live diagnostic plot
            # (keep light; full curves saved at end of restart)
            # plt.figure(); ... ; plt.show()

        # stop rules
        stop_best    = (no_improve_cnt   >= PATIENCE_BEST)
        stop_plateau = (plateau_cnt      >= PATIENCE_PLATEAU)
        do_stop = (stop_best and stop_plateau) if REQUIRE_BOTH_TO_STOP else (stop_best or stop_plateau)
        if do_stop:
            reason = ("both rules" if REQUIRE_BOTH_TO_STOP else
                      "no best improvement" if stop_best else "moving-average plateau")
            print(f"[seed {seed}] Early stopping at iter {i} due to {reason}. "
                  f"best={best_cost_this:.6e} @ {best_iter_this}")
            break

    # ===== save per-restart artifacts =====
    run_dir = os.path.join(SAVE_DIR, f"seed_{seed}")
    os.makedirs(run_dir, exist_ok=True)

    np.savetxt(os.path.join(run_dir, f"params_n{n}_retain{retain}_seed{seed}.txt"), best_params_this)
    np.savetxt(os.path.join(run_dir, f"cost_history_seed{seed}.txt"), np.array(cost_history))
    with open(os.path.join(run_dir, "meta.txt"), "w") as f:
        f.write(f"best_cost={best_cost_this}\n")
        f.write(f"best_iter={best_iter_this}\n")
        f.write(f"seed={seed}\n")
        f.write(f"n={n}\nretain={retain}\n")
        f.write(f"PRINT_EVERY={PRINT_EVERY}\n")
        f.write(f"PATIENCE_BEST={PATIENCE_BEST}, MIN_DELTA_REL_BEST={MIN_DELTA_REL_BEST}\n")
        f.write(f"PATIENCE_PLATEAU={PATIENCE_PLATEAU}, MA_WINDOW={MA_WINDOW}, MIN_DELTA_REL={MIN_DELTA_REL}\n")

    # per-restart convergence plot (with optional MA overlay)
    plt.figure()
    plt.plot(range(len(cost_history)), cost_history, label=f'seed {seed}')
    # moving-average overlay (in raw-iter x-axis)
    ds = cost_history[::PRINT_EVERY] if PRINT_EVERY > 0 else cost_history[:]
    ma = moving_average(ds, MA_WINDOW)
    if ma is not None:
        xs = np.arange(len(ma)) * PRINT_EVERY + PRINT_EVERY*(MA_WINDOW-1)//2
        plt.plot(xs, ma, linestyle='--', label='moving avg')
    plt.yscale('log'); plt.xlabel('Iteration'); plt.ylabel('Cost')
    plt.title(f'Convergence (seed={seed}) | best={best_cost_this:.3e} @ iter {best_iter_this}')
    plt.legend()
    png_path = os.path.join(run_dir, f'cost_curve_seed{seed}.png')
    pdf_path = os.path.join(run_dir, f'cost_curve_seed{seed}.pdf')
    plt.savefig(png_path, bbox_inches='tight', dpi=150)
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.show()

    # update global best
    if best_cost_this < global_best_cost:
        global_best_cost   = best_cost_this
        global_best_params = best_params_this.copy()
        global_best_meta   = {"seed": seed, "iter": best_iter_this}

# save global best
np.savetxt(f'params_best_n{n}_retain{retain}.txt', global_best_params)
print(f"\n*** GLOBAL BEST *** cost={global_best_cost:.6e} "
      f"(seed={global_best_meta['seed']}, iter={global_best_meta['iter']})")
print(f"Saved per-seed runs in {SAVE_DIR}/ and best as params_best_n{n}_retain{retain}.txt")

# persist the train set for reference (unchanged)
with open("train_set.txt", "w") as f:
    for tup in train_set:
        f.write(",".join(map(str, tup)) + "\n")
