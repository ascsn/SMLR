#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 27 13:19:35 2025

@author: anteravlic
"""

import os, re, random
import numpy as np
import matplotlib.pyplot as plt
import helper
import tensorflow as tf
import csv

# -----------------------------
# 0) Config
# -----------------------------
strength_dir = '../dipoles_data_all/total_strength/'
pattern = re.compile(r'strength_([0-9.]+)_([0-9.]+)\.out')
SEED = 123
PARAM_SEED = 123
TEST_PICK = 50
N_TIERS = 10

# -----------------------------
# 1) Collect (alpha, beta) as STRINGS, but we’ll also keep float versions for geometry
# -----------------------------
combined_str = []  # list of (alpha_str, beta_str)
for fname in os.listdir(strength_dir):
    m = pattern.match(fname)
    if m:
        beta_s  = m.group(1)   # string
        alpha_s = m.group(2)   # string
        combined_str.append((alpha_s, beta_s))

if not combined_str:
    raise RuntimeError("No points found. Check 'strength_dir' and filename pattern.")

# Float views for geometry
combined_f = [(float(a), float(b)) for (a, b) in combined_str]

# Helper maps for easy conversion back to string tuples
def f2s_tuple(pt_f):
    a, b = pt_f
    # find exact match in original string list (avoid float->str drift):
    # since filenames provided exact strings, we reconstruct by formatting through the originals
    # Build a dict once for speed:
    return (format(a, ".12g"), format(b, ".12g"))

# Build a lookup from float to original string by nearest exact string in combined_str
# (we’ll use exact positions instead to avoid ambiguity)
float_to_str = { (float(a), float(b)):(a, b) for (a,b) in combined_str }

def as_str_tuple(pt_f):
    # Use exact lookup if possible; otherwise fall back to formatted strings
    return float_to_str.get((pt_f[0], pt_f[1]), (format(pt_f[0], ".12g"), format(pt_f[1], ".12g")))

# -----------------------------
# 2) Find center (nearest to box center) and “corner” samples
# -----------------------------
alpha_vals = [a for a, b in combined_f]
beta_vals  = [b for a, b in combined_f]
Amin, Amax = min(alpha_vals), max(alpha_vals)
Bmin, Bmax = min(beta_vals),  max(beta_vals)

cx, cy = (Amin + Amax) / 2.0, (Bmin + Bmax) / 2.0

def nearest_to(target, points):
    x0, y0 = target
    return min(points, key=lambda t: (t[0]-x0)**2 + (t[1]-y0)**2)

central_f = nearest_to((cx, cy), combined_f)
corners_target = [(Amin, Bmin), (Amin, Bmax), (Amax, Bmin), (Amax, Bmax)]
corner_fs = [nearest_to(c, combined_f) for c in corners_target]

# Deduplicate (center could coincide with a corner)
fixed_f = []
seen = set()
for p in [central_f] + corner_fs:
    if p not in seen:
        fixed_f.append(p)
        seen.add(p)

# -----------------------------
# 3) Build test set and nested training tiers
# -----------------------------
random.seed(SEED)
np.random.seed(PARAM_SEED)

all_set_f = set(combined_f)
fixed_set_f = set(fixed_f)

# Candidates for test: exclude fixed so fixed remain in training
candidates_f = list(all_set_f - fixed_set_f)

# Pick test points
TEST_SIZE = min(TEST_PICK, len(candidates_f))
test_set_f = set(random.sample(candidates_f, TEST_SIZE))

# Remaining pool for training (besides fixed)
train_pool_f = list((all_set_f - test_set_f) - fixed_set_f)
random.shuffle(train_pool_f)

# Create 5 nested tiers:
# sizes grow roughly linearly from ~20% of pool up to 100% of pool
if len(train_pool_f) == 0:
    sizes = [0] * N_TIERS
else:
    base = int(np.ceil(len(train_pool_f) / N_TIERS))  # first tier ~ 1/N
    sizes = np.linspace(base, len(train_pool_f), N_TIERS, dtype=int)
    sizes = np.maximum.accumulate(sizes)  # strictly non-decreasing

train_tiers_f = []
for s in sizes:
    add = train_pool_f[:s]
    tier_pts = list(fixed_f) + add
    train_tiers_f.append(tier_pts)

# Final sets (float)
test_points_f = list(test_set_f)

# -----------------------------
# 4) Plot
# -----------------------------
plt.figure(figsize=(7.6, 6.4))

# All points (background)
all_a = [a for a, b in combined_f]
all_b = [b for a, b in combined_f]
plt.scatter(all_a, all_b, s=12, color="#d0d0d0", label="All points (background)", zorder=1)

# Test set
ta = [a for a, b in test_points_f]
tb = [b for a, b in test_points_f]
plt.scatter(ta, tb, s=28, marker="x", label=f"Test ({len(test_points_f)})", zorder=3)

# Training tiers (nested). To avoid clutter, show all with low alpha; last tier darker.
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
for idx, tier in enumerate(train_tiers_f):
    a = [p[0] for p in tier]
    b = [p[1] for p in tier]
    alpha_val = 0.25 if idx < len(train_tiers_f) - 1 else 0.85
    plt.scatter(a, b, s=16, alpha=alpha_val, color=colors[idx % len(colors)],
                label=f"Train tier {idx+1} ({len(tier)})", zorder=2)

# Fixed points (center + corners)
fa = [p[0] for p in fixed_f]
fb = [p[1] for p in fixed_f]
plt.scatter(fa, fb, s=72, facecolors='none', edgecolors='k', linewidths=1.6,
            label=f"Fixed (center+corners) ({len(fixed_f)})", zorder=4)

plt.xlabel("alpha")
plt.ylabel("beta")
plt.title("Train/Test split with nested training tiers")
plt.legend(loc="best", fontsize=8)
plt.tight_layout()
plt.show()

# -----------------------------
# 5) Convert outputs back to STRING tuples in original format
# -----------------------------
def to_str_list(points_f_list):
    return [(float_to_str.get((a, b), (format(a, ".12g"), format(b, ".12g"))))
            for (a, b) in points_f_list]

fixed_points_str = to_str_list(fixed_f)
test_points_str  = to_str_list(test_points_f)
train_tiers_str  = [to_str_list(tier) for tier in train_tiers_f]

central_point_str = float_to_str.get(central_f, (format(central_f[0], ".12g"), format(central_f[1], ".12g")))
corner_points_str = [float_to_str.get(p, (format(p[0], ".12g"), format(p[1], ".12g"))) for p in corner_fs]

split_info = {
    "fixed_points": fixed_points_str,     # list[tuple[str,str]]
    "test_points":  test_points_str,      # list[tuple[str,str]]
    "train_tiers":  train_tiers_str,      # list[list[tuple[str,str]]], nested
    "center":       central_point_str,    # tuple[str,str]
    "corners":      corner_points_str,    # list[tuple[str,str]]
    "seed":         SEED,
    "sizes":        [len(t) for t in train_tiers_f],
}

print("Total points:", len(combined_str))
print("Fixed points (center+corners):", len(fixed_points_str))
print("Test points:", len(test_points_str))
print("Training tier sizes:", split_info["sizes"])

def alphaD_rel_errors(params, eval_points_str, strength_true, alphaD_true, *,
                      save_csv_path=None, tier_idx=None, label):
    """
    Returns dict with per-sample alphaD metrics and (optionally) saves a CSV.

    rel_abs = |pred - true| / true
    rel_signed = (pred - true) / true
    """

    total, alphaD_calc_tf = helper.cost_function_only_alphaD_batched(params, n, eval_points_str, alphaD_true, central_point)
    alphaD_pred = np.asarray(alphaD_calc_tf.numpy(), dtype=float)
    alphaD_true_np = np.asarray(alphaD_true, dtype=float)

    rel_signed = (alphaD_pred - alphaD_true_np) / (alphaD_true_np + 1e-12)
    rel_abs = np.abs(rel_signed)

    out = {
        "alphaD_pred": alphaD_pred,
        "alphaD_true": alphaD_true_np,
        "rel_signed": rel_signed,
        "rel_abs": rel_abs,
        "mean_rel_abs": float(np.mean(rel_abs)),
        "median_rel_abs": float(np.median(rel_abs)),
        "max_rel_abs": float(np.max(rel_abs)),
    }

    # Optional CSV dump
    if save_csv_path is not None:
        # Include the original (alpha,beta) string keys to keep your framework happy
        rows = []
        for (ab, y_true, y_pred, ra, rs) in zip(eval_points_str, alphaD_true_np, alphaD_pred, rel_abs, rel_signed):
            a_str, b_str = ab
            rows.append([a_str, b_str, y_true, y_pred, ra, rs])

        header = ["alpha", "beta", "alphaD_true", "alphaD_pred", "rel_abs", "rel_signed"]
        tier_tag = f"_tier{tier_idx}" if tier_idx is not None else ""
        csv_file = os.path.join(save_csv_path, f"alphaD_relerr_{label}{tier_tag}.csv")
        with open(csv_file, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)
        out["csv_path"] = csv_file

    return out



# -----------------------------
# Config
# -----------------------------
n       = 5
WARM_START = False                       # if False, re-init from scratch per tier
LR      = 5e-2
NUM_ITERS_TIER = 20000                   # iterations per tier
PRINT_EVERY    = 1000


EPS        = 1e-8

# folders and center file
strength_dir = '../dipoles_data_all/total_strength/'
alphaD_dir   = '../dipoles_data_all/total_alphaD/'

# from your split
train_tiers_str = split_info["train_tiers"]      # list[list[ (alpha_str, beta_str) ]]
test_points_str = split_info["test_points"]      # list[(alpha_str, beta_str)]
central_point   = split_info["center"]           # (alpha_str, beta_str)



# Build base random param vector
nec_num_param = n + 3*int(n * (n + 1) / 2) + 1 



random_initial_guess = np.random.uniform(0, 2, nec_num_param)

# -----------------------------
# Data loaders (string points -> tensors used by the cost)
# -----------------------------
def load_set(points_str):
    """
    points_str: list[(alpha_str, beta_str)]
    returns: strength_list, alphaD_true_list
    """
    strength_list, alphaD_like = helper.data_table(points_str)
    alphaD_true_list = [float(a[2]) for a in np.vstack(alphaD_like)]
    return strength_list, alphaD_true_list

strength_test, alphaD_test = load_set(test_points_str)

# -----------------------------
# Training loop across tiers
# -----------------------------
tier_results = []  # will store dicts per tier with costs etc.

# params variable (warm-start across tiers if desired)
params = tf.Variable(random_initial_guess, dtype=tf.float32)

optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=LR)

# Make sure `optimizer` is defined outside (e.g. optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=LR))

@tf.function
def train_step(params, train_set_pts, alphaD_true):
    # snapshot params BEFORE the update to measure how much they move
    old_params = tf.identity(params)

    with tf.GradientTape() as tape:
        cost, alphaD_train = helper.cost_function_only_alphaD_batched(params, n, train_set_pts, alphaD_true, central_point)

    # gradient and apply
    grads = tape.gradient(cost, params)
    # (optional) sanity: replace None with zeros
    if grads is None:
        grads = tf.zeros_like(params)
    # (optional) clip if needed
    # grads = tf.clip_by_norm(grads, 1e3)

    optimizer.apply_gradients([(grads, params)])

    # debug metrics
    grad_norm = tf.norm(grads)
    param_delta = tf.norm(params - old_params)

    return cost, alphaD_train


def evaluate_on(params, eval_points_str, alphaD_true):
    cost, alphaD_train = helper.cost_function_only_alphaD_batched(params, n, eval_points_str, alphaD_true, central_point)
    return float(cost.numpy())

for tier_idx, tier_points_str in enumerate(train_tiers_str, start=1):
    print(f"\n=== Tier {tier_idx} / {len(train_tiers_str)} | Train size: {len(tier_points_str)} ===")

    # (Re)load tier data
    strength_train, alphaD_train = load_set(tier_points_str)

    # Reinitialize or warm-start
    if not WARM_START:
        params.assign(random_initial_guess)

    # Train on this tier
    costs_hist = []
    for it in range(1, NUM_ITERS_TIER + 1):
        total, alphaD_calc = \
            train_step(params, tier_points_str, alphaD_train)
        costs_hist.append(float(total.numpy()))

        if it % PRINT_EVERY == 0 or it == 1 or it == NUM_ITERS_TIER:
            print(f"[Tier {tier_idx}] iter {it:6d} | total={total.numpy():.6e} ")

    # Evaluate on fixed test set
    test_total = evaluate_on(params, test_points_str, alphaD_test)
    
    # Evaluate on fixed train set
    train_total = evaluate_on(params, tier_points_str, alphaD_train)

    print(f"[Tier {tier_idx}] Test: total={test_total:.6e} ")
    
    # --- NEW: alphaD relative errors on test set ---
    save_dir = "alphaD_eval_Em2_"+str(PARAM_SEED)  # change if you want a different folder
    os.makedirs(save_dir, exist_ok=True)
    alphaD_metrics = alphaD_rel_errors(
        params,
        test_points_str,
        strength_test,
        alphaD_test,
        save_csv_path=save_dir,
        tier_idx=tier_idx,
        label="test",
    )
    
    
   # --- NEW: alphaD relative errors on train set ---
    os.makedirs(save_dir, exist_ok=True)
    alphaD_metrics_train = alphaD_rel_errors(
        params,
        tier_points_str,
        strength_test,
        alphaD_train,
        save_csv_path=save_dir,
        tier_idx=tier_idx,
        label="train",
    )
    print(f"[Tier {tier_idx}] αD rel-err (mean/median/max) = "
          f"{alphaD_metrics['mean_rel_abs']:.3e} / {alphaD_metrics['median_rel_abs']:.3e} / {alphaD_metrics['max_rel_abs']:.3e}")
    print(f"[Tier {tier_idx}] αD rel-err CSV:", alphaD_metrics.get("csv_path", "n/a"))
    
    # Plot per-sample rel abs (sorted) and a histogram
    rel_abs = alphaD_metrics["rel_abs"]
    idx_sorted = np.argsort(rel_abs)
    
    plt.figure(figsize=(5,4))

    plt.plot(np.arange(len(rel_abs)), rel_abs[idx_sorted], marker='.', lw=1)
    plt.ylabel("|rel error| in αD")
    plt.xlabel("samples (sorted)")
    plt.yscale('log')
    plt.title(f"Tier {tier_idx} — αD |rel| (mean {alphaD_metrics['mean_rel_abs']:.2e})")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"alphaD_relerr_test_tier{tier_idx}.png"), dpi=150)
    plt.show()
    
    # Store summary in tier_results
    tier_results.append({
        "tier": tier_idx,
        "train_size": len(tier_points_str),
        "train_cost_last": costs_hist[-1],
        "test_total": test_total,
        "alphaD_mean_rel_abs": alphaD_metrics["mean_rel_abs"],
        "alphaD_median_rel_abs": alphaD_metrics["median_rel_abs"],
        "alphaD_max_rel_abs": alphaD_metrics["max_rel_abs"],
        "hist": costs_hist,
    })


# -----------------------------
# Summary plots
# -----------------------------
plt.figure(figsize=(6,4.5))
plt.plot([r["train_size"] for r in tier_results],
         [r["alphaD_mean_rel_abs"] for r in tier_results],
         marker='o')
plt.xlabel("Training size (tier)")
plt.ylabel("Mean |rel error| in αD (test set)")
plt.title("αD mean |relative error| vs training tier")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.yscale('log')
plt.show()

# optional: print a compact table
print("\nTier summary:")
for r in tier_results:
    print(f" tier {r['tier']:>2d} | train {r['train_size']:>4d} | "
          f"test total {r['test_total']:.3e}  (S {r['test_strength']:.3e}, "
          f"m-1 {r['test_mminus1']:.3e}, m+1 {r['test_mplus1']:.3e})")

