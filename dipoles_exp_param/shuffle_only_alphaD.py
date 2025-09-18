#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sequential PARAM_SEED runs (Alg 2: alphaD-only training)
"""

import os, re, random
import numpy as np
import matplotlib.pyplot as plt
import helper
import tensorflow as tf
import csv

# -----------------------------
# 0) Config (split is fixed by SEED)
# -----------------------------
strength_dir = '../dipoles_data_all/total_strength/'
pattern = re.compile(r'strength_([0-9.]+)_([0-9.]+)\.out')
SEED = 123                # used ONLY for building the split (train/test/tiers)
TEST_PICK = 50
N_TIERS = 10

# -----------------------------
# 1) Collect (alpha, beta)
# -----------------------------
combined_str = []
for fname in os.listdir(strength_dir):
    m = pattern.match(fname)
    if m:
        beta_s  = m.group(1)
        alpha_s = m.group(2)
        combined_str.append((alpha_s, beta_s))

if not combined_str:
    raise RuntimeError("No points found. Check 'strength_dir' and filename pattern.")

combined_f = [(float(a), float(b)) for (a, b) in combined_str]

# exact string lookup for any float pair that matches a discovered file
float_to_str = {(float(a), float(b)):(a, b) for (a,b) in combined_str}

# -----------------------------
# 2) Center & corners (in float)
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

fixed_f = []
seen = set()
for p in [central_f] + corner_fs:
    if p not in seen:
        fixed_f.append(p); seen.add(p)

# -----------------------------
# 3) Build test set + nested training tiers (SPLIT FIXED by SEED)
# -----------------------------
random.seed(SEED)

all_set_f   = set(combined_f)
fixed_set_f = set(fixed_f)

candidates_f = list(all_set_f - fixed_set_f)   # keep fixed in train only

TEST_SIZE = min(TEST_PICK, len(candidates_f))
test_set_f = set(random.sample(candidates_f, TEST_SIZE))

train_pool_f = list((all_set_f - test_set_f) - fixed_set_f)
random.shuffle(train_pool_f)

if len(train_pool_f) == 0:
    sizes = [0] * N_TIERS
else:
    base = int(np.ceil(len(train_pool_f) / N_TIERS))
    sizes = np.linspace(base, len(train_pool_f), N_TIERS, dtype=int)
    sizes = np.maximum.accumulate(sizes)

train_tiers_f = []
for s in sizes:
    add = train_pool_f[:s]
    tier_pts = list(fixed_f) + add
    train_tiers_f.append(tier_pts)

test_points_f = list(test_set_f)

# -----------------------------
# 4) Plot split
# -----------------------------
plt.figure(figsize=(7.6, 6.4))
all_a = [a for a, b in combined_f]
all_b = [b for a, b in combined_f]
plt.scatter(all_a, all_b, s=12, color="#d0d0d0", label="All points", zorder=1)

ta = [a for a, b in test_points_f]
tb = [b for a, b in test_points_f]
plt.scatter(ta, tb, s=28, marker="x", label=f"Test ({len(test_points_f)})", zorder=3)

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
for idx, tier in enumerate(train_tiers_f):
    a = [p[0] for p in tier]; b = [p[1] for p in tier]
    alpha_val = 0.25 if idx < len(train_tiers_f) - 1 else 0.85
    plt.scatter(a, b, s=16, alpha=alpha_val, color=colors[idx % len(colors)],
                label=f"Train tier {idx+1} ({len(tier)})", zorder=2)

fa = [p[0] for p in fixed_f]; fb = [p[1] for p in fixed_f]
plt.scatter(fa, fb, s=72, facecolors='none', edgecolors='k', linewidths=1.6,
            label=f"Fixed (center+corners) ({len(fixed_f)})", zorder=4)

plt.xlabel("alpha"); plt.ylabel("beta")
plt.title("Train/Test split with nested training tiers")
plt.legend(loc="best", fontsize=8)
plt.tight_layout(); plt.show()

# -----------------------------
# 5) Convert float → original string tuples & split_info
# -----------------------------
def to_str_list(points_f_list):
    out = []
    for (a, b) in points_f_list:
        out.append(float_to_str.get((a, b), (format(a, ".12g"), format(b, ".12g"))))
    return out

fixed_points_str = to_str_list(fixed_f)
test_points_str  = to_str_list(test_points_f)
train_tiers_str  = [to_str_list(tier) for tier in train_tiers_f]

central_point_str = float_to_str.get(central_f, (format(central_f[0], ".12g"), format(central_f[1], ".12g")))
corner_points_str = [float_to_str.get(p, (format(p[0], ".12g"), format(p[1], ".12g"))) for p in corner_fs]

split_info = {
    "fixed_points": fixed_points_str,
    "test_points":  test_points_str,
    "train_tiers":  train_tiers_str,
    "center":       central_point_str,
    "corners":      corner_points_str,
    "seed":         SEED,
    "sizes":        [len(t) for t in train_tiers_f],
}

print("Total points:", len(combined_str))
print("Fixed points (center+corners):", len(fixed_points_str))
print("Test points:", len(test_points_str))
print("Training tier sizes:", split_info["sizes"])

# αD error helper (uses your alphaD-only batched cost)
def alphaD_rel_errors(params, eval_points_str, alphaD_true, *,
                      save_csv_path=None, tier_idx=None, label="set"):
    cost, alphaD_calc_tf = helper.cost_function_only_alphaD_batched(
        params, n, eval_points_str, alphaD_true, central_point
    )
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

    if save_csv_path is not None:
        rows = []
        for (ab, y_true, y_pred, ra, rs) in zip(eval_points_str, alphaD_true_np, alphaD_pred, rel_abs, rel_signed):
            a_str, b_str = ab
            rows.append([a_str, b_str, y_true, y_pred, ra, rs])
        header = ["alpha", "beta", "alphaD_true", "alphaD_pred", "rel_abs", "rel_signed"]
        tier_tag = f"_tier{tier_idx}" if tier_idx is not None else ""
        csv_file = os.path.join(save_csv_path, f"alphaD_relerr_{label}{tier_tag}.csv")
        with open(csv_file, "w", newline="") as f:
            w = csv.writer(f); w.writerow(header); w.writerows(rows)
        out["csv_path"] = csv_file
    return out

# -----------------------------
# Training config (constant across seeds)
# -----------------------------
n            = 5
WARM_START   = False      # if False, re-init from scratch per tier
LR           = 5e-2
NUM_ITERS_TIER = 20000
PRINT_EVERY    = 1000
EPS          = 1e-8

# from split
train_tiers_str = split_info["train_tiers"]
test_points_str = split_info["test_points"]
central_point   = split_info["center"]

# loader
def load_set(points_str):
    strength_list, alphaD_like = helper.data_table(points_str)
    alphaD_true_list = [float(a[2]) for a in np.vstack(alphaD_like)]
    return strength_list, alphaD_true_list

# fixed test set (once)
_, alphaD_test = load_set(test_points_str)

# -----------------------------
# 6) Sequential runs over PARAM_SEEDs
# -----------------------------
PARAM_SEEDS = [123, 321, 777]   # edit this list as you like

all_runs_summary = []

for PARAM_SEED in PARAM_SEEDS:
    print(f"\n================ PARAM_SEED = {PARAM_SEED} ================\n")

    # Re-seed *only* the parameter-related RNGs for this run
    np.random.seed(PARAM_SEED)
    tf.random.set_seed(PARAM_SEED)

    # Param vector size (Alg 2): n + 3*n(n+1)/2 + 1
    nec_num_param = n + 3*int(n * (n + 1) / 2) + 1
    random_initial_guess = np.random.uniform(0, 2, nec_num_param).astype(np.float32)

    # Fresh params & optimizer per seed
    params = tf.Variable(random_initial_guess, dtype=tf.float32)
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=LR)

    @tf.function
    def train_step(params, train_set_pts, alphaD_true):
        with tf.GradientTape() as tape:
            cost, alphaD_train = helper.cost_function_only_alphaD_batched(
                params, n, train_set_pts, alphaD_true, central_point
            )
        grads = tape.gradient(cost, params)
        if grads is None:
            grads = tf.zeros_like(params)
        optimizer.apply_gradients([(grads, params)])
        return cost, alphaD_train

    def evaluate_on(params, eval_points_str, alphaD_true):
        cost, _ = helper.cost_function_only_alphaD_batched(
            params, n, eval_points_str, alphaD_true, central_point
        )
        return float(cost.numpy())

    # Seed-specific output dir
    save_dir = f"alphaD_eval_Em2_{PARAM_SEED}"
    os.makedirs(save_dir, exist_ok=True)

    tier_results = []

    for tier_idx, tier_points_str in enumerate(train_tiers_str, start=1):
        print(f"\n=== [seed {PARAM_SEED}] Tier {tier_idx} / {len(train_tiers_str)} "
              f"| Train size: {len(tier_points_str)} ===")

        _, alphaD_train = load_set(tier_points_str)

        if not WARM_START:
            params.assign(random_initial_guess)

        costs_hist = []
        for it in range(1, NUM_ITERS_TIER + 1):
            total, alphaD_calc = train_step(params, tier_points_str, alphaD_train)
            costs_hist.append(float(total.numpy()))

            if it % PRINT_EVERY == 0 or it == 1 or it == NUM_ITERS_TIER:
                print(f"[Tier {tier_idx}] iter {it:6d} | total={total.numpy():.6e}")

        # Evaluate on test and train
        test_total  = evaluate_on(params, test_points_str, alphaD_test)
        train_total = evaluate_on(params, tier_points_str, alphaD_train)
        print(f"[Tier {tier_idx}] Test: total={test_total:.6e}")

        # αD relative errors (test)
        alphaD_metrics = alphaD_rel_errors(
            params, test_points_str, alphaD_test,
            save_csv_path=save_dir, tier_idx=tier_idx, label="test",
        )
        # αD relative errors (train)
        alphaD_metrics_train = alphaD_rel_errors(
            params, tier_points_str, alphaD_train,
            save_csv_path=save_dir, tier_idx=tier_idx, label="train",
        )

        print(f"[Tier {tier_idx}] αD rel-err (mean/median/max) = "
              f"{alphaD_metrics['mean_rel_abs']:.3e} / "
              f"{alphaD_metrics['median_rel_abs']:.3e} / "
              f"{alphaD_metrics['max_rel_abs']:.3e}")
        print(f"[Tier {tier_idx}] αD rel-err CSV:", alphaD_metrics.get("csv_path", "n/a"))

        # Plot sorted |rel error| on test
        rel_abs = alphaD_metrics["rel_abs"]
        idx_sorted = np.argsort(rel_abs)
        plt.figure(figsize=(5,4))
        plt.plot(np.arange(len(rel_abs)), rel_abs[idx_sorted], marker='.', lw=1)
        plt.ylabel("|rel error| in αD")
        plt.xlabel("samples (sorted)")
        plt.yscale('log')
        plt.title(f"Seed {PARAM_SEED} — Tier {tier_idx} — αD |rel| (mean {alphaD_metrics['mean_rel_abs']:.2e})")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"alphaD_relerr_test_tier{tier_idx}.png"), dpi=150)
        plt.close()

        tier_results.append({
            "tier": tier_idx,
            "train_size": len(tier_points_str),
            "train_cost_last": costs_hist[-1],
            "test_total": test_total,
            "alphaD_mean_rel_abs": alphaD_metrics["mean_rel_abs"],
            "alphaD_median_rel_abs": alphaD_metrics["median_rel_abs"],
            "alphaD_max_rel_abs": alphaD_metrics["max_rel_abs"],
        })

    all_runs_summary.append({"seed": PARAM_SEED, "tiers": tier_results})

# -----------------------------
# 7) Summary plots per last run (optional)
# -----------------------------
last_run = all_runs_summary[-1]
plt.figure(figsize=(6,4.5))
plt.plot([r["train_size"] for r in last_run["tiers"]],
         [r["alphaD_mean_rel_abs"] for r in last_run["tiers"]],
         marker='o')
plt.xlabel("Training size (tier)")
plt.ylabel("Mean |rel error| in αD (test set)")
plt.title(f"αD mean |relative error| vs training tier (seed {last_run['seed']})")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.yscale('log')
plt.show()

# -----------------------------
# 8) Cross-seed summary (last tier of each seed)
# -----------------------------
print("\n=== Cross-seed summary (last tier) ===")
for run in all_runs_summary:
    r_last = run["tiers"][-1]
    print(f" seed {run['seed']:>4d} | train {r_last['train_size']:>4d} | "
          f"test total {r_last['test_total']:.3e} | "
          f"αD mean |rel| {r_last['alphaD_mean_rel_abs']:.2e} "
          f"(median {r_last['alphaD_median_rel_abs']:.2e})")
