#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CAT plot for Alg 2 (alphaD-only), using YOUR timing/eval function:
    alphaD, alphaD_test, times = helper.plot_alphaD_simple(test_str, params, n, central_point)

Additions:
- Track training cost every iteration.
- After each n finishes training, plot "cost vs iterations" and save as PNG.

Outputs:
- CSV per n: CAT_alg2_n{n}.csv
- Cost curve per n: cost_curve_alg2_n{n}.png
- Final CAT figure: CAT_alg2_plotAlphaDsimple.png
"""

import os, re, random, csv
# Headless-safe; if you want interactive windows, comment the next two lines.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import helper

# =========================
# HARD-CODED CONFIG
# =========================
STRENGTH_DIR = "../dipoles_data_all/total_strength/"
SAVE_DIR     = "CAT_Alg2"         # output folder for CSVs and figure
TRAIN_SIZE   = 50                 # fixed # training points
N_LIST       = [4, 7, 10]         # model sizes to sweep
PARAM_SEED   = 123                # base seed for params / TF init (varied per n)
SPLIT_SEED   = 123                # seed for train/test split
ITERS        = 30_000             # training iterations per n
LR           = 5e-2               # learning rate
PRINT_EVERY  = 1_000

# rcParams: ticks in, readable sizes
plt.rcParams.update({
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "xtick.minor.visible": True, "ytick.minor.visible": True,
    "xtick.labelsize": 12, "ytick.labelsize": 12,
    "axes.labelsize": 14, "axes.titlesize": 15,
})

# (Optional) keep BLAS/TF threads sane & encourage determinism
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")
# To force CPU uncomment:
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# -----------------------------
# Helpers
# -----------------------------
def discover_points(strength_dir):
    pat = re.compile(r"strength_([0-9.]+)_([0-9.]+)\.out")
    combined_str = []
    for fname in os.listdir(strength_dir):
        m = pat.match(fname)
        if m:
            beta_s  = m.group(1)
            alpha_s = m.group(2)
            combined_str.append((alpha_s, beta_s))
    if not combined_str:
        raise RuntimeError("No points found. Check STRENGTH_DIR and filename pattern.")
    combined_f = [(float(a), float(b)) for (a, b) in combined_str]
    float_to_str = {(float(a), float(b)):(a, b) for (a, b) in combined_str}
    return combined_f, float_to_str

def build_fixed_split(combined_f, float_to_str, train_size, split_seed):
    # center + corners always in train; fill rest randomly up to train_size
    alpha_vals = [a for a, _ in combined_f]
    beta_vals  = [b for _, b in combined_f]
    Amin, Amax = min(alpha_vals), max(alpha_vals)
    Bmin, Bmax = min(beta_vals),  max(beta_vals)
    cx, cy = (Amin + Amax)/2.0, (Bmin + Bmax)/2.0

    def nearest_to(target, points):
        x0, y0 = target
        return min(points, key=lambda t: (t[0]-x0)**2 + (t[1]-y0)**2)

    central_f = nearest_to((cx, cy), combined_f)
    corners = [(Amin,Bmin),(Amin,Bmax),(Amax,Bmin),(Amax,Bmax)]
    corner_fs = [nearest_to(c, combined_f) for c in corners]

    fixed_f, seen = [], set()
    for p in [central_f] + corner_fs:
        if p not in seen:
            fixed_f.append(p); seen.add(p)

    random.seed(SPLIT_SEED)
    pool = list(set(combined_f) - set(fixed_f))
    random.shuffle(pool)

    need = max(0, train_size - len(fixed_f))
    if need > len(pool):
        raise RuntimeError(f"Requested TRAIN_SIZE={train_size} but dataset too small.")
    train_f = fixed_f + pool[:need]
    test_f  = list(set(combined_f) - set(train_f))

    def to_str_list(points_f):
        out = []
        for (a, b) in points_f:
            out.append(float_to_str.get((a, b), (format(a, '.12g'), format(b, '.12g'))))
        return out

    train_str = to_str_list(train_f)
    test_str  = to_str_list(test_f)
    central_point = float_to_str.get(central_f, (format(central_f[0], '.12g'), format(central_f[1], '.12g')))
    return train_str, test_str, central_point

def load_alphaD(points_str):
    """Return αD_true list using helper.data_table()."""
    _, alphaD_like = helper.data_table(points_str)
    alphaD_true = [float(a[2]) for a in np.vstack(alphaD_like)]
    return alphaD_true

def train_alg2(params, n, train_points_str, alphaD_train, central_point,
               lr=5e-2, iters=20_000, print_every=1_000):
    """Train Alg2 and return list of per-iteration costs."""
    opt = tf.keras.optimizers.legacy.Adam(learning_rate=lr)

    @tf.function
    def step(params, points_str, y_true):
        with tf.GradientTape() as tape:
            cost, _ = helper.cost_function_only_alphaD_batched(params, n, points_str, y_true, central_point)
        grads = tape.gradient(cost, params)
        if grads is None:
            grads = tf.zeros_like(params)
        opt.apply_gradients([(grads, params)])
        return cost

    costs = []
    for it in range(1, iters + 1):
        cost = step(params, train_points_str, alphaD_train)
        cval = float(cost.numpy())
        costs.append(cval)
        if (it % print_every == 0) or it in (1, iters):
            print(f"  [n={n}] iter {it:6d} | cost={cval:.6e}")
    return costs


# -----------------------------
# Main
# -----------------------------
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Discover & split (fixed train size, rest = test)
    combined_f, float_to_str = discover_points(STRENGTH_DIR)
    train_str, test_str, central_point = build_fixed_split(combined_f, float_to_str, TRAIN_SIZE, SPLIT_SEED)
    print(f"Train size = {len(train_str)} | Test size = {len(test_str)}")

    # Prepare true αD for training
    alphaD_train = load_alphaD(train_str)

    # Figure for CAT scatter
    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    colors  = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    markers = ["o", "s", "D", "^", "v", "x"]

    for i, n in enumerate(N_LIST):
        print(f"\n=== Alg 2 using plot_alphaD_simple | n = {n} ===")
        # Param vector size (Alg 2)
        nec_num_param = n + 3*int(n*(n+1)/2) + 1

        # Seed per-n (reproducible)
        np.random.seed(PARAM_SEED + 17*n)
        tf.random.set_seed(PARAM_SEED + 17*n)

        # Random init
        params = tf.Variable(np.random.uniform(0, 2, nec_num_param).astype(np.float32))

        # Train on TRAIN set and collect cost history
        costs = train_alg2(params, n, train_str, alphaD_train, central_point,
                           lr=LR, iters=ITERS, print_every=PRINT_EVERY)

        # ---- NEW: cost vs iterations plot for this n ----
        iters = np.arange(1, len(costs)+1)
        plt.figure(figsize=(6.2, 4.2))
        plt.plot(iters, costs, lw=1.2)
        plt.yscale("log")
        plt.xlabel("Iteration")
        plt.ylabel("Cost")
        plt.title(f"Alg 2 training cost — n = {n}")
        plt.grid(alpha=0.3, which="both")
        plt.tight_layout()
        out_cost = os.path.join(SAVE_DIR, f"cost_curve_alg2_n{n}.png")
        plt.savefig(out_cost, dpi=140)
        plt.show()
        print(f"  Saved: {out_cost}")

        # === Use YOUR function on the TEST set to get predictions & timing ===
        alphaD_pred, alphaD_true, times_s = helper.plot_alphaD_simple(test_str, params, n, central_point)

        alphaD_pred = np.asarray(alphaD_pred, dtype=float)
        alphaD_true = np.asarray(alphaD_true, dtype=float)
        times_ms    = 1000.0 * np.asarray(times_s, dtype=float)  # sec -> ms

        # Relative error
        rel_abs = np.abs((alphaD_pred - alphaD_true) / (alphaD_true + 1e-12))

        # Save CSV
        csv_path = os.path.join(SAVE_DIR, f"CAT_alg2_n{n}.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["alpha", "beta", "time_ms", "rel_abs", "alphaD_true", "alphaD_pred"])
            for (ab, t, r, y_true, y_pred) in zip(test_str, times_ms, rel_abs, alphaD_true, alphaD_pred):
                a_str, b_str = ab
                w.writerow([a_str, b_str, f"{t:.6f}", f"{r:.6e}", f"{y_true:.8f}", f"{y_pred:.8f}"])
        print(f"  Saved: {csv_path}")

        # Scatter + guide lines (means)
        ax.scatter(times_ms, rel_abs,
                   s=18, alpha=0.55,
                   color=colors[i % len(colors)],
                   marker=markers[i % len(markers)],
                   label=f"n = {n}")
        ax.axhline(np.mean(rel_abs),  color=colors[i % len(colors)], ls="-", lw=1.0, alpha=0.6)
        ax.axvline(np.mean(times_ms), color=colors[i % len(colors)], ls="-", lw=1.0, alpha=0.6)

        print(f"  n={n}: mean |rel|={np.mean(rel_abs):.3e}, "
              f"median |rel|={np.median(rel_abs):.3e}, "
              f"mean time={np.mean(times_ms):.3f} ms")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Inference time per sample (ms)")
    ax.set_ylabel("|relative error| in αD")
    ax.set_title("CAT plot — Alg 2 (via plot_alphaD_simple on test set)")
    ax.grid(alpha=0.25, which="both")
    ax.legend(frameon=False, ncol=2, fontsize=10)
    fig.tight_layout()
    out_png = os.path.join(SAVE_DIR, "CAT_alg2_plotAlphaDsimple.png")
    fig.savefig(out_png, dpi=150)
    print(f"\nFigure saved to: {out_png}")

if __name__ == "__main__":
    main()
