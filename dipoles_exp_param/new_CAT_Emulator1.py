#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CAT plots for Emulator 1 with an inside-vs-outside split.

Figures produced (saved in SAVE_DIR):
1) CAT_em1_alphaD_inside_outside.png  — |relative error| in αD vs inference time (ms)
2) CAT_em1_strength_inside_outside.png — Strength loss Ls vs time (ms)

Assumes your helper provides:
  - helper.data_table(points_str)
  - helper.fit_strength_with_tf_lorentzian(...)
  - helper.encode_initial_guess(...)
  - helper.cost_function_batched_mixed(...)
  - helper.plot_alphaD(points_str, params, n, central_point, retain)
"""

import os, re, csv, time
# Headless-safe backend (comment if you want interactive windows)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.lines import Line2D
import helper

# =========================
# HARD-CODED CONFIG
# =========================
STRENGTH_DIR = "../dipoles_data_all/total_strength/"
SAVE_DIR     = "CAT_Em1_inside_outside"   # outputs (CSVs + figures)

N_LIST       = [7, 12, 16]   # model sizes
RETAIN       = 0.60
PARAM_SEED   = 123
FOLD         = 2.0

LR           = 1e-2
NUM_ITERS    = 20_000
PRINT_EVERY  = 1_000

# Loss weights for training (mixed loss)
W_STRENGTH   = 100.0
W_MMINUS1    = 5000.0
W_MPLUS1     = 0.0
M1_TARGET    = 875.0
EPS          = 1e-8

# ---- Inside region definition ----
# Option A (explicit box): uncomment and set if you want precise bounds
# EXPLICIT_BOX = {"alpha_min": 0.2, "alpha_max": 1.3, "beta_min": 0.2, "beta_max": 0.8}

# Option B (automatic inner box via quantiles if EXPLICIT_BOX is None)
EXPLICIT_BOX = None
INNER_QUANTILES = (0.2, 0.8)  # inner 60% box

# rcParams (ticks inward, readable)
plt.rcParams.update({
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "xtick.minor.visible": True, "ytick.minor.visible": True,
    "xtick.labelsize": 12, "ytick.labelsize": 12,
    "axes.labelsize": 14, "axes.titlesize": 15,
})

# (Optional) more deterministic runtime
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")
# To force CPU: os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# -----------------------------
# Utilities: discover & split
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

def nearest_to(target, points):
    x0, y0 = target
    return min(points, key=lambda t: (t[0]-x0)**2 + (t[1]-y0)**2)

def inside_outside_split(combined_f, float_to_str):
    # Determine inside box
    alpha_vals = np.array([a for a, _ in combined_f], dtype=float)
    beta_vals  = np.array([b for _, b in combined_f], dtype=float)

    if EXPLICIT_BOX is not None:
        a_min = EXPLICIT_BOX["alpha_min"]; a_max = EXPLICIT_BOX["alpha_max"]
        b_min = EXPLICIT_BOX["beta_min"];  b_max = EXPLICIT_BOX["beta_max"]
    else:
        qlo, qhi = INNER_QUANTILES
        a_min, a_max = np.quantile(alpha_vals, [qlo, qhi])
        b_min, b_max = np.quantile(beta_vals,  [qlo, qhi])

    def is_inside(p):
        a, b = p
        return (a_min <= a <= a_max) and (b_min <= b <= b_max)

    inside_f  = [p for p in combined_f if is_inside(p)]
    outside_f = [p for p in combined_f if not is_inside(p)]

    # Center (for init)
    Amin, Amax = float(alpha_vals.min()), float(alpha_vals.max())
    Bmin, Bmax = float(beta_vals.min()),  float(beta_vals.max())
    cx, cy = (Amin + Amax)/2.0, (Bmin + Bmax)/2.0
    central_f = nearest_to((cx, cy), combined_f)

    def to_str_list(points_f):
        out = []
        for (a, b) in points_f:
            out.append(float_to_str.get((a, b), (format(a, ".12g"), format(b, ".12g"))))
        return out

    inside_str  = to_str_list(inside_f)
    outside_str = to_str_list(outside_f)
    central_point = float_to_str.get(central_f, (format(central_f[0], ".12g"), format(central_f[1], ".12g")))
    box_info = dict(alpha_min=a_min, alpha_max=a_max, beta_min=b_min, beta_max=b_max)
    return inside_str, outside_str, central_point, box_info


# -----------------------------
# Data & training helpers
# -----------------------------
def load_set(points_str):
    """Return (strength_list, alphaD_true_list) from helper.data_table()."""
    strength_list, alphaD_like = helper.data_table(points_str)
    alphaD_true_list = [float(a[2]) for a in np.vstack(alphaD_like)]
    return strength_list, alphaD_true_list

def center_fit_initial_guess(central_point, n, retain, fold):
    """Fit Lorentzian at center and build retain-aware initial params."""
    keep = round(retain * n)
    cent_alpha, cent_beta = central_point
    cent_path = os.path.join(STRENGTH_DIR, f"strength_{cent_beta}_{cent_alpha}.out")
    data = np.loadtxt(cent_path)
    omega = data[:, 0].astype(np.float32)
    y     = data[:, 1].astype(np.float32)

    omega_tf = tf.convert_to_tensor(omega, dtype=tf.float32)
    y_tf     = tf.convert_to_tensor(y,     dtype=tf.float32)
    eta_tf   = tf.convert_to_tensor(FOLD,  dtype=tf.float32)

    E_hat, B_hat, _ = helper.fit_strength_with_tf_lorentzian(
        omega_tf, y_tf, keep, eta_tf, min_spacing=0.01
    )

    # Param vector size (Emulator 1):
    nec_num_param = 1 + 3*n + n + 4 * int(n*(n+1)/2) + 4

    base_random = np.random.uniform(-0.1, 0.1, nec_num_param).astype(np.float32)
    base_random[0]   = FOLD  # eta
    base_random[-4]  = 1.0   # x1
    base_random[-3]  = 1.0   # x2
    base_random[-2]  = 1.0   # x3
    base_random[-1]  = 1.0   # x4

    init = helper.encode_initial_guess(base_random, E_hat, B_hat, n, retain)
    return init

def train_em1(params, n, retain, train_points_str, strength_true, alphaD_true, central_point):
    opt = tf.keras.optimizers.legacy.Adam(learning_rate=LR)

    @tf.function
    def step(params, pts, s_true, a_true):
        with tf.GradientTape() as tape:
            total, Ls, Lm1, Lmplus1, _, _, _, _, _, _ = helper.cost_function_batched_mixed(
                params, n,
                pts, s_true, a_true,
                central_point, retain,
                w_strength=W_STRENGTH,
                w_mminus1=W_MMINUS1,
                w_mplus1=W_MPLUS1,
                m1_target=M1_TARGET,
                eps=EPS
            )
        grads = tape.gradient(total, params)
        if grads is None:
            grads = tf.zeros_like(params)
        opt.apply_gradients([(grads, params)])
        return total, Ls, Lm1, Lmplus1

    for it in range(1, NUM_ITERS + 1):
        total, Ls, Lm1, Lmplus1 = step(params, train_points_str, strength_true, alphaD_true)
        if (it % PRINT_EVERY == 0) or it in (1, NUM_ITERS):
            print(f"  iter {it:6d} | total={total.numpy():.6e} "
                  f"| S={Ls.numpy():.3e} | m-1={Lm1.numpy():.3e} | m+1={Lmplus1.numpy():.3e}")

def per_point_strength_cost_and_time(params, n, pts_str, strength_true_list, alphaD_true_list,
                                     central_point, retain):
    """
    For each single point, compute (Ls, elapsed_time_sec) by calling the mixed loss
    on a singleton batch and extracting the strength term Ls.
    """
    costs = []
    times = []
    for ab, G_true, aD_true in zip(pts_str, strength_true_list, alphaD_true_list):
        start = time.time()
        total, Ls, Lm1, Lmplus1, *_ = helper.cost_function_batched_mixed(
            params, n,
            [ab],                 # singleton
            [G_true],             # list-of-one strength array
            [aD_true],            # list-of-one alphaD value
            central_point, retain,
            w_strength=W_STRENGTH,
            w_mminus1=W_MMINUS1,
            w_mplus1=W_MPLUS1,
            m1_target=M1_TARGET,
            eps=EPS
        )
        end = time.time()
        costs.append(float(Ls.numpy()))         # strength cost for this sample
        times.append(end - start)               # secs for this sample
    return np.array(costs, dtype=float), np.array(times, dtype=float)


# -----------------------------
# Main
# -----------------------------
def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Discover & split (inside vs outside)
    combined_f, float_to_str = discover_points(STRENGTH_DIR)
    inside_str, outside_str, central_point, box = inside_outside_split(combined_f, float_to_str)
    print(f"Inside (train) points : {len(inside_str)}")
    print(f"Outside (test) points : {len(outside_str)}")
    print(f"Inside box: "
          f"alpha∈[{box['alpha_min']:.4g},{box['alpha_max']:.4g}], "
          f"beta∈[{box['beta_min']:.4g},{box['beta_max']:.4g}]")

    if len(inside_str) == 0 or len(outside_str) == 0:
        raise RuntimeError("Inside/outside split is degenerate. Adjust EXPLICIT_BOX or quantiles.")

    # Prepare data
    strength_in,  alphaD_in_true  = load_set(inside_str)
    strength_out, alphaD_out_true = load_set(outside_str)

    # --- Figure A: αD CAT (|relative error| vs time) ---
    fig_a, ax_a = plt.subplots(figsize=(8.0, 6.0))
    colors  = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    markers = ["o", "s", "D", "^", "v", "x"]

    # --- Figure B: Strength-cost CAT (Ls vs time) ---
    fig_b, ax_b = plt.subplots(figsize=(8.0, 6.0))

    for i, n in enumerate(N_LIST):
        print(f"\n=== Emulator 1 (n = {n}, retain = {RETAIN:.2f}) ===")

        # Reproducible seed per n
        np.random.seed(PARAM_SEED + 17*n)
        tf.random.set_seed(PARAM_SEED + 17*n)

        # Initial params from center fit
        params = tf.Variable(center_fit_initial_guess(central_point, n, RETAIN, FOLD), dtype=tf.float32)

        # Train on INSIDE
        train_em1(params, n, RETAIN, inside_str, strength_in, alphaD_in_true, central_point)

        # ===== αD CAT (inside + outside) via YOUR helper.plot_alphaD =====
        # OUTSIDE (test)
        aD_pred_out, aD_true_out, t_s_out = helper.plot_alphaD(outside_str, params, n, central_point, RETAIN)
        aD_pred_out = np.asarray(aD_pred_out, dtype=float)
        aD_true_out = np.asarray(aD_true_out, dtype=float)
        t_ms_out    = 1000.0 * np.asarray(t_s_out, dtype=float)
        rel_out     = np.abs((aD_pred_out - aD_true_out) / (aD_true_out + 1e-12))

        # INSIDE (train)
        aD_pred_in, aD_true_in, t_s_in = helper.plot_alphaD(inside_str, params, n, central_point, RETAIN)
        aD_pred_in = np.asarray(aD_pred_in, dtype=float)
        aD_true_in = np.asarray(aD_true_in, dtype=float)
        t_ms_in    = 1000.0 * np.asarray(t_s_in, dtype=float)
        rel_in     = np.abs((aD_pred_in - aD_true_in) / (aD_true_in + 1e-12))

        color = colors[i % len(colors)]
        # Outside = hollow
        ax_a.scatter(t_ms_out, rel_out, s=18, alpha=0.50, facecolors="none", edgecolors=color,
                     marker=markers[i % len(markers)], label=f"n={n} outside")
        # Inside = filled + black edge
        ax_a.scatter(t_ms_in,  rel_in,  s=28, alpha=0.80, facecolors=color, edgecolors="k", linewidths=0.7,
                     marker=markers[i % len(markers)], label=f"n={n} inside")
        # mean guide (outside)
        ax_a.axhline(np.mean(rel_out),  color=color, ls="-", lw=1.0, alpha=0.55)
        ax_a.axvline(np.mean(t_ms_out), color=color, ls="-", lw=1.0, alpha=0.55)

        # Save αD CSVs
        with open(os.path.join(SAVE_DIR, f"CAT_em1_alphaD_outside_n{n}.csv"), "w", newline="") as f:
            w = csv.writer(f); w.writerow(["alpha","beta","time_ms","rel_abs","alphaD_true","alphaD_pred"])
            for (ab, t, r, yt, yp) in zip(outside_str, t_ms_out, rel_out, aD_true_out, aD_pred_out):
                w.writerow([ab[0], ab[1], f"{t:.6f}", f"{r:.6e}", f"{yt:.8f}", f"{yp:.8f}"])
        with open(os.path.join(SAVE_DIR, f"CAT_em1_alphaD_inside_n{n}.csv"), "w", newline="") as f:
            w = csv.writer(f); w.writerow(["alpha","beta","time_ms","rel_abs","alphaD_true","alphaD_pred"])
            for (ab, t, r, yt, yp) in zip(inside_str, t_ms_in, rel_in, aD_true_in, aD_pred_in):
                w.writerow([ab[0], ab[1], f"{t:.6f}", f"{r:.6e}", f"{yt:.8f}", f"{yp:.8f}"])

        # ===== Strength-cost CAT (inside + outside): Ls vs time =====
        # Per-sample Ls + timing (singleton calls)
        Ls_out, t2_s_out = per_point_strength_cost_and_time(params, n, outside_str, strength_out, alphaD_out_true, central_point, RETAIN)
        Ls_in,  t2_s_in  = per_point_strength_cost_and_time(params, n, inside_str,  strength_in,  alphaD_in_true,  central_point, RETAIN)
        t2_ms_out = 1000.0 * t2_s_out
        t2_ms_in  = 1000.0 * t2_s_in

        # Outside = hollow; Inside = filled+black edge
        ax_b.scatter(t2_ms_out, Ls_out, s=18, alpha=0.50, facecolors="none", edgecolors=color,
                     marker=markers[i % len(markers)], label=f"n={n} outside")
        ax_b.scatter(t2_ms_in,  Ls_in,  s=28, alpha=0.80, facecolors=color, edgecolors="k", linewidths=0.7,
                     marker=markers[i % len(markers)], label=f"n={n} inside")
        # mean guide (outside)
        ax_b.axhline(np.mean(Ls_out),  color=color, ls="-", lw=1.0, alpha=0.55)
        ax_b.axvline(np.mean(t2_ms_out), color=color, ls="-", lw=1.0, alpha=0.55)

        # Save strength-cost CSVs
        with open(os.path.join(SAVE_DIR, f"CAT_em1_strength_outside_n{n}.csv"), "w", newline="") as f:
            w = csv.writer(f); w.writerow(["alpha","beta","time_ms","Ls"])
            for (ab, t, c) in zip(outside_str, t2_ms_out, Ls_out):
                w.writerow([ab[0], ab[1], f"{t:.6f}", f"{c:.6e}"])
        with open(os.path.join(SAVE_DIR, f"CAT_em1_strength_inside_n{n}.csv"), "w", newline="") as f:
            w = csv.writer(f); w.writerow(["alpha","beta","time_ms","Ls"])
            for (ab, t, c) in zip(inside_str, t2_ms_in, Ls_in):
                w.writerow([ab[0], ab[1], f"{t:.6f}", f"{c:.6e}"])

    # ===== Finalize Figure A (αD CAT) =====
    ax_a.set_xscale("log"); ax_a.set_yscale("log")
    ax_a.set_xlabel("Inference time per sample (ms)")
    ax_a.set_ylabel("|relative error| in αD")
    ax_a.set_title("CAT — Emulator 1 (inside=train • outside=test) — αD")

    region_handles = [
        Line2D([], [], marker="o", linestyle="None", color="k", markerfacecolor="white",
               markeredgecolor="k", label="outside (test)"),
        Line2D([], [], marker="o", linestyle="None", color="k", markerfacecolor="gray",
               markeredgecolor="k", label="inside (train)")
    ]
    leg_region = ax_a.legend(handles=region_handles, title="Region", loc="lower left", frameon=False)
    ax_a.add_artist(leg_region)

    size_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    size_handles = [Line2D([], [], marker="o", linestyle="None",
                           color=size_colors[i % len(size_colors)], label=f"n={n}")
                    for i, n in enumerate(N_LIST)]
    ax_a.legend(handles=size_handles, title="Model size", loc="upper right", frameon=False)

    ax_a.grid(alpha=0.25, which="both")
    fig_a.tight_layout()
    out_png_a = os.path.join(SAVE_DIR, "CAT_em1_alphaD_inside_outside.png")
    fig_a.savefig(out_png_a, dpi=160)
    print(f"\nFigure (αD) saved to: {out_png_a}")

    # ===== Finalize Figure B (Strength-cost CAT) =====
    ax_b.set_xscale("log"); ax_b.set_yscale("log")
    ax_b.set_xlabel("Per-sample evaluation time (ms)")
    ax_b.set_ylabel("Strength loss  Ls")
    ax_b.set_title("CAT — Emulator 1 (inside=train • outside=test) — strength cost")

    leg_region_b = ax_b.legend(handles=region_handles, title="Region", loc="lower left", frameon=False)
    ax_b.add_artist(leg_region_b)
    size_handles_b = [Line2D([], [], marker="o", linestyle="None",
                             color=size_colors[i % len(size_colors)], label=f"n={n}")
                      for i, n in enumerate(N_LIST)]
    ax_b.legend(handles=size_handles_b, title="Model size", loc="upper right", frameon=False)

    ax_b.grid(alpha=0.25, which="both")
    fig_b.tight_layout()
    out_png_b = os.path.join(SAVE_DIR, "CAT_em1_strength_inside_outside.png")
    fig_b.savefig(out_png_b, dpi=160)
    print(f"Figure (strength cost) saved to: {out_png_b}")


if __name__ == "__main__":
    main()
