#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 11:16:09 2025

@author: anteravlic
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train a single emulator (one value of n) with CLI args, cluster-safe.
Saves params, logs, and a quick initialization plot per run.

Usage (local):
  python train_pmm.py --n 12 --outdir runs/n12_r0.6

Recommended (SLURM array):
  sbatch submit_array.sbatch
"""

import os, re, sys, argparse, json, time, math
from pathlib import Path

# --- Headless plotting for clusters ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import tensorflow as tf

# If helper.py is one level up, make it importable
sys.path.insert(0, str(Path("..").resolve()))
import helper  # your project helper module

# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------
def set_seeds(seed: int):
    np.random.seed(seed)
    tf.random.set_seed(seed)

def ensure_dir(d):
    Path(d).mkdir(parents=True, exist_ok=True)

def write_lines(path, seq_of_tuples):
    with open(path, "w") as f:
        for tup in seq_of_tuples:
            f.write(",".join(map(str, tup)) + "\n")

def scan_param_grid(strength_dir: str,
                    alpha_min=0.4, alpha_max=1.8,
                    beta_min=1.5, beta_max=4.0):
    """
    Parse (alpha,beta) from filenames: strength_{beta}_{alpha}.out
    Keep the ORIGINAL string tokens so we can reconstruct exact filenames.
    Use floats only for filtering and centroid computation.
    Returns:
      combined: list[(alpha_raw, beta_raw)]
      central_point: (alpha_raw, beta_raw) closest to box center
      bounds: (Amin, Amax, Bmin, Bmax) as floats from the used data
    """
    pat = re.compile(r"^strength_([0-9.]+)_([0-9.]+)\.out$")
    combined = []  # list of (alpha_raw, beta_raw)

    try:
        files = os.listdir(strength_dir)
    except FileNotFoundError:
        raise FileNotFoundError(f"strength_dir not found: {strength_dir}")

    for fname in files:
        m = pat.match(fname)
        if not m:
            continue
        beta_raw  = m.group(1)  # keep exact tokens
        alpha_raw = m.group(2)
        beta_val  = float(beta_raw)
        alpha_val = float(alpha_raw)

        if (alpha_min <= alpha_val <= alpha_max) and (beta_min <= beta_val <= beta_max):
            combined.append((alpha_raw, beta_raw))

    if not combined:
        raise RuntimeError(f"No points found in {strength_dir} within alpha∈[{alpha_min},{alpha_max}], "
                           f"beta∈[{beta_min},{beta_max}]")

    A = np.array([float(a) for a, _ in combined], dtype=float)
    B = np.array([float(b) for _, b in combined], dtype=float)
    Amin, Amax = A.min(), A.max()
    Bmin, Bmax = B.min(), B.max()
    cx, cy = (Amin + Amax) / 2.0, (Bmin + Bmax) / 2.0

    central_point = min(
        combined,  # raw strings
        key=lambda t: (float(t[0]) - cx)**2 + (float(t[1]) - cy)**2
    )
    return combined, central_point, (Amin, Amax, Bmin, Bmax)

def resolve_strength_path(strength_dir: str, beta_token: str, alpha_token: str) -> str:
    """
    Try exact 'strength_{beta}_{alpha}.out' first.
    If missing (e.g., padding differences), search the directory for the closest numeric match.
    """
    candidate = os.path.join(strength_dir, f"strength_{beta_token}_{alpha_token}.out")
    if os.path.exists(candidate):
        return candidate

    pat = re.compile(r"^strength_([0-9.]+)_([0-9.]+)\.out$")
    target_b = float(beta_token)
    target_a = float(alpha_token)
    best = None
    best_err = float("inf")

    for fname in os.listdir(strength_dir):
        m = pat.match(fname)
        if not m:
            continue
        b = float(m.group(1))
        a = float(m.group(2))
        err = abs(b - target_b) + abs(a - target_a)
        if err < best_err:
            best_err = err
            best = fname

    if best is None:
        raise FileNotFoundError(
            f"{candidate} not found and no close match under {strength_dir}"
        )
    return os.path.join(strength_dir, best)

def init_guess_from_central(strength_dir: str, central_point, n: int, retain: float, fold: float):
    """
    Build an encoded initial guess using your existing helper utilities.
    central_point is (alpha_raw, beta_raw) as strings to preserve filename fidelity.
    """
    keep = int(round(retain * n))

    alpha_c, beta_c = central_point[0], central_point[1]
    fname = resolve_strength_path(strength_dir, beta_c, alpha_c)

    data = np.loadtxt(fname)
    omega = data[:, 0].astype(np.float32)
    y     = data[:, 1].astype(np.float32)

    omega_tf = tf.convert_to_tensor(omega, dtype=tf.float32)
    y_tf     = tf.convert_to_tensor(y,     dtype=tf.float32)
    eta_tf   = tf.convert_to_tensor(fold,  dtype=tf.float32)

    # Extract a few Lorentzians to seed parameters
    E_hat, B_hat, y_hat = helper.fit_strength_with_tf_lorentzian(
        omega_tf, y_tf, keep, eta_tf, min_spacing=0.01
    )

    # Parameter layout (as in your training script)
    nec_num_param = 1 + 3*n + n + 4 * int(n * (n + 1) / 2) + 4
    rnd = np.random.uniform(0, 1, nec_num_param).astype(np.float32)
    rnd[0]  = fold
    rnd[-4] = 1.0  # x1
    rnd[-3] = 1.0  # x1 (kept to mirror your previous code)
    rnd[-2] = 1.0  # x2
    rnd[-1] = 1.0  # x3

    encoded = helper.encode_initial_guess(rnd, E_hat, B_hat, n, retain)
    return encoded, (omega, y, y_hat.numpy(), E_hat.numpy(), B_hat.numpy())

def save_init_plot(outdir, omega, y, y_hat, E_hat, B_hat):
    plt.figure(figsize=(6, 4), dpi=140)
    plt.plot(omega, y, label="True", lw=2)
    plt.plot(omega, y_hat, "--", label="Init fit", lw=2)
    # Your prior scripts stem B_hat directly (not sqrt)
    markerline, stemlines, baseline = plt.stem(E_hat, B_hat, linefmt='C2-', markerfmt='C2o', basefmt=" ")
    plt.setp(stemlines, linewidth=1.5)
    plt.xlabel(r"$\omega$ (MeV)")
    plt.ylabel(r"$S$")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "init_fit.png"))
    plt.close()

# ------------------------------------------------------------
# Training
# ------------------------------------------------------------
def train_one(args):
    t0 = time.time()
    set_seeds(args.seed)
    ensure_dir(args.outdir)
    ensure_dir(os.path.join(args.outdir, "logs"))

    # Data scan
    combined, central_point, bounds = scan_param_grid(
        args.strength_dir, args.alpha_min, args.alpha_max, args.beta_min, args.beta_max
    )
    train_set = combined  # using all points as training set (as in your base script)
    write_lines(os.path.join(args.outdir, "train_set.txt"), train_set)

    # (Optional) sanity check of shapes:
    _D, _S1, _S2 = helper.nec_mat(args.n)

    # Build training tables using your helper
    strength, alphaD = helper.data_table(train_set)
    alphaD = np.vstack(alphaD)
    alphaD_list = [float(a[2]) for a in alphaD]

    # Initial guess from the central spectrum
    params0, (omega, y, y_hat, E_hat, B_hat) = init_guess_from_central(
        args.strength_dir, central_point, args.n, args.retain, args.fold
    )
    save_init_plot(args.outdir, omega, y, y_hat, E_hat, B_hat)

    # TF variable and optimizer
    params = tf.Variable(params0, dtype=tf.float32)
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=args.lr)

    # One optimization step
    @tf.function
    def optimization_step():
        with tf.GradientTape() as tape:
            # cost_function_batched_mixed signature maintained from your code:
            cost, strength_cost, alphaD_cost, m1_cost, \
            Lor, Lor_true, x, alphaD_train, B, eigenvalues = \
                helper.cost_function_batched_mixed(
                    params, args.n,
                    train_set,
                    strength,
                    alphaD_list,
                    central_point,
                    args.retain,
                    100.0, 200.0, 0.0, 875.0, 1e-8
                )
        grads = tape.gradient(cost, [params])
        optimizer.apply_gradients(zip(grads, [params]))
        return (cost, strength_cost, alphaD_cost, m1_cost,
                Lor, Lor_true, x, alphaD_train, B, eigenvalues)

    # Logging
    hist_path = os.path.join(args.outdir, "logs", "history.csv")
    with open(hist_path, "w") as f:
        f.write("iter,cost,strength_frac,alphaD_frac,m1_frac,mean_rel_alphaD\n")

    best_cost = math.inf
    best_iter = -1
    params_best = None

    for it in range(args.iters):
        (cost, strength_cost, alphaD_cost, m1_cost,
         Lor, Lor_true, x, alphaD_train, B, eigs) = optimization_step()

        # Rel. error on alphaD (monitoring)
        rel = np.abs(np.array(alphaD_train) - np.array(alphaD[:, 2])) / np.array(alphaD[:, 2])
        rel_mean = float(np.mean(rel))

        c = float(cost)
        if c < best_cost:
            best_cost = c
            best_iter = it
            params_best = params.numpy().copy()

        if (it % args.log_every) == 0:
            denom = c if c != 0.0 else 1.0
            with open(hist_path, "a") as f:
                f.write(f"{it},{c},{float(strength_cost)/denom},"
                        f"{float(alphaD_cost)/denom},{float(m1_cost)/denom},{rel_mean}\n")

        if (it % args.save_every) == 0 and it > 0:
            # checkpoint (not just best)
            np.savetxt(os.path.join(args.outdir, f"params_iter_{it}.txt"), params.numpy())

    # Final saves
    if params_best is None:
        params_best = params.numpy()
    final_path = os.path.join(args.outdir, f"params_{args.n}_{args.retain}.txt")
    np.savetxt(final_path, params_best)

    # Run summary
    with open(os.path.join(args.outdir, "run_summary.json"), "w") as f:
        json.dump({
            "n": args.n,
            "retain": args.retain,
            "iters": args.iters,
            "best_iter": best_iter,
            "best_cost": best_cost,
            "central_point": central_point,
            "bounds": {
                "alpha_min": bounds[0], "alpha_max": bounds[1],
                "beta_min": bounds[2],  "beta_max": bounds[3]
            },
            "fold": args.fold,
            "lr": args.lr,
            "strength_dir": os.path.abspath(args.strength_dir)
        }, f, indent=2)

    dur = time.time() - t0
    print(f"[DONE] n={args.n}, retain={args.retain}, best_iter={best_iter}, "
          f"best_cost={best_cost:.4e}, wall={dur/60:.1f} min")
    print(f"Saved best params to: {final_path}")

# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Train one emulator size (n)")
    p.add_argument("--n", type=int, required=True, help="Model dimension")
    p.add_argument("--retain", type=float, default=0.6, help="Retention ratio (0,1]")
    p.add_argument("--iters", type=int, default=30000, help="Training iterations")
    p.add_argument("--lr", type=float, default=1e-2, help="Adam learning rate")
    p.add_argument("--seed", type=int, default=30, help="RNG seed")
    p.add_argument("--fold", type=float, default=2.0, help="Base width parameter")
    p.add_argument("--alpha-min", dest="alpha_min", type=float, default=0.4)
    p.add_argument("--alpha-max", dest="alpha_max", type=float, default=1.8)
    p.add_argument("--beta-min",  dest="beta_min",  type=float, default=1.5)
    p.add_argument("--beta-max",  dest="beta_max",  type=float, default=4.0)
    p.add_argument("--strength-dir", type=str, default="../dipoles_data_all/total_strength/",
                   help="Folder with strength_{beta}_{alpha}.out used for init guess")
    p.add_argument("--outdir", type=str, required=True,
                   help="Output directory for this run")
    p.add_argument("--log-every", type=int, default=1000)
    p.add_argument("--save-every", type=int, default=5000)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train_one(args)
