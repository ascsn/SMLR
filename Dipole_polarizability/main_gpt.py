#!/usr/bin/env python3
from __future__ import annotations

import os
import json
import argparse
import random as rn
from dataclasses import asdict

import numpy as np
import tensorflow as tf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import helper_gpt


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Generic emulator trainer for strength + alphaD datasets with an arbitrary number of parameters."
    )

    # Data
    p.add_argument("--strength-dir", type=str, default="../dipoles_data_all/total_strength/",
                   help="Directory containing strength files.")
    p.add_argument("--alphaD-dir", type=str, default="../dipoles_data_all/total_alphaD/",
                   help="Directory containing alphaD files. Can be omitted if alphaD is computed from strength.")
    p.add_argument("--strength-regex", type=str,
                   default=r"strength_(?P<p2>[0-9.]+)_(?P<p1>[0-9.]+)\.out",
                   help="Regex used to parse parameters from strength filenames.")
    p.add_argument("--alphaD-regex", type=str, default=None,
                   help="Regex used to parse parameters from alphaD filenames. Defaults to strength-regex with alphaD_ prefix.")
    p.add_argument("--filter-ranges", type=str, default=None,
                   help='Optional JSON dict of filename-parameter filters, e.g. {"p1":[0.4,1.8],"p2":[1.5,4.0]}.')
    p.add_argument("--central-point", type=str, default=None,
                   help='Optional JSON list specifying the central point, e.g. "[1.0, 2.5]".')

    # Model structure
    p.add_argument("--n", type=int, default=10, help="Matrix size.")
    p.add_argument("--retain", type=float, default=0.5, help="Retained fraction of eigenmodes.")
    p.add_argument("--fold", type=float, default=2.0, help="Base Lorentzian width.")
    p.add_argument("--ansatz", choices=["linear", "quadratic", "linear_exp"], default="linear_exp",
                   help="Matrix ansatz family.")
    p.add_argument("--width-model", choices=["constant", "affine"], default="affine",
                   help="How eta depends on parameters.")
    p.add_argument("--no-vector-terms", action="store_true",
                   help="Disable affine dependence of the external-field vector on the input parameters.")

    # Loss weights
    p.add_argument("--w-strength", type=float, default=1.0)
    p.add_argument("--w-alphaD", type=float, default=2.0)
    p.add_argument("--w-m1", type=float, default=0.0)
    p.add_argument("--m1-target", type=float, default=875.0)

    # Optimization
    p.add_argument("--learning-rate", type=float, default=1e-2)
    p.add_argument("--n-restarts", type=int, default=10)
    p.add_argument("--seed0", type=int, default=42)
    p.add_argument("--num-iter", type=int, default=30000)
    p.add_argument("--print-every", type=int, default=1000)

    # Output
    p.add_argument("--plots", choices=["none", "save"], default="save")
    p.add_argument("--save-dir", type=str, default="runs_em1")
    return p.parse_args()


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------
def set_all_seeds(seed: int):
    rn.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)



def moving_average(arr, k):
    if len(arr) < k:
        return None
    return np.convolve(arr, np.ones(k) / k, mode="valid")



def ensure_same_omega_grid(strengths):
    ref = np.asarray(strengths[0])[:, 0]
    for i, s in enumerate(strengths[1:], start=1):
        if s.shape[0] != len(ref) or not np.allclose(s[:, 0], ref):
            raise ValueError(
                f"All strength files must share the same omega grid. File index 0 and {i} differ."
            )



def parse_optional_json(value):
    if value is None:
        return None
    return json.loads(value)


# -----------------------------------------------------------------------------
# Main training flow
# -----------------------------------------------------------------------------
def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    central_point = parse_optional_json(args.central_point)
    dataset = helper_gpt.load_dataset(
        strength_dir=args.strength_dir,
        alphaD_dir=args.alphaD_dir,
        strength_regex=args.strength_regex,
        alphaD_regex=args.alphaD_regex,
        filter_ranges=args.filter_ranges,
        central_point=central_point,
    )
    ensure_same_omega_grid(dataset.strengths)

    n_params = int(dataset.param_values.shape[1])
    config = helper_gpt.AnsatzConfig(
        n=args.n,
        n_params=n_params,
        ansatz=args.ansatz,
        width_model=args.width_model,
        use_vector_terms=not args.no_vector_terms,
    )
    config_summary = helper_gpt.summarize_config(config)

    print("Loaded dataset")
    print("  param names:", dataset.param_names)
    print("  samples:", len(dataset.strengths))
    print("  central point:", dataset.central_point.tolist())
    print("  config:", config_summary)

    param_values_tf = tf.convert_to_tensor(dataset.param_values, dtype=tf.float32)
    central_point_tf = tf.convert_to_tensor(dataset.central_point, dtype=tf.float32)

    # Central-spectrum fit used to initialize D and v0.
    central_idx = int(np.argmin(np.sum((dataset.param_values - dataset.central_point[None, :]) ** 2, axis=1)))
    central_strength = dataset.strengths[central_idx]
    omega = central_strength[:, 0].astype(np.float32)
    y = central_strength[:, 1].astype(np.float32)
    keep = max(1, round(args.retain * args.n))
    E_hat, B_hat, y_hat = helper_gpt.fit_strength_with_tf_lorentzian(omega, y, keep, args.fold, min_spacing=0.01)

    if args.plots == "save":
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(omega, y, label="true")
        ax.plot(omega, y_hat, label="central fit")
        ax.stem(E_hat, B_hat, linefmt='C2-', markerfmt='C2o', basefmt=' ', label="fitted poles")
        ax.set_title("Central spectrum fit")
        ax.legend()
        fig.savefig(os.path.join(args.save_dir, "central_spectrum_fit.png"), bbox_inches="tight", dpi=150)
        plt.close(fig)

    global_best_cost = np.inf
    global_best_params = None
    global_best_meta = {"seed": None, "iter": None}

    # Early stopping rules.
    PATIENCE_BEST = 120
    PATIENCE_PLATEAU = 120
    MA_WINDOW = 10
    MIN_DELTA_REL_BEST = 2e-4
    MIN_DELTA_REL = 3e-4
    WARMUP_LOGS = 12
    REQUIRE_BOTH_TO_STOP = True
    MIN_ITERATIONS = args.num_iter

    for r in range(args.n_restarts):
        seed = args.seed0 + r
        run_dir = os.path.join(args.save_dir, f"seed_{seed}")
        os.makedirs(run_dir, exist_ok=True)

        print(f"\n========== RESTART {r+1}/{args.n_restarts} (seed={seed}) ==========")
        set_all_seeds(seed)

        init_vec = helper_gpt.make_random_initial_guess(config, seed=seed, fold=args.fold)
        init_vec = helper_gpt.encode_initial_guess(init_vec, E_hat, B_hat, config, args.retain)
        params = tf.Variable(init_vec, dtype=tf.float32)
        optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)

        @tf.function
        def optimization_step():
            with tf.GradientTape() as tape:
                out = helper_gpt.cost_function_batched_generic(
                    params=params,
                    config=config,
                    param_values=param_values_tf,
                    strength_true=dataset.strengths,
                    alphaD_true=dataset.alphaD_values,
                    central_point=central_point_tf,
                    retain=args.retain,
                    w_strength=args.w_strength,
                    w_mminus1=args.w_alphaD,
                    w_mplus1=args.w_m1,
                    m1_target=args.m1_target,
                    eps=1e-8,
                )
            grads = tape.gradient(out[0], [params])
            optimizer.apply_gradients(zip(grads, [params]))
            return out

        cost_history = []
        best_cost_this = np.inf
        best_params_this = None
        best_iter_this = -1
        no_improve_cnt = 0
        plateau_cnt = 0
        last_ma = None
        logs_done = 0

        for i in range(args.num_iter):
            cost, strength_cost, alphaD_cost, m1_cost, Lor, Lor_true, x, alphaD_train, B, eigs = optimization_step()
            c = float(cost.numpy())
            cost_history.append(c)

            rel_impr_best = (best_cost_this - c) / max(abs(best_cost_this), 1e-12) if best_cost_this < np.inf else np.inf
            improved = (rel_impr_best >= MIN_DELTA_REL_BEST) or (best_cost_this == np.inf)
            if improved:
                best_cost_this = c
                best_params_this = params.numpy().copy()
                best_iter_this = i
                no_improve_cnt = 0
            else:
                no_improve_cnt += 1

            if (i % args.print_every) == 0:
                logs_done += 1
                alphaD_rel = np.mean(np.abs(np.array(alphaD_train.numpy()) - dataset.alphaD_values) /
                                     np.maximum(np.abs(dataset.alphaD_values), 1e-8))
                print(
                    f"[seed {seed}] iter {i:6d} | cost={c:.6e} | "
                    f"mean alphaD rel={alphaD_rel:.3e} | "
                    f"strength={float(strength_cost.numpy()):.3e} | "
                    f"alphaD={float(alphaD_cost.numpy()):.3e} | "
                    f"m1={float(m1_cost.numpy()):.3e}"
                )

                if logs_done <= WARMUP_LOGS:
                    no_improve_cnt = 0
                    plateau_cnt = 0
                    last_ma = None
                else:
                    ds = cost_history[::args.print_every] if args.print_every > 0 else cost_history[:]
                    ma = moving_average(ds, MA_WINDOW)
                    if ma is not None:
                        current_ma = ma[-1]
                        if last_ma is None:
                            last_ma = current_ma
                            plateau_cnt = 0
                        else:
                            rel_impr_ma = (last_ma - current_ma) / max(abs(last_ma), 1e-12)
                            if rel_impr_ma >= MIN_DELTA_REL:
                                plateau_cnt = 0
                                last_ma = current_ma
                            else:
                                plateau_cnt += 1

            stop_best = (no_improve_cnt >= PATIENCE_BEST)
            stop_plateau = (plateau_cnt >= PATIENCE_PLATEAU)
            do_stop = (stop_best and stop_plateau) if REQUIRE_BOTH_TO_STOP else (stop_best or stop_plateau)
            if do_stop and i >= MIN_ITERATIONS:
                print(f"[seed {seed}] Early stopping at iter {i}. best={best_cost_this:.6e} @ {best_iter_this}")
                break

        np.savetxt(os.path.join(run_dir, "best_params.txt"), best_params_this)
        np.savetxt(os.path.join(run_dir, "cost_history.txt"), np.array(cost_history))
        with open(os.path.join(run_dir, "meta.json"), "w") as f:
            json.dump({
                "seed": seed,
                "best_cost": best_cost_this,
                "best_iter": best_iter_this,
                "dataset_param_names": dataset.param_names,
                "central_point": dataset.central_point.tolist(),
                "config": config_summary,
            }, f, indent=2)

        if args.plots == "save":
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(cost_history)
            ax.set_yscale("log")
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Cost")
            ax.set_title(f"Convergence (seed={seed})")
            fig.savefig(os.path.join(run_dir, "cost_curve.png"), bbox_inches="tight", dpi=150)
            plt.close(fig)

            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(x.numpy(), Lor_true.numpy(), label="true")
            ax.plot(x.numpy(), Lor.numpy(), label="pred")
            ax.set_title(f"Final spectrum preview (seed={seed})")
            ax.legend()
            fig.savefig(os.path.join(run_dir, "final_spectrum_preview.png"), bbox_inches="tight", dpi=150)
            plt.close(fig)

        if best_cost_this < global_best_cost:
            global_best_cost = best_cost_this
            global_best_params = best_params_this.copy()
            global_best_meta = {"seed": seed, "iter": best_iter_this}

    np.savetxt(os.path.join(args.save_dir, "best_params_global.txt"), global_best_params)
    np.savetxt(os.path.join(args.save_dir, "train_param_values.txt"), dataset.param_values)
    with open(os.path.join(args.save_dir, "run_summary.json"), "w") as f:
        json.dump({
            "dataset_param_names": dataset.param_names,
            "n_samples": len(dataset.strengths),
            "central_point": dataset.central_point.tolist(),
            "config": config_summary,
            "global_best_cost": global_best_cost,
            "global_best_meta": global_best_meta,
            "args": vars(args),
        }, f, indent=2)

    print(f"\n*** GLOBAL BEST *** cost={global_best_cost:.6e} (seed={global_best_meta['seed']}, iter={global_best_meta['iter']})")
    print(f"Saved outputs in {args.save_dir}")


if __name__ == "__main__":
    main()
