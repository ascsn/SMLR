#!/usr/bin/env python3
from __future__ import annotations

import os
import json
import argparse
import random as rn

import numpy as np
import tensorflow as tf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import src.helper_gpt as helper_gpt


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Generic alphaD-only emulator trainer for datasets with an arbitrary number of parameters."
    )

    # Data
    p.add_argument("--strength-dir", type=str, default="../dipoles_data_all/total_strength/",
                   help="Directory containing strength files, used to define the parameter grid.")
    p.add_argument("--alphaD-dir", type=str, default="../dipoles_data_all/total_alphaD/",
                   help="Directory containing alphaD files.")
    p.add_argument("--strength-regex", type=str,
                   default=r"strength_(?P<p2>[0-9.]+)_(?P<p1>[0-9.]+)\.out",
                   help="Regex used to parse parameters from strength filenames.")
    p.add_argument("--alphaD-regex", type=str, default=None,
                   help="Regex used to parse parameters from alphaD filenames. Defaults to strength-regex with alphaD_ prefix.")
    p.add_argument("--filter-ranges", type=str, default=r'{"p1":[0.4,1.8],"p2":[1.5,4.0]}',
                   help='Optional JSON dict of filename-parameter filters, e.g. {"p1":[0.4,1.8],"p2":[1.5,4.0]}.')
    p.add_argument("--central-point", type=str, default=None,
                   help='Optional JSON list specifying the central point, e.g. "[1.0, 2.5]".')

    # Model structure
    p.add_argument("--n", type=int, default=10, help="Matrix size.")
    p.add_argument("--ansatz", choices=["linear", "quadratic", "linear_exp", "paper_dipole"], default="linear_exp",
                   help="Matrix ansatz family.")
    p.add_argument("--alphaD-mode", choices=["mid_eigenvalue", "sum_inverse_positive"], default="mid_eigenvalue",
                   help="How the emulator converts eigenvalues into the alphaD prediction.")
    p.add_argument("--l2", type=float, default=0.0,
                   help="Optional L2 regularization on the trainable parameters.")
    p.add_argument("--init-scale", type=float, default=2.0,
                   help="Uniform initialization range [0, init-scale].")

    # Optimization
    p.add_argument("--learning-rate", type=float, default=5e-2)
    p.add_argument("--n-restarts", type=int, default=5)
    p.add_argument("--seed0", type=int, default=42)
    p.add_argument("--num-iter", type=int, default=30000)
    p.add_argument("--print-every", type=int, default=1000)

    # Output
    p.add_argument("--plots", choices=["none", "save"], default="save")
    p.add_argument("--save-dir", type=str, default="runs_em2")
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



def parse_optional_json(value):
    if value is None:
        return None
    return json.loads(value)



def save_alphaD_preview(out_path: str, truth: np.ndarray, pred: np.ndarray, title: str):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(range(len(truth)), truth, marker='.', ls='--', label='true')
    ax.plot(range(len(truth)), pred, marker='.', ls='--', label='pred')
    ax.set_yscale('log')
    ax.set_title(title)
    ax.legend()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main training flow
# -----------------------------------------------------------------------------
def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    MIN_ITERATIONS = 20000
    if args.num_iter < MIN_ITERATIONS:
        raise ValueError(f"--num-iter ({args.num_iter}) must be >= {MIN_ITERATIONS}.")

    central_point = parse_optional_json(args.central_point)
    dataset = helper_gpt.load_generic_alphaD_dataset(
        strength_dir=args.strength_dir,
        alphaD_dir=args.alphaD_dir,
        strength_regex=args.strength_regex,
        alphaD_regex=args.alphaD_regex,
        filter_ranges=args.filter_ranges,
        central_point=central_point,
    )

    n_params = int(dataset.param_values.shape[1])
    config = helper_gpt.AlphaDOnlyConfig(
        n=args.n,
        n_params=n_params,
        ansatz=args.ansatz,
        alphaD_mode=args.alphaD_mode,
    )
    config_summary = helper_gpt.summarize_alphaD_only_config(config)
    n_trainable = helper_gpt.count_alphaD_only_parameters(config)

    print("Loaded dataset")
    print("  param names:", dataset.param_names)
    print("  samples:", len(dataset.alphaD_values))
    print("  central point:", dataset.central_point.tolist())
    print("  config:", config_summary)

    objective_fn = helper_gpt.make_alphaD_only_loss_fn_generic(
        n=args.n,
        param_values=dataset.param_values,
        alphaD_true=dataset.alphaD_values,
        central_point=dataset.central_point,
        ansatz=args.ansatz,
        alphaD_mode=args.alphaD_mode,
        l2_reg=args.l2,
    )

    global_best_cost = np.inf
    global_best_params = None
    global_best_pred = None
    global_best_meta = {"seed": None, "iter": None}

    PATIENCE_BEST = 120
    PATIENCE_PLATEAU = 120
    MA_WINDOW = 10
    MIN_DELTA_REL_BEST = 2e-4
    MIN_DELTA_REL = 3e-4
    WARMUP_LOGS = 12
    REQUIRE_BOTH_TO_STOP = True

    for r in range(args.n_restarts):
        seed = args.seed0 + r
        run_dir = os.path.join(args.save_dir, f"seed_{seed}")
        os.makedirs(run_dir, exist_ok=True)

        print(f"\n========== RESTART {r+1}/{args.n_restarts} (seed={seed}) ==========")
        set_all_seeds(seed)

        init_vec = np.random.uniform(0.0, args.init_scale, n_trainable).astype(np.float32)
        params = tf.Variable(init_vec, dtype=tf.float32)
        optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)

        @tf.function
        def optimization_step():
            with tf.GradientTape() as tape:
                out = objective_fn(params)
            grads = tape.gradient(out[0], [params])
            optimizer.apply_gradients(zip(grads, [params]))
            return out

        cost_history = []
        best_cost_this = np.inf
        best_params_this = None
        best_pred_this = None
        best_iter_this = -1
        no_improve_cnt = 0
        plateau_cnt = 0
        last_ma = None
        logs_done = 0

        for i in range(args.num_iter):
            cost, alphaD_pred = optimization_step()
            c = float(cost.numpy())
            pred_np = alphaD_pred.numpy()
            cost_history.append(c)

            rel_impr_best = (best_cost_this - c) / max(abs(best_cost_this), 1e-12) if best_cost_this < np.inf else np.inf
            improved = (rel_impr_best >= MIN_DELTA_REL_BEST) or (best_cost_this == np.inf)
            if improved:
                best_cost_this = c
                best_params_this = params.numpy().copy()
                best_pred_this = pred_np.copy()
                best_iter_this = i
                no_improve_cnt = 0
            else:
                no_improve_cnt += 1

            if (i % args.print_every) == 0:
                logs_done += 1
                alphaD_rel = np.mean(np.abs(pred_np - dataset.alphaD_values) /
                                     np.maximum(np.abs(dataset.alphaD_values), 1e-8))
                print(
                    f"[seed {seed}] iter {i:6d} | cost={c:.6e} | "
                    f"mean alphaD rel={alphaD_rel:.3e}"
                )

                if args.plots == "save":
                    save_alphaD_preview(
                        os.path.join(run_dir, f"train_alphaD_compare_iter{i}.png"),
                        dataset.alphaD_values,
                        pred_np,
                        f"iter = {i} (seed={seed})",
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
        np.savetxt(
            os.path.join(run_dir, "pred_vs_true.txt"),
            np.column_stack([dataset.alphaD_values, best_pred_this]),
            header="alphaD_true alphaD_pred",
        )
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

            save_alphaD_preview(
                os.path.join(run_dir, "final_alphaD_preview.png"),
                dataset.alphaD_values,
                best_pred_this,
                f"Final alphaD preview (seed={seed})",
            )

        if best_cost_this < global_best_cost:
            global_best_cost = best_cost_this
            global_best_params = best_params_this.copy()
            global_best_pred = best_pred_this.copy()
            global_best_meta = {"seed": seed, "iter": best_iter_this}

    np.savetxt(os.path.join(args.save_dir, "best_params_global.txt"), global_best_params)
    np.savetxt(os.path.join(args.save_dir, "train_param_values.txt"), dataset.param_values)
    np.savetxt(
        os.path.join(args.save_dir, "best_pred_vs_true_global.txt"),
        np.column_stack([dataset.alphaD_values, global_best_pred]),
        header="alphaD_true alphaD_pred",
    )
    with open(os.path.join(args.save_dir, "run_summary.json"), "w") as f:
        json.dump({
            "dataset_param_names": dataset.param_names,
            "n_samples": len(dataset.alphaD_values),
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
