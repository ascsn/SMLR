#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import re
import argparse
import numpy as np
import tensorflow as tf

# Headless backend (never shows figures)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from numpy.polynomial.polynomial import Polynomial
import random as rn

try:
    from . import helper_gpt as helper
except ImportError:  # pragma: no cover - direct script execution
    import archive.Beta_decay_package.src.helper_gpt as helper
try:
    from smlr.core import EDIT_training as core_training
    from smlr.core import RetainedModePolicy
    from smlr.serialization import save_emulator
    from smlr.specs import paper_beta_em1_spec
except ModuleNotFoundError:  # pragma: no cover - source-tree execution before install
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from smlr.core import EDIT_training as core_training
    from smlr.core import RetainedModePolicy
    from smlr.serialization import save_emulator
    from smlr.specs import paper_beta_em1_spec


# -------------------- CLI --------------------
def parse_args():
    p = argparse.ArgumentParser(description="Beta-decay EM1 training")
    # core knobs
    p.add_argument("--n",            type=int,   default=8,       help="Model dimension")
    p.add_argument("--retain",       type=float, default=0.9,     help="Fraction of eigenmodes kept (centered)")
    p.add_argument("--weight",       type=float, default=1.0,     help="Initial HL weight")
    p.add_argument("--n-restarts",   type=int,   default=1,       help="Number of different seeds")
    p.add_argument("--seed0",        type=int,   default=42,      help="Base seed (seeds = seed0..seed0+n_restarts-1)")
    p.add_argument("--num-iter",     type=int,   default=30000,   help="Max iterations per restart")
    p.add_argument("--min-iter",     type=int,   default=20000,   help=argparse.SUPPRESS)
    p.add_argument("--print-every",  type=int,   default=1000,     help="Logging cadence (iterations)")
    p.add_argument("--save-dir",     type=str,   default="Beta_decay_package/runs_em1", help="Directory to save run artifacts")
    p.add_argument("--data-dir",     type=str,   default="beta_decay_80Ni", help="Root directory for beta-decay data")
    p.add_argument("--strength-window", type=float, nargs=2, metavar=("LOW", "HIGH"), default=None,
                   help="Optional energy window for strength files. Use only for strength-only datasets.")
    p.add_argument("--fixed-width",  type=float, default=None,
                   help="If set, use this fixed Lorentzian width instead of the affine width model.")
    p.add_argument("--s-init-scale", type=float, default=None,
                   help="Scale for random S_i initialization. Defaults to 0.1 for 4D, 1.0 for 2D.")
    p.add_argument("--s-init-sweep", type=float, nargs="+", default=None,
                   help="Optional list of S_i initialization scales to sweep.")
    p.add_argument("--no-coordinate-normalization", action="store_true", default=False,
                   help="Disable coordinate normalization in the matrix and affine-width ansatz.")
    p.add_argument("--reference-index", type=int, default=0,
                   help="Dataset index to use as the linear expansion/reference point.")

    # plot controls (no on-screen display ever)
    p.add_argument("--plots", choices=["none", "save"], default="save",
                   help="Training plots: none (default) or save to disk")
    p.add_argument("--phase-plot", action="store_true", default=False,
                   help="If set, also save a phase-space plot (saved per seed when --plots save).")
    return p.parse_args()


def main():
    args = parse_args()

    # bind parsed values
    n              = args.n
    retain         = args.retain
    weight         = args.weight
    N_RESTARTS     = args.n_restarts
    SEED0          = args.seed0
    NUM_ITERATIONS = args.num_iter
    MIN_ITERATIONS = args.min_iter
    PRINT_EVERY    = args.print_every
    SAVE_DIR       = args.save_dir
    DATA_DIR       = args.data_dir
    STRENGTH_WINDOW = tuple(args.strength_window) if args.strength_window is not None else None
    FIXED_WIDTH    = args.fixed_width
    S_INIT_SCALE   = args.s_init_scale
    S_INIT_SWEEP   = args.s_init_sweep
    REFERENCE_INDEX = args.reference_index
    NORMALIZE_COORDINATES = not args.no_coordinate_normalization
    PLOTS_MODE     = args.plots              # "none" | "save"
    DO_PHASE_PLOT  = args.phase_plot         # True/False

    os.makedirs(SAVE_DIR, exist_ok=True)
    os.environ["SMLR_BETA_DATA_DIR"] = DATA_DIR

    # -------------------- policy: minimum iterations --------------------
    if NUM_ITERATIONS < MIN_ITERATIONS:
        raise ValueError(
            f"--num-iter ({NUM_ITERATIONS}) must be >= {MIN_ITERATIONS}. "
            f"Increase --num-iter or adjust MIN_ITERATIONS in the script."
        )

    # -------------------- constants --------------------
    A = 80
    Z = 28
    g_A = 1.2
    nucnam = 'Ni_80'

    # -------------------- phase-space polynomial (for HL) --------------------
    poly = helper.fit_phase_space(0, Z, A, 15)
    coeffs = Polynomial(poly).coef

    coefficients = tf.constant(coeffs[::-1], dtype=tf.float64)  # reversed for Horner
    x_values = np.linspace(0.611, 15, 100, dtype=float)
    x_tensor = tf.constant(x_values, dtype=tf.float64)
    y_values = helper.evaluate_polynomial_tf(coefficients, x_tensor).numpy()

    def save_phase_plot(out_dir: str):
        if not (DO_PHASE_PLOT and PLOTS_MODE == "save"):
            return
        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(111)
        ax.plot(x_values, y_values, label='phase-space poly')
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('f(x)', fontsize=12)
        ax.set_title('Polynomial Plot', fontsize=14)
        ax.axhline(0, color='gray', lw=0.8)
        ax.axvline(0, color='gray', lw=0.8)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()
        ax.set_yscale('log')
        for i, xv in enumerate(x_values):
            if (i % 10 == 0):
                ax.scatter(xv, helper.phase_factor(0, Z, A, xv/helper.emass), color='red', s=10)
        fig.savefig(os.path.join(out_dir, "phase_space_poly.png"), bbox_inches="tight", dpi=150)
        plt.close(fig)

    # -------------------- dataset from files --------------------
    rn.seed(20)

    strength_dir = helper.beta_data_dir(nucnam)
    pattern_old = re.compile(r'lorm_' + re.escape(nucnam) + r'_(-?[0-9.]+)_(-?[0-9.]+)\.out')
    pattern_new = re.compile(r'strength_(-?[0-9.]+)_(-?[0-9.]+)_(-?[0-9.]+)_(-?[0-9.]+)\.out')

    combined = []
    for fname in sorted(os.listdir(strength_dir)):
        old_match = pattern_old.match(fname)
        new_match = pattern_new.match(fname)
        path = os.path.join(strength_dir, fname)
        if old_match:
            beta_val = old_match.group(1)
            alpha_val = old_match.group(2)
            params = (float(alpha_val), float(beta_val))
            # keep region
            if ((0.1 <= float(beta_val) <= 0.9) and (0.2 <= float(alpha_val) <= 1.8)):
                combined.append((params, path))
        elif new_match:
            params = tuple(float(v) for v in new_match.groups())
            combined.append((params, path))

    if not combined:
        raise ValueError(f"No beta strength files found in {strength_dir!r}.")

    # splits
    train_ratio = 0.6; cv_ratio = 0.0; test_ratio = 0.4
    n_total = len(combined)
    n_train = int(n_total * train_ratio)
    n_cv    = int(n_total * cv_ratio)
    n_test  = n_total - n_train - n_cv

    train_set = combined[:n_train]
    cv_set    = combined[n_train:n_train + n_cv]
    test_set  = combined[n_train + n_cv:]

    # Use a concrete sampled point as the linear expansion/reference point.
    combined_ar = np.array([helper.dataset_entry_params(entry) for entry in combined], dtype=float)
    reference_index = max(0, min(int(REFERENCE_INDEX), len(combined) - 1))
    central_point = helper.dataset_entry_params(combined[reference_index])
    print('Reference data point:', central_point)
    print('Sizes -> test:', len(test_set), 'cv:', len(cv_set), 'train:', len(train_set))

    num_components = len(helper.dataset_entry_params(combined[0]))
    coordinate_scales = None
    if NORMALIZE_COORDINATES:
        coord_ranges = np.ptp(combined_ar, axis=0)
        coord_ranges = np.where(coord_ranges > 0.0, coord_ranges, 1.0)
        coordinate_scales = tuple(float(v) for v in coord_ranges)
    print('Coordinate scales:', coordinate_scales)

    # -------------------- model & data table --------------------
    np.random.seed(42)
    D_and_S = helper.nec_mat(n, num_components=num_components)
    D, *S_list = D_and_S
    print('Matrix shapes:', D.shape, [S.shape for S in S_list])

    data_nucnam = nucnam if num_components == 2 else None
    if data_nucnam is None and weight != 0.0:
        raise ValueError("The 4D strength-only dataset has no half-life files; use --weight 0.0.")
    Lors, HLs = helper.data_table(
        train_set, coeffs, g_A, data_nucnam, strength_window=STRENGTH_WINDOW
    )

    num_upper = int(n * (n + 1) / 2)
    nec_num_param = 2 * n + num_components * num_upper + num_components + 2
    print('Number of parameters:', nec_num_param)

    # -------------------- utils --------------------
    def set_all_seeds(seed: int):
        core_training.set_all_seeds(seed)

    if S_INIT_SWEEP is not None:
        s_init_scales = [float(x) for x in S_INIT_SWEEP]
    else:
        if S_INIT_SCALE is None:
            S_INIT_SCALE = 0.1 if (num_components > 2 and NORMALIZE_COORDINATES) else \
                           (1e-4 if num_components > 2 else 1.0)
        s_init_scales = [float(S_INIT_SCALE)]

    def scale_label(scale: float) -> str:
        return f"{scale:.0e}".replace("+", "").replace("-", "m")

    def make_initial_guess(nec_num_param: int, seed: int, s_init_scale: float):
        rng = np.random.default_rng(seed)
        init = rng.uniform(0.0, 1.0, size=nec_num_param)

        idx = 0
        init[idx] = 1.0 if FIXED_WIDTH is None else float(FIXED_WIDTH)
        idx += 1

        init[idx:idx + n] = rng.uniform(0.05, 1.0, size=n)
        idx += n

        energy_min = min(float(np.min(Lor[:, 0])) for Lor in Lors)
        energy_max = max(float(np.max(Lor[:, 0])) for Lor in Lors)
        init[idx:idx + n] = np.linspace(energy_min, energy_max, n)
        idx += n

        for _ in range(num_components):
            init[idx:idx + num_upper] = rng.normal(0.0, s_init_scale, size=num_upper)
            idx += num_upper

        init[idx:idx + num_components + 1] = 0.0
        return init

    def make_optimizer(learning_rate: float):
        return core_training.make_optimizer(learning_rate)

    # early-stopping thresholds
    PATIENCE_BEST       = 120
    MIN_DELTA_REL_BEST  = 2e-4
    PATIENCE_PLATEAU    = 300
    MA_WINDOW           = 10
    MIN_DELTA_REL       = 3e-4

    def moving_average(arr, k):
        return core_training.moving_average(arr, k)

    def maybe_training_plots(iter_idx, rel, E_last, B_last, x, Lor, Lor_true, seed, out_dir):
        """Save training-time diagnostics if requested. Never shows on screen."""
        if PLOTS_MODE != "save":
            return
        fig = plt.figure(figsize=(10, 4))
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.plot(np.arange(len(rel)), rel, marker='.', ls='--')
        ax1.axhline(np.mean(rel), color='k', ls='--')
        ax1.set_yscale('log')
        ax1.set_title(f'Half-life rel. err. (iter={iter_idx})')

        ax2 = fig.add_subplot(1, 2, 2)
        ax2.stem(E_last, B_last) #, use_line_collection=True)
        ax2.plot(x, Lor, label='pred')
        ax2.plot(x, Lor_true, label='true')
        ax2.set_ylim(0)
        ax2.set_xlim(-10, np.max(x))
        ax2.axvline(0.8, ls='--')
        ax2.legend()
        fig.tight_layout()

        out = os.path.join(out_dir, f"train_diag_iter{iter_idx}.png")
        fig.savefig(out, bbox_inches='tight', dpi=130)
        plt.close(fig)

    # -------------------- training --------------------
    global_best_cost  = np.inf
    global_best_params = None
    global_best_meta  = {"seed": None, "iter": None}
    last_cost_history = None

    for s_init_scale in s_init_scales:
        scale_dir = os.path.join(SAVE_DIR, f"sinit_{scale_label(s_init_scale)}")
        os.makedirs(scale_dir, exist_ok=True)

        for r in range(N_RESTARTS):
            seed = SEED0 + r
            run_dir = os.path.join(scale_dir, f"seed_{seed}")
            os.makedirs(run_dir, exist_ok=True)

            print(f"\n========== S_INIT {s_init_scale:.3e} | RESTART {r+1}/{N_RESTARTS} (seed={seed}) ==========")
            print(f"Policy: MIN_ITERATIONS = {MIN_ITERATIONS}, user max = {NUM_ITERATIONS}")
            set_all_seeds(seed)

            # per-seed phase plot (if requested)
            save_phase_plot(run_dir)

            init_vec = make_initial_guess(nec_num_param, seed, s_init_scale)
            params = tf.Variable(init_vec, dtype=tf.float64)
            optimizer = make_optimizer(learning_rate=0.01)

            @tf.function
            def optimization_step():
                with tf.GradientTape() as tape:
                    cost, Lor, Lor_true, x, HLs_calc, B_last, E_last = helper.cost_function(
                        params, n, train_set, Lors, HLs, coeffs, g_A, weight, central_point, retain,
                        num_components=num_components, fixed_width=FIXED_WIDTH,
                        coordinate_scales=coordinate_scales
                    )
                grads = tape.gradient(cost, [params])
                optimizer.apply_gradients(zip(grads, [params]))
                return cost, Lor, Lor_true, x, HLs_calc, B_last, E_last

            cost_history   = []
            best_cost_this = np.inf
            best_params_this = None
            best_iter_this = -1

            no_improve_cnt = 0  # raw iters
            plateau_cnt    = 0  # logging steps
            last_ma        = None

            for i in range(NUM_ITERATIONS):
                cost, Lor, Lor_true, x, HLs_check_train, B_last, E_last = optimization_step()
                cost_val = float(cost.numpy())
                cost_history.append(cost_val)

                improved = (best_cost_this - cost_val) >= MIN_DELTA_REL_BEST * max(abs(best_cost_this), 1e-12)
                if improved or (best_cost_this == np.inf):
                    best_cost_this  = cost_val
                    best_params_this = params.numpy().copy()
                    best_iter_this  = i
                    no_improve_cnt  = 0
                else:
                    no_improve_cnt += 1

                if (i % PRINT_EVERY) == 0:
                    try:
                        current_lr = float(optimizer._decayed_lr(tf.float64).numpy())
                    except Exception:
                        current_lr = 0.01
                    rel = np.abs(np.array(HLs_check_train) - np.array(HLs)) / np.array(HLs)
                    print(f"[s_init {s_init_scale:.1e} seed {seed}] iter {i:6d} | cost={cost_val:.6e} "
                          f"| lr={current_lr:.3e} | mean r={np.mean(rel):.3e} | no_improve={no_improve_cnt}")

                    # diagnostics plot (saved only if requested) -> in seed subdir
                    maybe_training_plots(i, rel, E_last, B_last, x, Lor, Lor_true, seed, run_dir)

                    # plateau check at logging cadence
                    ds_costs = cost_history[::PRINT_EVERY] if PRINT_EVERY > 0 else cost_history[:]
                    ma = moving_average(ds_costs, MA_WINDOW)
                    if ma is not None:
                        current_ma = ma[-1]
                        if last_ma is None:
                            last_ma = current_ma; plateau_cnt = 0
                        else:
                            rel_impr = (last_ma - current_ma) / max(abs(last_ma), 1e-12)
                            if rel_impr >= MIN_DELTA_REL:
                                plateau_cnt = 0; last_ma = current_ma
                            else:
                                plateau_cnt += 1

                # early-stopping gates (not allowed before MIN_ITERATIONS)
                stop_best    = (no_improve_cnt >= PATIENCE_BEST)
                stop_plateau = (plateau_cnt    >= PATIENCE_PLATEAU)
                can_stop     = (i + 1) >= MIN_ITERATIONS  # i is 0-based

                if can_stop and (stop_best or stop_plateau):
                    reason = "no best improvement" if stop_best else "moving-average plateau"
                    print(f"[s_init {s_init_scale:.1e} seed {seed}] Early stopping at iter {i} due to {reason}. "
                          f"best_cost={best_cost_this:.6e} (iter={best_iter_this})")
                    break

            # save per-restart artifacts (always in seed subdir)
            np.savetxt(os.path.join(run_dir, f"params_n{n}_retain{retain}_seed{seed}.txt"),
                       best_params_this)
            np.savetxt(os.path.join(run_dir, f"cost_history_seed{seed}.txt"),
                       np.array(cost_history))
            with open(os.path.join(run_dir, "meta.txt"), "w") as f:
                f.write(f"best_cost={best_cost_this}\n")
                f.write(f"best_iter={best_iter_this}\n")
                f.write(f"seed={seed}\n")
                f.write(f"s_init_scale={s_init_scale}\n")
                f.write(f"coordinate_scales={coordinate_scales}\n")
                f.write(f"n={n}\nretain={retain}\nweight={weight}\n")
                f.write(f"stopped_at={len(cost_history)-1}\n")
                f.write(f"min_iterations_enforced={MIN_ITERATIONS}\n")

            # convergence curve (saved only if requested) -> in seed subdir
            if PLOTS_MODE == "save":
                fig = plt.figure()
                ax = fig.add_subplot(111)
                ax.plot(range(len(cost_history)), cost_history, label=f'seed {seed}')
                ax.set_yscale('log')
                ax.set_xlabel('Iteration')
                ax.set_ylabel('Cost')
                ax.set_title(f'Convergence (s_init={s_init_scale:.1e}, seed={seed}) '
                             f'| best={best_cost_this:.3e} @ iter {best_iter_this}')
                ax.legend()
                fig.savefig(os.path.join(run_dir, f'cost_curve_seed{seed}.png'), bbox_inches='tight', dpi=150)
                fig.savefig(os.path.join(run_dir, f'cost_curve_seed{seed}.pdf'), bbox_inches='tight')
                plt.close(fig)

            print(f"[s_init {s_init_scale:.1e} seed {seed}] best_iter={best_iter_this} "
                  f"best_cost={best_cost_this:.6e}")

            # update global best
            if best_cost_this < global_best_cost:
                global_best_cost   = best_cost_this
                global_best_params = best_params_this.copy()
                global_best_meta   = {"seed": seed, "iter": best_iter_this, "s_init_scale": s_init_scale}

            last_cost_history = cost_history

    # -------------------- save global best & summary --------------------
    np.savetxt(os.path.join(SAVE_DIR, f'params_best_n{n}_retain{retain}.txt'), global_best_params)

    with open(os.path.join(SAVE_DIR, "train_set.txt"), "w") as f:
        for tup in train_set:
            f.write(",".join(map(str, helper.dataset_entry_params(tup))) + "\n")
    spec = paper_beta_em1_spec(data_dir=str(DATA_DIR), nucnam=nucnam, output_dir=SAVE_DIR)
    spec = spec.__class__(
        name=spec.name,
        strength=spec.strength,
        observable=spec.observable,
        model={**dict(spec.model), "n": n, "retain": retain},
        train_filter_ranges=spec.train_filter_ranges,
        central_point=tuple(float(x) for x in central_point),
        output_dir=SAVE_DIR,
        metadata=spec.metadata,
    )
    if num_components == 2:
        save_emulator(
            SAVE_DIR,
            params=global_best_params,
            name="paper_beta_em1",
            adapter="PaperBetaDecayAdapter",
            spec=spec,
            retained_modes=RetainedModePolicy(kind="centered", retain=retain),
            metadata={"global_best_cost": global_best_cost, "global_best_meta": global_best_meta},
        )
    else:
        print(
            "Skipping packaged emulator save: PaperBetaDecayAdapter serialization currently supports "
            "the 2D beta-decay layout only."
        )

    # final quick plot (saved only if requested) -> in global-best seed subdir
    if last_cost_history is not None and PLOTS_MODE == "save":
        best_scale_dir = os.path.join(SAVE_DIR, f"sinit_{scale_label(global_best_meta['s_init_scale'])}")
        best_seed_dir = os.path.join(best_scale_dir, f"seed_{global_best_meta['seed']}")
        os.makedirs(best_seed_dir, exist_ok=True)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(range(len(last_cost_history)), last_cost_history, label='Last restart cost')
        ax.set_yscale('log')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Cost')
        ax.legend()
        ax.set_title(f"Global best cost={global_best_cost:.3e} "
                     f"(seed={global_best_meta['seed']}, iter={global_best_meta['iter']})")
        fig.savefig(os.path.join(best_seed_dir, "last_restart_cost.png"), bbox_inches='tight', dpi=150)
        plt.close(fig)

    print(f"\n*** GLOBAL BEST *** cost={global_best_cost:.6e} "
          f"(s_init={global_best_meta['s_init_scale']:.3e}, "
          f"seed={global_best_meta['seed']}, iter={global_best_meta['iter']})")
    print(f"Saved: {SAVE_DIR}/params_best_n{n}_retain{retain}.txt and per-seed artifacts in "
          f"{SAVE_DIR}/sinit_<scale>/seed_<seed>/")


if __name__ == "__main__":
    main()
