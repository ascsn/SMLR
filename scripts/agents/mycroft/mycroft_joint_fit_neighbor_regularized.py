#!/usr/bin/env python
"""Mycroft: joint neighbor-regularized Lorentzian snapshot fit."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "smlr_mplconfig"))

import numpy as np
import tensorflow as tf
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[3]
COMMON_DIR = ROOT / "scripts" / "agents" / "shared"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from snapshot_labeling_utils import (
    apply_orders,
    build_neighbor_edges,
    central_index,
    coordinate_smoothness,
    inverse_permutation,
    mst_parent_edges,
    reconstruct_spectrum,
    spectrum_metrics,
    trapz_weights,
    write_csv,
    write_metrics_text,
)


DEFAULT_SNAPSHOT = ROOT / "tests" / "cache" / "yb_lorentzian_compression" / "snapshots_nstar.npz"
DEFAULT_OUTPUT = ROOT / "tests" / "cache" / "yb_lorentzian_compression" / "agents" / "mycroft"
HQC = 197.33
ALPHAD_FAC = 8.0 * np.pi * (7.29735e-3) * HQC / 9.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Jointly fit Lorentzian coordinates with neighbor regularization.")
    parser.add_argument("--snapshots", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--k-neighbors", type=int, default=6)
    parser.add_argument("--no-grid-edges", action="store_true")
    parser.add_argument("--lambda-neighbor", type=float, default=10.0)
    parser.add_argument("--lambda-alphaD", type=float, default=0.1)
    parser.add_argument("--lambda-spacing", type=float, default=1.0)
    parser.add_argument("--anneal-factor", type=float, default=0.1)
    parser.add_argument("--stages", type=int, default=4)
    parser.add_argument("--steps-per-stage", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--min-spacing", type=float, default=0.01)
    parser.add_argument("--train-shared-eta", action="store_true")
    parser.add_argument("--sample-limit", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def load_snapshot(path: Path, sample_limit: int | None):
    from snapshot_labeling_utils import load_snapshots

    return load_snapshots(path, sample_limit=sample_limit)


def inv_softplus(y: np.ndarray) -> np.ndarray:
    y = np.maximum(np.asarray(y, dtype=np.float64), 1e-12)
    return np.log(np.expm1(y))


def alphaD_from_strength(omega: np.ndarray, y: np.ndarray) -> np.ndarray:
    return ALPHAD_FAC * np.trapezoid(y / np.maximum(omega[None, :], 1e-6), omega, axis=1)


def initialize_by_neighbor_continuation(data, edges: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray, int, list[tuple[int, int]]]:
    root = central_index(data.alpha)
    tree_edges, _, _ = mst_parent_edges(data.alpha, edges, root)
    n_samples, n_modes = data.e_raw.shape
    orders = np.empty((n_samples, n_modes), dtype=np.int64)
    orders[root] = np.argsort(data.e_raw[root])
    for a, b in tree_edges:
        e_a = data.e_raw[a, orders[a]]
        b_a = data.b_raw[a, orders[a]]
        sqrt_a = np.sqrt(np.maximum(b_a, 0.0))
        sqrt_b = np.sqrt(np.maximum(data.b_raw[b], 0.0))
        sigma_e = max(float(np.std(data.e_raw)), 1e-8)
        sigma_b = max(float(np.std(np.sqrt(np.maximum(data.b_raw, 0.0)))), 1e-8)
        cost = ((e_a[:, None] - data.e_raw[b][None, :]) / sigma_e) ** 2
        cost += ((sqrt_a[:, None] - sqrt_b[None, :]) / sigma_b) ** 2
        rows, cols = linear_sum_assignment(cost)
        perm = np.empty(n_modes, dtype=np.int64)
        perm[rows] = cols
        orders[b] = perm
    e0, b0 = apply_orders(data.e_raw, data.b_raw, orders)
    return e0, b0, root, tree_edges


def tf_lorentzian_batch(omega: tf.Tensor, e: tf.Tensor, b: tf.Tensor, eta: tf.Tensor) -> tf.Tensor:
    half = 0.5 * eta
    omega_exp = omega[None, None, :]
    e_exp = e[:, :, None]
    b_exp = b[:, :, None]
    return tf.reduce_sum(b_exp * (half / np.pi) / (tf.square(omega_exp - e_exp) + tf.square(half)), axis=1)


def train(args: argparse.Namespace, data, edges: list[tuple[int, int]]) -> dict[str, object]:
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)
    weights_np = trapz_weights(data.omega).astype(np.float64)
    e0, b0, root, tree_edges = initialize_by_neighbor_continuation(data, edges)
    z0 = inv_softplus(np.sqrt(np.maximum(b0, 1e-12)))

    e_var = tf.Variable(e0, dtype=tf.float64, name="E")
    z_b_var = tf.Variable(z0, dtype=tf.float64, name="zB")
    eta_var = tf.Variable(np.log(np.expm1(float(data.eta))), dtype=tf.float64, name="z_eta")
    variables: list[tf.Variable] = [e_var, z_b_var]
    if args.train_shared_eta:
        variables.append(eta_var)

    omega = tf.constant(data.omega, dtype=tf.float64)
    s_true = tf.constant(data.s_true, dtype=tf.float64)
    weights = tf.constant(weights_np, dtype=tf.float64)
    edges_tf = tf.constant(np.asarray(edges, dtype=np.int64), dtype=tf.int32)
    alphaD_true = tf.constant(alphaD_from_strength(data.omega, data.s_true), dtype=tf.float64)
    norm = tf.reduce_sum(weights[None, :] * tf.square(s_true), axis=1) + tf.constant(1e-12, tf.float64)
    sigma_e = tf.constant(max(float(np.std(e0)), 1e-8), dtype=tf.float64)
    sigma_b = tf.constant(max(float(np.std(np.sqrt(np.maximum(b0, 0.0)))), 1e-8), dtype=tf.float64)
    omega_min = tf.constant(float(np.min(data.omega)), dtype=tf.float64)
    omega_max = tf.constant(float(np.max(data.omega)), dtype=tf.float64)
    min_spacing = tf.constant(float(args.min_spacing), dtype=tf.float64)

    optimizer = tf.keras.optimizers.Adam(learning_rate=float(args.learning_rate))
    history = []

    def losses(lambda_neighbor: float) -> tuple[tf.Tensor, dict[str, tf.Tensor], tf.Tensor, tf.Tensor, tf.Tensor]:
        sqrt_b = tf.nn.softplus(z_b_var)
        b = tf.square(sqrt_b)
        eta = tf.nn.softplus(eta_var) if args.train_shared_eta else tf.constant(float(data.eta), dtype=tf.float64)
        s_pred = tf_lorentzian_batch(omega, e_var, b, eta)
        spectrum_loss = tf.reduce_mean(tf.reduce_sum(weights[None, :] * tf.square(s_pred - s_true), axis=1) / norm)

        alphaD_pred = tf.reduce_sum(b * tf.cast(e_var > 1.0, tf.float64) / tf.maximum(e_var, 1e-6), axis=1) * ALPHAD_FAC
        alphaD_loss = tf.reduce_mean(tf.square((alphaD_pred - alphaD_true) / tf.maximum(tf.abs(alphaD_true), 1e-12)))

        if len(edges):
            ea = tf.gather(e_var, edges_tf[:, 0])
            eb = tf.gather(e_var, edges_tf[:, 1])
            ba = tf.gather(sqrt_b, edges_tf[:, 0])
            bb = tf.gather(sqrt_b, edges_tf[:, 1])
            neighbor_loss = tf.reduce_mean(tf.square((ea - eb) / sigma_e)) + tf.reduce_mean(tf.square((ba - bb) / sigma_b))
        else:
            neighbor_loss = tf.constant(0.0, dtype=tf.float64)

        gap_loss = tf.reduce_mean(tf.square(tf.nn.relu(min_spacing - (e_var[:, 1:] - e_var[:, :-1]))))
        bound_loss = tf.reduce_mean(tf.square(tf.nn.relu(omega_min - e_var)) + tf.square(tf.nn.relu(e_var - omega_max)))
        spacing_loss = gap_loss + bound_loss

        total = spectrum_loss
        total += tf.constant(args.lambda_alphaD, tf.float64) * alphaD_loss
        total += tf.constant(lambda_neighbor, tf.float64) * neighbor_loss
        total += tf.constant(args.lambda_spacing, tf.float64) * spacing_loss
        parts = {
            "spectrum_loss": spectrum_loss,
            "alphaD_loss": alphaD_loss,
            "neighbor_loss": neighbor_loss,
            "spacing_loss": spacing_loss,
            "total_loss": total,
        }
        return total, parts, s_pred, b, eta

    for stage in range(int(args.stages)):
        lambda_neighbor = float(args.lambda_neighbor) * (float(args.anneal_factor) ** stage)
        for step in range(int(args.steps_per_stage)):
            with tf.GradientTape() as tape:
                total, parts, _, _, eta = losses(lambda_neighbor)
            grads = tape.gradient(total, variables)
            optimizer.apply_gradients(zip(grads, variables))
            if step == 0 or step == int(args.steps_per_stage) - 1:
                history.append(
                    {
                        "stage": stage,
                        "step": step,
                        "lambda_neighbor": lambda_neighbor,
                        "eta": float(eta.numpy()),
                        **{name: float(value.numpy()) for name, value in parts.items()},
                    }
                )
        print(
            f"stage {stage + 1}/{args.stages}: lambda_neighbor={lambda_neighbor:g}, "
            f"loss={history[-1]['total_loss']:.4g}, spectrum={history[-1]['spectrum_loss']:.4g}"
        )

    _, parts, s_pred_tf, b_tf, eta_tf = losses(float(args.lambda_neighbor) * (float(args.anneal_factor) ** max(int(args.stages) - 1, 0)))
    return {
        "E": e_var.numpy(),
        "B": b_tf.numpy(),
        "S": s_pred_tf.numpy(),
        "eta": float(eta_tf.numpy()),
        "root": root,
        "tree_edges": tree_edges,
        "history": history,
        "initial_E": e0,
        "initial_B": b0,
        "final_losses": {name: float(value.numpy()) for name, value in parts.items()},
    }


def main() -> None:
    args = parse_args()
    data = load_snapshot(args.snapshots, args.sample_limit)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    edges = build_neighbor_edges(data.alpha, k=args.k_neighbors, include_grid=not args.no_grid_edges)
    result = train(args, data, edges)

    s_initial = reconstruct_spectrum(data.omega, result["initial_E"], result["initial_B"], data.eta)
    initial_metrics = spectrum_metrics(data.s_true, s_initial)
    final_metrics = spectrum_metrics(data.s_true, result["S"])
    metrics = {
        "root_index": result["root"],
        "n_edges": len(edges),
        "eta_initial": data.eta,
        "eta_final": result["eta"],
        **{f"initial_{key}": value for key, value in initial_metrics.items()},
        **{f"final_{key}": value for key, value in final_metrics.items()},
        **{f"final_{key}": value for key, value in coordinate_smoothness(edges, result["E"], result["B"]).items()},
        **result["final_losses"],
    }

    np.savez_compressed(
        args.output_dir / "mycroft_joint_fit_neighbor_regularized.npz",
        alpha_points=data.alpha,
        param_names=data.param_names,
        sample_ids=data.sample_ids,
        omega=data.omega,
        eta=np.asarray(result["eta"]),
        E_mycroft=result["E"],
        B_mycroft=result["B"],
        S_reconstructed=result["S"],
        E_initial=result["initial_E"],
        B_initial=result["initial_B"],
        neighbor_edges=np.asarray(edges, dtype=np.int64),
        tree_edges=np.asarray(result["tree_edges"], dtype=np.int64),
        root_index=np.asarray(result["root"], dtype=np.int64),
    )
    write_csv(args.output_dir / "mycroft_training_history.csv", result["history"])
    write_metrics_text(args.output_dir / "mycroft_metrics.txt", metrics)
    print(f"Saved Mycroft joint fit to {args.output_dir / 'mycroft_joint_fit_neighbor_regularized.npz'}")
    print(f"Initial median RMSE: {metrics['initial_rmse_median']:.4g}")
    print(f"Final median RMSE: {metrics['final_rmse_median']:.4g}")


if __name__ == "__main__":
    main()
