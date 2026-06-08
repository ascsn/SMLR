#!/usr/bin/env python
"""Sherlock: relabel mobile Lorentzian coordinates by graph OT matching."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
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
    lorentzian_atoms,
    mst_parent_edges,
    reconstruct_spectrum,
    rmse,
    spectrum_metrics,
    trapz_weights,
    write_csv,
    write_metrics_text,
)


DEFAULT_SNAPSHOT = ROOT / "tests" / "cache" / "yb_lorentzian_compression" / "snapshots_nstar.npz"
DEFAULT_OUTPUT = ROOT / "tests" / "cache" / "yb_lorentzian_compression" / "agents" / "sherlock"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Relabel snapshot Lorentzians using OT on a neighbor graph.")
    parser.add_argument("--snapshots", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--k-neighbors", type=int, default=6)
    parser.add_argument("--no-grid-edges", action="store_true")
    parser.add_argument("--cE", type=float, default=1.0)
    parser.add_argument("--cB", type=float, default=1.0)
    parser.add_argument("--cshape", type=float, default=1.0)
    parser.add_argument("--sample-limit", type=int, default=None)
    return parser.parse_args()


def load_snapshot(path: Path, sample_limit: int | None):
    from snapshot_labeling_utils import load_snapshots

    return load_snapshots(path, sample_limit=sample_limit)


def robust_scale(x: np.ndarray, floor: float = 1e-8) -> float:
    x = np.asarray(x, dtype=np.float64).ravel()
    value = float(np.nanmedian(np.abs(x - np.nanmedian(x))) * 1.4826)
    if not np.isfinite(value) or value < floor:
        value = float(np.nanstd(x))
    return max(value, floor)


def pair_cost(
    omega: np.ndarray,
    weights: np.ndarray,
    eta: float,
    e_a: np.ndarray,
    b_a: np.ndarray,
    e_b: np.ndarray,
    b_b: np.ndarray,
    sigma_e: float,
    sigma_sqrt_b: float,
    sigma_shape: float,
    c_e: float,
    c_b: float,
    c_shape: float,
) -> np.ndarray:
    sqrt_a = np.sqrt(np.maximum(b_a, 0.0))
    sqrt_b = np.sqrt(np.maximum(b_b, 0.0))
    c = c_e * ((e_a[:, None] - e_b[None, :]) / sigma_e) ** 2
    c += c_b * ((sqrt_a[:, None] - sqrt_b[None, :]) / sigma_sqrt_b) ** 2

    atoms_a = b_a[:, None] * lorentzian_atoms(omega, e_a, eta)
    atoms_b = b_b[:, None] * lorentzian_atoms(omega, e_b, eta)
    diff = atoms_a[:, None, :] - atoms_b[None, :, :]
    shape = np.sum(weights[None, None, :] * diff**2, axis=-1) / max(sigma_shape**2, 1e-12)
    return c + c_shape * shape


def edge_permutation(cost: np.ndarray) -> np.ndarray:
    rows, cols = linear_sum_assignment(cost)
    perm = np.empty(cost.shape[0], dtype=np.int64)
    perm[rows] = cols
    return perm


def directed_perm(
    edge_perms: dict[tuple[int, int], np.ndarray],
    a: int,
    b: int,
) -> np.ndarray:
    key = tuple(sorted((a, b)))
    perm_low_high = edge_perms[key]
    if a < b:
        return perm_low_high
    return inverse_permutation(perm_low_high)


def main() -> None:
    args = parse_args()
    data = load_snapshot(args.snapshots, args.sample_limit)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    edges = build_neighbor_edges(data.alpha, k=args.k_neighbors, include_grid=not args.no_grid_edges)
    root = central_index(data.alpha)
    tree_edges, non_tree_edges, parent = mst_parent_edges(data.alpha, edges, root)
    weights = trapz_weights(data.omega)

    sigma_e = robust_scale(np.diff(np.sort(data.e_raw, axis=1), axis=1))
    sigma_sqrt_b = robust_scale(np.sqrt(np.maximum(data.b_raw, 0.0)))
    raw_atoms = data.b_raw[:, :, None] * lorentzian_atoms(data.omega, data.e_raw.reshape(-1), data.eta).reshape(
        data.e_raw.shape[0], data.e_raw.shape[1], -1
    )
    sigma_shape = robust_scale(np.sqrt(np.sum(weights[None, None, :] * raw_atoms**2, axis=-1)))

    edge_perms: dict[tuple[int, int], np.ndarray] = {}
    edge_rows = []
    for a, b in edges:
        cost = pair_cost(
            data.omega,
            weights,
            data.eta,
            data.e_raw[a],
            data.b_raw[a],
            data.e_raw[b],
            data.b_raw[b],
            sigma_e,
            sigma_sqrt_b,
            sigma_shape,
            args.cE,
            args.cB,
            args.cshape,
        )
        perm = edge_permutation(cost)
        edge_perms[(a, b)] = perm
        edge_rows.append(
            {
                "a": a,
                "b": b,
                "param_dist": float(np.linalg.norm(data.alpha[a] - data.alpha[b])),
                "assignment_cost": float(cost[np.arange(cost.shape[0]), perm].mean()),
            }
        )

    n_samples, n_modes = data.e_raw.shape
    orders = np.empty((n_samples, n_modes), dtype=np.int64)
    orders[root] = np.arange(n_modes)
    for a, b in tree_edges:
        perm = directed_perm(edge_perms, a, b)
        orders[b] = perm[orders[a]]

    e_labeled, b_labeled = apply_orders(data.e_raw, data.b_raw, orders)
    s_labeled = reconstruct_spectrum(data.omega, e_labeled, b_labeled, data.eta)

    conflict_rows = []
    for a, b in non_tree_edges:
        perm = directed_perm(edge_perms, a, b)
        implied_b = perm[orders[a]]
        mismatch = implied_b != orders[b]
        conflict_rows.append(
            {
                "a": a,
                "b": b,
                "mismatch_fraction": float(np.mean(mismatch)),
                "mismatch_count": int(np.sum(mismatch)),
                "mean_abs_E_disagreement": float(np.mean(np.abs(data.e_raw[b, implied_b] - e_labeled[b]))),
                "mean_abs_sqrtB_disagreement": float(
                    np.mean(np.abs(np.sqrt(np.maximum(data.b_raw[b, implied_b], 0.0)) - np.sqrt(np.maximum(b_labeled[b], 0.0))))
                ),
            }
        )

    unchanged = rmse(data.s_fit, s_labeled, axis=1)
    metrics = {
        "root_index": root,
        "root_sample_id": str(data.sample_ids[root]),
        "n_edges": len(edges),
        "n_tree_edges": len(tree_edges),
        "n_non_tree_edges": len(non_tree_edges),
        "sigma_E": sigma_e,
        "sigma_sqrtB": sigma_sqrt_b,
        "sigma_shape": sigma_shape,
        **coordinate_smoothness(edges, e_labeled, b_labeled),
        **{f"spectrum_{k}": v for k, v in spectrum_metrics(data.s_true, s_labeled).items()},
        "unchanged_check_rmse_max_vs_snapshot_fit": float(np.max(unchanged)),
        "unchanged_check_rmse_median_vs_snapshot_fit": float(np.median(unchanged)),
        "mean_loop_mismatch_fraction": float(np.mean([r["mismatch_fraction"] for r in conflict_rows])) if conflict_rows else 0.0,
        "max_loop_mismatch_fraction": float(np.max([r["mismatch_fraction"] for r in conflict_rows])) if conflict_rows else 0.0,
    }

    np.savez_compressed(
        args.output_dir / "sherlock_relabel_ot.npz",
        alpha_points=data.alpha,
        param_names=data.param_names,
        sample_ids=data.sample_ids,
        omega=data.omega,
        eta=np.asarray(data.eta),
        E_sherlock=e_labeled,
        B_sherlock=b_labeled,
        label_orders=orders,
        S_reconstructed=s_labeled,
        permutation_conflicts=np.asarray(
            [[r["a"], r["b"], r["mismatch_fraction"], r["mismatch_count"]] for r in conflict_rows],
            dtype=float,
        ),
        neighbor_edges=np.asarray(edges, dtype=np.int64),
        tree_edges=np.asarray(tree_edges, dtype=np.int64),
        non_tree_edges=np.asarray(non_tree_edges, dtype=np.int64),
        root_index=np.asarray(root, dtype=np.int64),
    )
    write_csv(args.output_dir / "edge_assignments.csv", edge_rows)
    write_csv(args.output_dir / "permutation_conflicts.csv", conflict_rows)
    write_metrics_text(args.output_dir / "sherlock_metrics.txt", metrics)

    print(f"Saved Sherlock relabeling to {args.output_dir / 'sherlock_relabel_ot.npz'}")
    print(f"Root index: {root}; loop conflicts: {metrics['mean_loop_mismatch_fraction']:.3f} mean mismatch")
    print(f"Unchanged spectrum check max RMSE vs snapshot fit: {metrics['unchanged_check_rmse_max_vs_snapshot_fit']:.3e}")


if __name__ == "__main__":
    main()
