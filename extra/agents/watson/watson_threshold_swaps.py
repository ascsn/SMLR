#!/usr/bin/env python
"""Watson: relabel Lorentzian snapshots by thresholded adjacent swaps."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
COMMON_DIR = ROOT / "scripts" / "agents" / "shared"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from snapshot_labeling_utils import (
    apply_orders,
    build_neighbor_edges,
    central_index,
    coordinate_smoothness,
    lorentzian_atoms,
    mst_parent_edges,
    nonoverlapping_swap_masks,
    reconstruct_spectrum,
    rmse,
    spectrum_metrics,
    trapz_weights,
    write_csv,
    write_metrics_text,
)


DEFAULT_SNAPSHOT = ROOT / "tests" / "cache" / "yb_lorentzian_compression" / "snapshots_nstar.npz"
DEFAULT_OUTPUT = ROOT / "tests" / "cache" / "yb_lorentzian_compression" / "agents" / "watson"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Relabel snapshots using thresholded adjacent swaps.")
    parser.add_argument("--snapshots", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--k-neighbors", type=int, default=6)
    parser.add_argument("--no-grid-edges", action="store_true")
    parser.add_argument("--delta-E", type=float, nargs="*", default=[0.1, 0.2, 0.5, 1.0])
    parser.add_argument("--wB", type=float, default=1.0)
    parser.add_argument("--wS", type=float, default=1.0)
    parser.add_argument("--max-candidates", type=int, default=2048)
    parser.add_argument("--sample-limit", type=int, default=None)
    return parser.parse_args()


def load_snapshot(path: Path, sample_limit: int | None):
    from snapshot_labeling_utils import load_snapshots

    return load_snapshots(path, sample_limit=sample_limit)


def local_shape_cost(
    omega: np.ndarray,
    weights: np.ndarray,
    eta: float,
    e_a: np.ndarray,
    b_a: np.ndarray,
    e_b: np.ndarray,
    b_b: np.ndarray,
) -> float:
    atoms_a = b_a[:, None] * lorentzian_atoms(omega, e_a, eta)
    atoms_b = b_b[:, None] * lorentzian_atoms(omega, e_b, eta)
    diff = atoms_a - atoms_b
    return float(np.mean(np.sum(weights[None, :] * diff**2, axis=1)))


def swappable_adjacent_indices(e_a: np.ndarray, e_b_sorted: np.ndarray, delta_e: float) -> list[int]:
    candidates = set()
    gaps_a = np.abs(np.diff(e_a))
    gaps_b = np.abs(np.diff(e_b_sorted))
    for i, gap in enumerate(gaps_a):
        if gap < delta_e:
            candidates.add(i)
    for i, gap in enumerate(gaps_b):
        if gap < delta_e:
            candidates.add(i)
    for i in range(len(e_a) - 1):
        cross = min(abs(e_a[i] - e_b_sorted[i + 1]), abs(e_a[i + 1] - e_b_sorted[i]))
        if cross < delta_e:
            candidates.add(i)
    return sorted(candidates)


def candidate_perm(n_modes: int, swaps: tuple[int, ...]) -> np.ndarray:
    perm = np.arange(n_modes)
    for i in swaps:
        perm[i], perm[i + 1] = perm[i + 1], perm[i]
    return perm


def score_candidate(
    omega: np.ndarray,
    weights: np.ndarray,
    eta: float,
    e_a: np.ndarray,
    b_a: np.ndarray,
    e_b: np.ndarray,
    b_b: np.ndarray,
    perm: np.ndarray,
    w_b: float,
    w_s: float,
) -> float:
    e_bp = e_b[perm]
    b_bp = b_b[perm]
    de = np.mean((e_a - e_bp) ** 2)
    db = np.mean((np.sqrt(np.maximum(b_a, 0.0)) - np.sqrt(np.maximum(b_bp, 0.0))) ** 2)
    ds = local_shape_cost(omega, weights, eta, e_a, b_a, e_bp, b_bp)
    return float(de + w_b * db + w_s * ds)


def best_edge_order(
    omega: np.ndarray,
    weights: np.ndarray,
    eta: float,
    e_a: np.ndarray,
    b_a: np.ndarray,
    e_b_raw: np.ndarray,
    b_b_raw: np.ndarray,
    delta_e: float,
    w_b: float,
    w_s: float,
    max_candidates: int,
) -> tuple[np.ndarray, float, int]:
    sorted_b = np.argsort(e_b_raw)
    e_b = e_b_raw[sorted_b]
    b_b = b_b_raw[sorted_b]
    swap_indices = swappable_adjacent_indices(e_a, e_b, delta_e)
    best_perm = np.arange(len(e_a))
    best_score = score_candidate(omega, weights, eta, e_a, b_a, e_b, b_b, best_perm, w_b, w_s)
    best_n_swaps = 0
    masks = nonoverlapping_swap_masks(swap_indices)
    if len(masks) > int(max_candidates):
        masks = [()] + [(i,) for i in swap_indices]
    for swaps in masks:
        perm = candidate_perm(len(e_a), swaps)
        score = score_candidate(omega, weights, eta, e_a, b_a, e_b, b_b, perm, w_b, w_s)
        if score < best_score:
            best_score = score
            best_perm = perm
            best_n_swaps = len(swaps)
    return sorted_b[best_perm], best_score, best_n_swaps


def run_for_delta(args: argparse.Namespace, data, edges: list[tuple[int, int]], delta_e: float) -> dict[str, object]:
    root = central_index(data.alpha)
    tree_edges, non_tree_edges, _ = mst_parent_edges(data.alpha, edges, root)
    weights = trapz_weights(data.omega)
    n_samples, n_modes = data.e_raw.shape
    orders = np.empty((n_samples, n_modes), dtype=np.int64)
    orders[root] = np.argsort(data.e_raw[root])

    edge_rows = []
    for a, b in tree_edges:
        order_b, score, n_swaps = best_edge_order(
            data.omega,
            weights,
            data.eta,
            data.e_raw[a, orders[a]],
            data.b_raw[a, orders[a]],
            data.e_raw[b],
            data.b_raw[b],
            delta_e,
            args.wB,
            args.wS,
            args.max_candidates,
        )
        orders[b] = order_b
        edge_rows.append({"a": a, "b": b, "score": score, "n_swaps": n_swaps, "is_tree": 1})

    e_labeled, b_labeled = apply_orders(data.e_raw, data.b_raw, orders)
    s_labeled = reconstruct_spectrum(data.omega, e_labeled, b_labeled, data.eta)

    conflict_rows = []
    for a, b in non_tree_edges:
        implied_order_b, score, n_swaps = best_edge_order(
            data.omega,
            weights,
            data.eta,
            e_labeled[a],
            b_labeled[a],
            data.e_raw[b],
            data.b_raw[b],
            delta_e,
            args.wB,
            args.wS,
            args.max_candidates,
        )
        mismatch = implied_order_b != orders[b]
        conflict_rows.append(
            {
                "a": a,
                "b": b,
                "score": score,
                "n_swaps": n_swaps,
                "mismatch_fraction": float(np.mean(mismatch)),
                "mismatch_count": int(np.sum(mismatch)),
            }
        )

    unchanged = rmse(data.s_fit, s_labeled, axis=1)
    metrics = {
        "delta_E": delta_e,
        "root_index": root,
        "n_edges": len(edges),
        "n_tree_edges": len(tree_edges),
        "n_non_tree_edges": len(non_tree_edges),
        **coordinate_smoothness(edges, e_labeled, b_labeled),
        **{f"spectrum_{k}": v for k, v in spectrum_metrics(data.s_true, s_labeled).items()},
        "unchanged_check_rmse_max_vs_snapshot_fit": float(np.max(unchanged)),
        "unchanged_check_rmse_median_vs_snapshot_fit": float(np.median(unchanged)),
        "mean_loop_mismatch_fraction": float(np.mean([r["mismatch_fraction"] for r in conflict_rows])) if conflict_rows else 0.0,
        "max_loop_mismatch_fraction": float(np.max([r["mismatch_fraction"] for r in conflict_rows])) if conflict_rows else 0.0,
        "mean_tree_swaps": float(np.mean([r["n_swaps"] for r in edge_rows])) if edge_rows else 0.0,
    }
    return {
        "delta": delta_e,
        "orders": orders,
        "E": e_labeled,
        "B": b_labeled,
        "S": s_labeled,
        "edge_rows": edge_rows,
        "conflict_rows": conflict_rows,
        "metrics": metrics,
    }


def main() -> None:
    args = parse_args()
    data = load_snapshot(args.snapshots, args.sample_limit)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    edges = build_neighbor_edges(data.alpha, k=args.k_neighbors, include_grid=not args.no_grid_edges)

    summary_rows = []
    for delta_e in args.delta_E:
        result = run_for_delta(args, data, edges, float(delta_e))
        tag = f"delta{float(delta_e):g}".replace(".", "p")
        out_npz = args.output_dir / f"watson_threshold_swaps_{tag}.npz"
        np.savez_compressed(
            out_npz,
            alpha_points=data.alpha,
            param_names=data.param_names,
            sample_ids=data.sample_ids,
            omega=data.omega,
            eta=np.asarray(data.eta),
            delta_E=np.asarray(float(delta_e)),
            E_watson=result["E"],
            B_watson=result["B"],
            label_orders=result["orders"],
            S_reconstructed=result["S"],
            neighbor_edges=np.asarray(edges, dtype=np.int64),
        )
        write_csv(args.output_dir / f"edge_choices_{tag}.csv", result["edge_rows"])
        write_csv(args.output_dir / f"permutation_conflicts_{tag}.csv", result["conflict_rows"])
        write_metrics_text(args.output_dir / f"watson_metrics_{tag}.txt", result["metrics"])
        summary_rows.append(result["metrics"])

    write_csv(args.output_dir / "watson_delta_sweep_summary.csv", summary_rows)
    print(f"Saved Watson delta sweep to {args.output_dir}")
    for row in summary_rows:
        print(
            f"delta_E={row['delta_E']}: E_l2={row['mean_neighbor_E_l2']:.3g}, "
            f"conflict={row['mean_loop_mismatch_fraction']:.3g}, unchanged={row['unchanged_check_rmse_max_vs_snapshot_fit']:.3e}"
        )


if __name__ == "__main__":
    main()
