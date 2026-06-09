"""Shared utilities for Lorentzian snapshot labeling experiments."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree


@dataclass(frozen=True)
class SnapshotData:
    alpha: np.ndarray
    omega: np.ndarray
    s_true: np.ndarray
    eta: float
    e_raw: np.ndarray
    b_raw: np.ndarray
    s_fit: np.ndarray
    rmse: np.ndarray
    sample_ids: np.ndarray
    param_names: np.ndarray


def load_snapshots(path: str | Path, sample_limit: int | None = None) -> SnapshotData:
    packed = np.load(path, allow_pickle=False)
    limit = None if sample_limit is None else int(sample_limit)
    sample_slice = slice(None, limit)
    return SnapshotData(
        alpha=np.asarray(packed["alpha_points"], dtype=np.float64)[sample_slice],
        omega=np.asarray(packed["omega"], dtype=np.float64),
        s_true=np.asarray(packed["S_true"], dtype=np.float64)[sample_slice],
        eta=float(np.asarray(packed["eta"])),
        e_raw=np.asarray(packed["E_raw"], dtype=np.float64)[sample_slice],
        b_raw=np.asarray(packed["B_raw"], dtype=np.float64)[sample_slice],
        s_fit=np.asarray(packed["S_fit"], dtype=np.float64)[sample_slice],
        rmse=np.asarray(packed["rmse"], dtype=np.float64)[sample_slice],
        sample_ids=np.asarray(packed["sample_ids"])[sample_slice] if "sample_ids" in packed.files else np.arange(len(packed["E_raw"]))[sample_slice],
        param_names=np.asarray(packed["param_names"]) if "param_names" in packed.files else np.asarray([f"p{i}" for i in range(packed["alpha_points"].shape[1])]),
    )


def trapz_weights(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    dx = np.diff(x)
    weights = np.empty_like(x)
    weights[0] = dx[0] / 2.0
    weights[-1] = dx[-1] / 2.0
    weights[1:-1] = (x[2:] - x[:-2]) / 2.0
    return weights


def lorentzian_atoms(omega: np.ndarray, centers: np.ndarray, eta: float) -> np.ndarray:
    omega = np.asarray(omega, dtype=np.float64)
    centers = np.asarray(centers, dtype=np.float64)
    half = 0.5 * float(eta)
    return (half / np.pi) / ((omega[None, :] - centers[:, None]) ** 2 + half**2)


def reconstruct_spectrum(omega: np.ndarray, e: np.ndarray, b: np.ndarray, eta: float) -> np.ndarray:
    e = np.asarray(e, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if e.ndim == 1:
        atoms = lorentzian_atoms(omega, e, eta)
        return b @ atoms
    if e.ndim != 2:
        raise ValueError(f"expected 1D or 2D centers, got shape {e.shape}")
    half = 0.5 * float(eta)
    omega = np.asarray(omega, dtype=np.float64)
    numerator = b[:, :, None] * (half / np.pi)
    denominator = (omega[None, None, :] - e[:, :, None]) ** 2 + half**2
    return np.sum(numerator / denominator, axis=1)


def rmse(a: np.ndarray, b: np.ndarray, axis=None) -> np.ndarray:
    return np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2, axis=axis))


def coordinate_smoothness(edges: Sequence[tuple[int, int]], e: np.ndarray, b: np.ndarray) -> dict[str, float]:
    if not edges:
        return {
            "mean_neighbor_E_l2": float("nan"),
            "mean_neighbor_sqrtB_l2": float("nan"),
            "mean_neighbor_E_abs": float("nan"),
            "mean_neighbor_sqrtB_abs": float("nan"),
        }
    e_l2 = []
    b_l2 = []
    e_abs = []
    b_abs = []
    sqrt_b = np.sqrt(np.maximum(b, 0.0))
    for a, c in edges:
        de = e[a] - e[c]
        db = sqrt_b[a] - sqrt_b[c]
        e_l2.append(float(np.linalg.norm(de)))
        b_l2.append(float(np.linalg.norm(db)))
        e_abs.append(float(np.mean(np.abs(de))))
        b_abs.append(float(np.mean(np.abs(db))))
    return {
        "mean_neighbor_E_l2": float(np.mean(e_l2)),
        "mean_neighbor_sqrtB_l2": float(np.mean(b_l2)),
        "mean_neighbor_E_abs": float(np.mean(e_abs)),
        "mean_neighbor_sqrtB_abs": float(np.mean(b_abs)),
    }


def spectrum_metrics(s_true: np.ndarray, s_pred: np.ndarray) -> dict[str, float]:
    per = rmse(s_true, s_pred, axis=1)
    return {
        "rmse_mean": float(np.mean(per)),
        "rmse_median": float(np.median(per)),
        "rmse_p90": float(np.quantile(per, 0.90)),
        "rmse_max": float(np.max(per)),
    }


def central_index(alpha: np.ndarray) -> int:
    mins = np.min(alpha, axis=0)
    maxs = np.max(alpha, axis=0)
    center = 0.5 * (mins + maxs)
    return int(np.argmin(np.sum((alpha - center[None, :]) ** 2, axis=1)))


def build_neighbor_edges(alpha: np.ndarray, k: int = 6, include_grid: bool = True) -> list[tuple[int, int]]:
    alpha = np.asarray(alpha, dtype=np.float64)
    n_samples = alpha.shape[0]
    edges: set[tuple[int, int]] = set()
    d2 = np.sum((alpha[:, None, :] - alpha[None, :, :]) ** 2, axis=-1)
    for i in range(n_samples):
        order = np.argsort(d2[i])
        for j in order[1 : min(n_samples, int(k) + 1)]:
            a, b = sorted((i, int(j)))
            edges.add((a, b))

    if include_grid and alpha.shape[1] == 2:
        rounded = np.round(alpha, decimals=10)
        for dim in range(2):
            other = 1 - dim
            for val in np.unique(rounded[:, other]):
                idx = np.where(rounded[:, other] == val)[0]
                idx = idx[np.argsort(rounded[idx, dim])]
                for a, b in zip(idx[:-1], idx[1:]):
                    edges.add(tuple(sorted((int(a), int(b)))))
    return sorted(edges)


def edge_weight(alpha: np.ndarray, a: int, b: int) -> float:
    return float(np.linalg.norm(alpha[a] - alpha[b]))


def mst_parent_edges(alpha: np.ndarray, edges: Sequence[tuple[int, int]], root: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]], dict[int, int]]:
    n_samples = alpha.shape[0]
    rows = []
    cols = []
    data = []
    for a, b in edges:
        w = edge_weight(alpha, a, b)
        rows.extend([a, b])
        cols.extend([b, a])
        data.extend([w, w])
    mat = csr_matrix((data, (rows, cols)), shape=(n_samples, n_samples))
    tree = minimum_spanning_tree(mat).toarray()
    tree_edges = {tuple(sorted((int(a), int(b)))) for a, b in np.argwhere(tree > 0)}

    adjacency: dict[int, list[int]] = {i: [] for i in range(n_samples)}
    for a, b in tree_edges:
        adjacency[a].append(b)
        adjacency[b].append(a)

    parent = {root: -1}
    directed = []
    queue = [root]
    while queue:
        node = queue.pop(0)
        for nbr in adjacency[node]:
            if nbr in parent:
                continue
            parent[nbr] = node
            directed.append((node, nbr))
            queue.append(nbr)

    non_tree = [edge for edge in edges if tuple(sorted(edge)) not in tree_edges]
    return directed, non_tree, parent


def inverse_permutation(perm: np.ndarray) -> np.ndarray:
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(perm))
    return inv


def apply_orders(e_raw: np.ndarray, b_raw: np.ndarray, orders: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rows = np.arange(e_raw.shape[0])[:, None]
    return e_raw[rows, orders], b_raw[rows, orders]


def write_csv(path: str | Path, rows: Sequence[dict[str, object]], columns: Sequence[str] | None = None) -> None:
    path = Path(path)
    if not rows:
        path.write_text("")
        return
    columns = tuple(columns or rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_metrics_text(path: str | Path, metrics: dict[str, object]) -> None:
    lines = [f"{key}: {value}" for key, value in metrics.items()]
    Path(path).write_text("\n".join(lines) + "\n")


def nonoverlapping_swap_masks(indices: Iterable[int]) -> list[tuple[int, ...]]:
    ordered = sorted(set(int(i) for i in indices))
    out: list[tuple[int, ...]] = [()]

    def rec(pos: int, chosen: list[int]) -> None:
        while pos < len(ordered) and chosen and ordered[pos] <= chosen[-1] + 1:
            pos += 1
        for k in range(pos, len(ordered)):
            chosen.append(ordered[k])
            out.append(tuple(chosen))
            rec(k + 1, chosen)
            chosen.pop()

    rec(0, [])
    return out
