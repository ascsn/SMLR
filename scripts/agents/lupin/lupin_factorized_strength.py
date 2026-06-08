#!/usr/bin/env python
"""Lupin: identity-free factorized strength coordinates."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import nnls

ROOT = Path(__file__).resolve().parents[3]
COMMON_DIR = ROOT / "scripts" / "agents" / "shared"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from snapshot_labeling_utils import load_snapshots, lorentzian_atoms, spectrum_metrics, write_metrics_text


DEFAULT_SNAPSHOT = ROOT / "tests" / "cache" / "yb_lorentzian_compression" / "snapshots_nstar.npz"
DEFAULT_OUTPUT = ROOT / "tests" / "cache" / "yb_lorentzian_compression" / "agents" / "lupin"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build identity-free reduced coordinates for strength functions.")
    parser.add_argument("--snapshots", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--method", choices=("dictionary", "nmf"), default="dictionary")
    parser.add_argument("--n-atoms", type=int, default=80)
    parser.add_argument("--eta", type=float, default=None, help="Dictionary Lorentzian width; defaults to snapshot eta.")
    parser.add_argument("--poly-degree", type=int, choices=(1, 2), default=2)
    parser.add_argument("--sample-limit", type=int, default=None)
    parser.add_argument("--nmf-max-iter", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def normalize_rows(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scale = np.maximum(np.linalg.norm(v, axis=1), 1e-12)
    return v / scale[:, None], scale


def design_matrix(alpha: np.ndarray, degree: int) -> tuple[np.ndarray, list[str]]:
    alpha = np.asarray(alpha, dtype=np.float64)
    cols = [np.ones(alpha.shape[0])]
    names = ["1"]
    for j in range(alpha.shape[1]):
        cols.append(alpha[:, j])
        names.append(f"a{j + 1}")
    if degree >= 2:
        for j in range(alpha.shape[1]):
            for k in range(j, alpha.shape[1]):
                cols.append(alpha[:, j] * alpha[:, k])
                names.append(f"a{j + 1}*a{k + 1}")
    return np.stack(cols, axis=1), names


def fit_factor_dynamics(alpha: np.ndarray, u: np.ndarray, degree: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    x, names = design_matrix(alpha, degree)
    coeff, *_ = np.linalg.lstsq(x, u, rcond=None)
    return x @ coeff, coeff, names


def dictionary_factorization(data, n_atoms: int, eta: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    e_grid = np.linspace(float(data.omega.min()), float(data.omega.max()), int(n_atoms))
    v = lorentzian_atoms(data.omega, e_grid, eta)
    v, row_scale = normalize_rows(v)
    u = np.empty((data.s_true.shape[0], int(n_atoms)), dtype=np.float64)
    a = v.T
    for i, y in enumerate(data.s_true):
        u[i], _ = nnls(a, y)
    # Move row normalization back into U so V remains the physical atom shape up to scaling.
    u = u / row_scale[None, :]
    v = v * row_scale[:, None]
    return u, v, e_grid


def nmf_factorization(data, n_atoms: int, seed: int, max_iter: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        from sklearn.decomposition import NMF
    except ModuleNotFoundError as exc:
        raise RuntimeError("scikit-learn is not installed; use --method dictionary or install sklearn.") from exc

    model = NMF(
        n_components=int(n_atoms),
        init="nndsvda",
        max_iter=int(max_iter),
        random_state=int(seed),
        solver="cd",
        beta_loss="frobenius",
    )
    u = model.fit_transform(np.maximum(data.s_true, 0.0))
    v = model.components_
    pseudo_centers = data.omega[np.argmax(v, axis=1)]
    return u, v, pseudo_centers


def main() -> None:
    args = parse_args()
    data = load_snapshots(args.snapshots, sample_limit=args.sample_limit)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    eta = float(data.eta if args.eta is None else args.eta)
    if args.method == "dictionary":
        u, v, centers = dictionary_factorization(data, args.n_atoms, eta)
    else:
        u, v, centers = nmf_factorization(data, args.n_atoms, args.seed, args.nmf_max_iter)

    s_reconstructed = u @ v
    u_hat, coeff, basis_names = fit_factor_dynamics(data.alpha, u, args.poly_degree)
    s_dynamics = np.maximum(u_hat, 0.0) @ v if args.method in {"dictionary", "nmf"} else u_hat @ v

    reconstruction_per = np.sqrt(np.mean((s_reconstructed - data.s_true) ** 2, axis=1))
    dynamics_per = np.sqrt(np.mean((s_dynamics - data.s_true) ** 2, axis=1))
    factor_per = np.sqrt(np.mean((u_hat - u) ** 2, axis=1))
    metrics = {
        "method": args.method,
        "n_atoms": args.n_atoms,
        "eta": eta,
        "poly_degree": args.poly_degree,
        **{f"reconstruction_{k}": v0 for k, v0 in spectrum_metrics(data.s_true, s_reconstructed).items()},
        **{f"dynamics_{k}": v0 for k, v0 in spectrum_metrics(data.s_true, s_dynamics).items()},
        "factor_rmse_mean": float(np.mean(factor_per)),
        "factor_rmse_median": float(np.median(factor_per)),
        "factor_rmse_p90": float(np.quantile(factor_per, 0.90)),
    }

    np.savez_compressed(
        args.output_dir / f"lupin_{args.method}_factors.npz",
        alpha_points=data.alpha,
        param_names=data.param_names,
        sample_ids=data.sample_ids,
        omega=data.omega,
        method=np.asarray(args.method),
        eta=np.asarray(eta),
        centers=centers,
        U=u,
        V=v,
        S_reconstructed=s_reconstructed,
        reconstruction_rmse=reconstruction_per,
        U_dynamics=u_hat,
        dynamics_coefficients=coeff,
        dynamics_basis_names=np.asarray(basis_names),
        S_dynamics=s_dynamics,
        dynamics_rmse=dynamics_per,
    )
    write_metrics_text(args.output_dir / f"lupin_{args.method}_metrics.txt", metrics)
    print(f"Saved Lupin {args.method} factors to {args.output_dir / f'lupin_{args.method}_factors.npz'}")
    print(f"Reconstruction RMSE median: {metrics['reconstruction_rmse_median']:.4g}")
    print(f"Polynomial dynamics RMSE median: {metrics['dynamics_rmse_median']:.4g}")


if __name__ == "__main__":
    main()
