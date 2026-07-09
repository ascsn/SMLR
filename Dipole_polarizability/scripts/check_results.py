#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import helper_gpt as helper_gpt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a Dipole_polarizability run directory against the current helper_gpt layout."
    )
    parser.add_argument("--run-dir", default="runs_em1")
    parser.add_argument("--strength-dir", default="data/nuclear/160Yb_2d/total_strength")
    parser.add_argument("--alphaD-dir", default="data/nuclear/160Yb_2d/total_alphaD")
    parser.add_argument(
        "--strength-regex",
        default=r"strength_(?P<p2>[0-9.]+)_(?P<p1>[0-9.]+)\.out",
    )
    parser.add_argument("--alphaD-regex", default=None)
    parser.add_argument("--filter-ranges", default='{"p1":[0.4,1.8],"p2":[1.5,4.0]}')
    parser.add_argument("--n", type=int, default=13)
    parser.add_argument("--ansatz", default="paper_dipole")
    parser.add_argument("--width-model", default="affine")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    params_path = run_dir / "best_params_global.txt"
    summary_path = run_dir / "run_summary.json"

    if not params_path.exists():
        raise FileNotFoundError(f"Missing parameters file: {params_path}")
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing run summary: {summary_path}")

    dataset = helper_gpt.load_dataset(
        strength_dir=args.strength_dir,
        alphaD_dir=args.alphaD_dir,
        strength_regex=args.strength_regex,
        alphaD_regex=args.alphaD_regex,
        filter_ranges=args.filter_ranges,
    )
    params = np.loadtxt(params_path).astype(np.float32)
    summary = json.loads(summary_path.read_text())

    config = helper_gpt.AnsatzConfig(
        n=args.n,
        n_params=int(dataset.param_values.shape[1]),
        ansatz=args.ansatz,
        width_model=args.width_model,
    )
    expected = helper_gpt.count_trainable_parameters(config)
    if params.shape != (expected,):
        raise ValueError(f"Expected params shape {(expected,)}, got {params.shape}")
    if not np.isfinite(params).all():
        raise ValueError("Parameters contain non-finite values.")

    print("Run check passed")
    print("  run dir:", run_dir)
    print("  samples:", len(dataset.strengths))
    print("  param names:", dataset.param_names)
    print("  params shape:", params.shape)
    print("  summary samples:", summary.get("n_train_samples", summary.get("n_samples")))


if __name__ == "__main__":
    main()
