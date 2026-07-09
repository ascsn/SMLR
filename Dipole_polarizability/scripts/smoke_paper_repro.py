#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
TRAINER = ROOT / "Dipole_polarizability" / "src" / "main_gpt2.py"
DIAGNOSTICS = ROOT / "Dipole_polarizability" / "src" / "diagnostics_general_gpt.py"
STRENGTH_DIR = "data/nuclear/160Yb_2d/total_strength"
ALPHAD_DIR = "data/nuclear/160Yb_2d/total_alphaD"
STRENGTH_REGEX = r"strength_(?P<p2>[0-9.]+)_(?P<p1>[0-9.]+)\.out"
FILTER_RANGES = '{"p1":[0.4,1.8],"p2":[1.5,4.0]}'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast smoke test for the dipole paper reproduction workflow.")
    parser.add_argument("--save-dir", default=None, help="Run directory. Defaults to a temporary directory.")
    parser.add_argument("--num-iter", type=int, default=1)
    parser.add_argument("--plots", choices=["none", "save"], default="save")
    parser.add_argument("--keep-temp", action="store_true", help="Do not delete the temporary run directory.")
    return parser.parse_args()


def run_command(cmd: list[str], save_dir: Path) -> None:
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", str(save_dir / "mpl-cache"))
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def validate_run(save_dir: Path) -> None:
    summary = json.loads((save_dir / "run_summary.json").read_text())
    params = np.loadtxt(save_dir / "best_params_global.txt")
    train = np.loadtxt(save_dir / "train_param_values.txt")
    test = np.loadtxt(save_dir / "test_param_values.txt")

    expected_files = [
        save_dir / "emulator.json",
        save_dir / "seed_1234" / "metrics.json",
        save_dir / "diagnostics" / "summary.txt",
        save_dir / "diagnostics" / "predictions.csv",
    ]
    missing = [str(path) for path in expected_files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing expected smoke outputs: {missing}")
    if summary["n_total_samples"] != 225 or summary["n_train_samples"] != 121 or summary["n_test_samples"] != 104:
        raise ValueError(f"Unexpected sample counts in {save_dir / 'run_summary.json'}: {summary}")
    if params.shape != (330,) or not np.isfinite(params).all():
        raise ValueError(f"Unexpected parameter vector: shape={params.shape}, finite={np.isfinite(params).all()}")
    if train.shape != (121, 2) or test.shape != (104, 2):
        raise ValueError(f"Unexpected train/test shapes: train={train.shape}, test={test.shape}")


def main() -> None:
    args = parse_args()
    temp_ctx = None
    if args.save_dir is None:
        if args.keep_temp:
            save_dir = Path(tempfile.mkdtemp(prefix="smlr_dipole_smoke_"))
        else:
            temp_ctx = tempfile.TemporaryDirectory(prefix="smlr_dipole_smoke_")
            save_dir = Path(temp_ctx.name)
    else:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    trainer_cmd = [
        sys.executable,
        str(TRAINER),
        "--strength-dir", STRENGTH_DIR,
        "--alphaD-dir", ALPHAD_DIR,
        "--strength-regex", STRENGTH_REGEX,
        "--filter-ranges", FILTER_RANGES,
        "--n", "13",
        "--retain", "0.6",
        "--fold", "2.0",
        "--ansatz", "paper_dipole",
        "--width-model", "affine",
        "--w-strength", "100.0",
        "--w-alphaD", "200.0",
        "--w-m1", "0.0",
        "--m1-target", "875.0",
        "--learning-rate", "1e-2",
        "--n-restarts", "1",
        "--seed0", "1234",
        "--num-iter", str(args.num_iter),
        "--print-every", "1",
        "--plots", args.plots,
        "--save-dir", str(save_dir),
    ]
    diagnostics_cmd = [
        sys.executable,
        str(DIAGNOSTICS),
        "--strength-dir", STRENGTH_DIR,
        "--alphaD-dir", ALPHAD_DIR,
        "--strength-regex", STRENGTH_REGEX,
        "--filter-ranges", FILTER_RANGES,
        "--n", "13",
        "--retain", "0.6",
        "--ansatz", "paper_dipole",
        "--width-model", "affine",
        "--max-label-points", "1000",
        "--plots", "save",
        "--save-dir", str(save_dir),
    ]

    try:
        run_command(trainer_cmd, save_dir)
        run_command(diagnostics_cmd, save_dir)
        validate_run(save_dir)
        print("Smoke paper reproduction check passed")
        print("  run dir:", save_dir)
    finally:
        if temp_ctx is not None and not args.keep_temp:
            temp_ctx.cleanup()


if __name__ == "__main__":
    main()
