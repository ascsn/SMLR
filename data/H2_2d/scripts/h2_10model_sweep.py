#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/Users/laurenjin/envs/smlr-frib/bin/python")
if not PYTHON.exists():
    PYTHON = Path(sys.executable)

CONFIGS = [
    {"name": "m01_n10_ret1_fold1_linear_exp_affine_lr1e-2", "n": 10, "retain": 1.0, "fold": 1.0, "ansatz": "linear_exp", "width_model": "affine", "lr": 1.0e-2, "seed": 42},
    {"name": "m02_n10_ret1_fold2_linear_exp_constant_lr1e-2", "n": 10, "retain": 1.0, "fold": 2.0, "ansatz": "linear_exp", "width_model": "constant", "lr": 1.0e-2, "seed": 43},
    {"name": "m03_n13_ret1_fold1_linear_exp_affine_lr1e-2", "n": 13, "retain": 1.0, "fold": 1.0, "ansatz": "linear_exp", "width_model": "affine", "lr": 1.0e-2, "seed": 44},
    {"name": "m04_n13_ret1_fold2_linear_exp_affine_lr1e-2", "n": 13, "retain": 1.0, "fold": 2.0, "ansatz": "linear_exp", "width_model": "affine", "lr": 1.0e-2, "seed": 45},
    {"name": "m05_n13_ret075_fold2_linear_exp_affine_lr5e-3", "n": 13, "retain": 0.75, "fold": 2.0, "ansatz": "linear_exp", "width_model": "affine", "lr": 5.0e-3, "seed": 46},
    {"name": "m06_n16_ret1_fold1_linear_exp_affine_lr5e-3", "n": 16, "retain": 1.0, "fold": 1.0, "ansatz": "linear_exp", "width_model": "affine", "lr": 5.0e-3, "seed": 47},
    {"name": "m07_n16_ret1_fold2_linear_exp_affine_lr3e-3", "n": 16, "retain": 1.0, "fold": 2.0, "ansatz": "linear_exp", "width_model": "affine", "lr": 3.0e-3, "seed": 48},
    {"name": "m08_n20_ret1_fold2_linear_exp_affine_lr3e-3", "n": 20, "retain": 1.0, "fold": 2.0, "ansatz": "linear_exp", "width_model": "affine", "lr": 3.0e-3, "seed": 49},
    {"name": "m09_n13_ret1_fold2_linear_affine_lr1e-2", "n": 13, "retain": 1.0, "fold": 2.0, "ansatz": "linear", "width_model": "affine", "lr": 1.0e-2, "seed": 50},
    {"name": "m10_n13_ret1_fold2_quadratic_affine_lr3e-3", "n": 13, "retain": 1.0, "fold": 2.0, "ansatz": "quadratic", "width_model": "affine", "lr": 3.0e-3, "seed": 51},
]


def command_for(cfg: dict, run_dir: Path) -> list[str]:
    return [
        str(PYTHON),
        "-u",
        str(ROOT / "emulator_src/main_gpt2.py"),
        "--strength-dir", str(ROOT / "data/total_strength"),
        "--strength-regex", r"strength_(?P<q>[0-9.]+)_(?P<theta>[0-9.]+)\.out",
        "--strength-only",
        "--strength-normalization", "median_l2",
        "--n", str(cfg["n"]),
        "--retain", str(cfg["retain"]),
        "--fold", str(cfg["fold"]),
        "--ansatz", cfg["ansatz"],
        "--width-model", cfg["width_model"],
        "--w-strength", "1.0",
        "--learning-rate", str(cfg["lr"]),
        "--n-restarts", "1",
        "--seed0", str(cfg["seed"]),
        "--num-iter", "30000",
        "--print-every", "1000",
        "--plots", "none",
        "--save-dir", str(run_dir),
    ]


def flatten_result(cfg: dict, run_dir: Path, status: str, elapsed_s: float) -> dict:
    row = dict(cfg)
    row["run_dir"] = str(run_dir)
    row["status"] = status
    row["elapsed_min"] = elapsed_s / 60.0
    summary_path = run_dir / "run_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        metrics = summary.get("global_best_meta", {}).get("train_metrics", {})
        row["global_best_cost"] = summary.get("global_best_cost")
        row["strength_cost"] = metrics.get("strength_cost")
        row["best_seed"] = summary.get("global_best_meta", {}).get("seed")
        row["best_iter"] = summary.get("global_best_meta", {}).get("iter")
    return row


def write_summary(rows: list[dict], path: Path) -> None:
    fields = [
        "name", "status", "strength_cost", "global_best_cost", "best_seed", "best_iter",
        "n", "retain", "fold", "ansatz", "width_model", "lr", "seed", "elapsed_min", "run_dir",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    sweep_dir = ROOT / "results/sweep_runs"
    logs_dir = sweep_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    (sweep_dir / "configs.json").write_text(json.dumps(CONFIGS, indent=2))
    rows = []

    print(f"Root: {ROOT}")
    print(f"Python: {PYTHON}")
    print(f"Training {len(CONFIGS)} H2 strength-only models")
    for i, cfg in enumerate(CONFIGS, start=1):
        run_dir = sweep_dir / cfg["name"]
        run_dir.mkdir(parents=True, exist_ok=True)
        log_path = logs_dir / f"{cfg['name']}.log"
        cmd = command_for(cfg, run_dir)
        (run_dir / "command.txt").write_text(" ".join(cmd) + "\n")

        print(f"[{i:02d}/{len(CONFIGS)}] START {cfg['name']}", flush=True)
        start = time.time()
        with log_path.open("w") as log:
            log.write(" ".join(cmd) + "\n\n")
            log.flush()
            proc = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        elapsed = time.time() - start
        status = "done" if proc.returncode == 0 else f"failed:{proc.returncode}"
        row = flatten_result(cfg, run_dir, status, elapsed)
        rows.append(row)
        rows.sort(key=lambda r: float("inf") if r.get("strength_cost") is None else float(r["strength_cost"]))
        write_summary(rows, sweep_dir / "sweep_summary.csv")
        print(f"[{i:02d}/{len(CONFIGS)}] {status.upper()} {cfg['name']} in {elapsed / 60.0:.1f} min; strength_cost={row.get('strength_cost')}", flush=True)

    print("Top models:")
    for row in rows[:5]:
        print(f"  {row.get('strength_cost')}  {row['name']}")
    return 0 if all(row["status"] == "done" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
