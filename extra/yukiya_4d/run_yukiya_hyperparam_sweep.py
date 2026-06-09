#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


STRENGTH_REGEX = r"strength_(?P<p1>-?[0-9.]+)_(?P<p2>-?[0-9.]+)_(?P<p3>-?[0-9.]+)_(?P<p4>-?[0-9.]+)\.out"


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[2]


def load_configs(path: Path) -> list[dict]:
    with path.open() as f:
        configs = json.load(f)
    if not isinstance(configs, list):
        raise ValueError(f"Expected a list of configs in {path}")
    return configs


def build_command(repo_root: Path, cfg: dict, run_dir: Path, args: argparse.Namespace) -> list[str]:
    return [
        args.python_exe,
        "-u",
        str(repo_root / "Dipole_polarizability/src/main_gpt2.py"),
        "--strength-dir", str(repo_root / "extra_docs/yukiya_4d/total_strength"),
        "--alphaD-dir", str(repo_root / "extra_docs/yukiya_4d/total_alphaD"),
        "--strength-regex", STRENGTH_REGEX,
        "--n", str(cfg["n"]),
        "--retain", str(cfg["retain"]),
        "--fold", str(cfg["fold"]),
        "--ansatz", cfg["ansatz"],
        "--width-model", cfg["width_model"],
        "--w-strength", str(args.w_strength),
        "--w-alphaD", str(args.w_alphaD),
        "--learning-rate", str(cfg["learning_rate"]),
        "--n-restarts", str(args.n_restarts),
        "--seed0", str(args.seed0),
        "--num-iter", str(args.num_iter),
        "--print-every", str(args.print_every),
        "--plots", args.plots,
        "--save-dir", str(run_dir),
    ]


def main() -> int:
    repo_root = repo_root_from_script()
    default_python = Path("/Users/laurenjin/envs/smlr-frib/bin/python")
    default_python_exe = str(default_python) if default_python.exists() else sys.executable

    parser = argparse.ArgumentParser(description="Run the Yukiya Ca48 GT K1 strength-function hyperparameter sweep.")
    parser.add_argument("--configs", type=Path, default=repo_root / "extra_docs/yukiya_4d/yukiya_sweep_configs.json")
    parser.add_argument("--sweep-dir", type=Path, default=repo_root / "extra_docs/yukiya_4d/runs_smlr/yukiya_strength_sweep")
    parser.add_argument("--python-exe", default=default_python_exe)
    parser.add_argument("--num-iter", type=int, default=4000)
    parser.add_argument("--print-every", type=int, default=500)
    parser.add_argument("--n-restarts", type=int, default=1)
    parser.add_argument("--seed0", type=int, default=420)
    parser.add_argument("--w-strength", type=float, default=1.0)
    parser.add_argument("--w-alphaD", type=float, default=0.0)
    parser.add_argument("--plots", choices=["none", "save"], default="save")
    parser.add_argument("--resume", action="store_true", help="Skip configs with an existing run_summary.json.")
    args = parser.parse_args()

    configs = load_configs(args.configs)
    args.sweep_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = args.sweep_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    index_path = args.sweep_dir / "sweep_index.json"
    with index_path.open("w") as f:
        json.dump({"configs": configs, "args": vars(args)}, f, indent=2, default=str)

    print(f"Running {len(configs)} Yukiya Ca48 strength configs")
    print(f"Sweep dir: {args.sweep_dir}")
    print(f"Python: {args.python_exe}")

    failures = []
    for i, cfg in enumerate(configs, start=1):
        name = cfg["name"]
        run_dir = args.sweep_dir / name
        run_dir.mkdir(parents=True, exist_ok=True)
        summary_path = run_dir / "run_summary.json"
        log_path = logs_dir / f"{i:02d}_{name}.log"

        if args.resume and summary_path.exists():
            print(f"[{i}/{len(configs)}] SKIP {name}: existing {summary_path}")
            continue

        command = build_command(repo_root, cfg, run_dir, args)
        with (run_dir / "command.txt").open("w") as f:
            f.write(" ".join(command) + "\n")

        print(f"[{i}/{len(configs)}] START {name}")
        started = time.time()
        with log_path.open("w") as log:
            log.write(f"Command: {' '.join(command)}\n\n")
            log.flush()
            proc = subprocess.run(command, cwd=repo_root, stdout=log, stderr=subprocess.STDOUT)

        elapsed = time.time() - started
        if proc.returncode == 0:
            print(f"[{i}/{len(configs)}] DONE {name} in {elapsed / 60.0:.1f} min")
        else:
            print(f"[{i}/{len(configs)}] FAIL {name} rc={proc.returncode} after {elapsed / 60.0:.1f} min")
            failures.append({"name": name, "returncode": proc.returncode, "log": str(log_path)})

    if failures:
        with (args.sweep_dir / "failures.json").open("w") as f:
            json.dump(failures, f, indent=2)
        print(f"Completed with {len(failures)} failures")
        return 1

    print("Sweep complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
