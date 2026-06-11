#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_csv(value: str, cast):
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def build_em1_commands(args):
    for n, retain, weight, seed0 in itertools.product(args.em1_n, args.retain, args.weight, args.seed0):
        tag = f"n{n}_retain{retain}_w{weight}_seed{seed0}".replace(".", "p")
        common = [
            "--n", str(n),
            "--retain", str(retain),
            "--weight", str(weight),
            "--n-restarts", str(args.n_restarts),
            "--seed0", str(seed0),
            "--num-iter", str(args.num_iter),
            "--print-every", str(args.print_every),
            "--plots", args.plots,
        ]
        yield {
            "label": f"em1_original_{tag}",
            "cmd": [args.python, "-m", "Beta_decay.main", *common, "--save-dir", str(args.out_dir / "original" / "em1" / tag)],
        }
        yield {
            "label": f"em1_package_{tag}",
            "cmd": [args.python, "-m", "Beta_decay_package.src.main_gpt2", *common, "--save-dir", str(args.out_dir / "package" / "em1" / tag)],
        }


def build_em2_commands(args):
    for n, seed0 in itertools.product(args.em2_n, args.seed0):
        tag = f"n{n}_seed{seed0}"
        common = [
            "--n", str(n),
            "--n-restarts", str(args.n_restarts),
            "--seed0", str(seed0),
            "--num-iter", str(args.num_iter),
            "--print-every", str(args.print_every),
            "--plots", args.plots,
        ]
        yield {
            "label": f"em2_original_{tag}",
            "cmd": [args.python, "-m", "Beta_decay.main_only_HL", *common, "--save-dir", str(args.out_dir / "original" / "em2" / tag)],
        }
        yield {
            "label": f"em2_package_{tag}",
            "cmd": [args.python, "-m", "Beta_decay_package.src.main_only_HL_gpt", *common, "--save-dir", str(args.out_dir / "package" / "em2" / tag)],
        }


def parse_args():
    p = argparse.ArgumentParser(description="Run matched original/package beta-decay emulator sweeps.")
    p.add_argument("--out-dir", type=Path, default=Path("Beta_decay_package/sweeps/beta_compare"))
    p.add_argument("--which", choices=["em1", "em2", "both"], default="both")
    p.add_argument("--em1-n", type=lambda s: parse_csv(s, int), default=[8, 13])
    p.add_argument("--em2-n", type=lambda s: parse_csv(s, int), default=[6, 9])
    p.add_argument("--retain", type=lambda s: parse_csv(s, float), default=[0.9])
    p.add_argument("--weight", type=lambda s: parse_csv(s, float), default=[1.0])
    p.add_argument("--seed0", type=lambda s: parse_csv(s, int), default=[42])
    p.add_argument("--n-restarts", type=int, default=1)
    p.add_argument("--num-iter", type=int, default=20000)
    p.add_argument("--print-every", type=int, default=1000)
    p.add_argument("--plots", choices=["none", "save"], default="save")
    p.add_argument("--python", default=sys.executable, help="Python executable used for launched training jobs.")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    args.out_dir = args.out_dir.resolve()
    os.chdir(REPO_ROOT)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    if args.which in {"em1", "both"}:
        jobs.extend(build_em1_commands(args))
    if args.which in {"em2", "both"}:
        jobs.extend(build_em2_commands(args))

    manifest_path = args.out_dir / "commands.json"
    with open(manifest_path, "w") as f:
        json.dump(jobs, f, indent=2)

    print(f"Wrote command manifest: {manifest_path}")
    for job in jobs:
        print("\n==", job["label"], "==")
        print(" ".join(job["cmd"]))
        if not args.dry_run:
            subprocess.run(job["cmd"], cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
