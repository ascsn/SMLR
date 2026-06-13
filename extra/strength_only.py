from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="smlr train strength-only",
        description="Stage a generic strength-only LRT run.",
    )
    parser.add_argument("--data-dir", required=True, help="Directory containing strength files")
    parser.add_argument("--regex", default=r"strength_(?P<p1>[0-9.]+)_(?P<p2>[0-9.]+)\.out")
    parser.add_argument("--output-dir", default=None, help="Override spec output directory")
    return parser


def run(argv: list[str] | None = None) -> None:
    """Stage the general strength-only trainer."""
    args = build_parser().parse_args(argv)
    output_dir = Path(args.output_dir or "results/strength_only")
    output_dir.mkdir(parents=True, exist_ok=True)
    plan = {
        "data_dir": args.data_dir,
        "filename_regex": args.regex,
        "output_dir": str(output_dir),
        "ready_for_training": False,
        "next_step": "Implement native generic strength-only optimization after paper trainers are migrated.",
    }
    (output_dir / "strength_only_run_plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True))
    print(json.dumps(plan, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> None:
    run(argv)
