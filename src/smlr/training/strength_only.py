from __future__ import annotations

import argparse
import json
from pathlib import Path

from smlr.specs import load_run_spec


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="smlr train strength-only",
        description="Validate and stage a generic strength-only LRT run spec.",
    )
    parser.add_argument("--spec", required=True, help="Path to an EmulatorRunSpec JSON/YAML file")
    parser.add_argument("--output-dir", default=None, help="Override spec output directory")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the run plan")
    return parser


def run(argv: list[str] | None = None) -> None:
    """Stage the general strength-only trainer.

    This deliberately stops before optimization for now: the backend-neutral
    objective is the next migration step after the paper trainers are moved out
    of their historical scripts.
    """

    args = build_parser().parse_args(argv)
    spec = load_run_spec(args.spec)
    report = spec.validate_data()
    output_dir = Path(args.output_dir or spec.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plan = {
        "spec": spec.to_dict(),
        "output_dir": str(output_dir),
        "validation": report.to_dict(),
        "ready_for_training": report.ok,
    }
    (output_dir / "strength_only_run_plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True))
    print(json.dumps(plan, indent=2, sort_keys=True))
    if not args.dry_run:
        raise NotImplementedError(
            "Generic strength-only optimization is staged but not yet migrated. "
            "Use --dry-run to validate user data/specs today."
        )


def main(argv: list[str] | None = None) -> None:
    run(argv)
