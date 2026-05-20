from __future__ import annotations

import argparse
import json
import sys

from .specs import h2_2d_strength_spec, paper_beta_em1_spec, paper_beta_em2_spec, paper_dipole_em1_spec
from .training.paper import run_beta_paper_em1, run_beta_paper_em2, run_dipole_paper_em1
from .validation import validate_paper_beta_data, validate_paper_dipole_data, validate_strength_grid
from .diagnostics.entrypoints import run_beta as run_beta_diagnostics
from .diagnostics.entrypoints import run_dipole_paper_em1 as run_dipole_paper_em1_diagnostics


def _print_report(report, *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
        return
    status = "ok" if report.ok else "failed"
    print(f"validation: {status}")
    print(f"files_checked: {report.files_checked}")
    if report.parameter_names:
        print(f"parameters: {', '.join(report.parameter_names)}")
    if report.points:
        print(f"grid_points: {len(report.points)}")
    for issue in report.issues:
        where = f" ({issue.path})" if issue.path else ""
        print(f"{issue.level}: {issue.message}{where}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="smlr", description="SMLR package utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="validate emulator input data")
    validate_sub = validate.add_subparsers(dest="kind", required=True)

    beta = validate_sub.add_parser("beta-paper", help="validate paper beta-decay Ni-80 data")
    beta.add_argument("--data-dir", default="beta_decay_data_Ni_80")
    beta.add_argument("--nucnam", default="Ni_80")
    beta.add_argument("--json", action="store_true")

    dipole = validate_sub.add_parser("dipole-paper", help="validate paper dipole strength data")
    dipole.add_argument("--strength-dir", default="dipoles_data_all/total_strength")
    dipole.add_argument("--json", action="store_true")

    generic = validate_sub.add_parser("strength-grid", help="validate a generic strength-function grid")
    generic.add_argument("data_dir")
    generic.add_argument("--regex", required=True, help="filename regex with named parameter groups")
    generic.add_argument("--parameter", action="append", dest="parameters", help="parameter group name, in grid order")
    generic.add_argument("--min-files", type=int, default=1)
    generic.add_argument("--min-columns", type=int, default=2)
    generic.add_argument("--allow-negative-strength", action="store_true")
    generic.add_argument("--no-rectangular-grid", action="store_true")
    generic.add_argument("--json", action="store_true")

    specs = subparsers.add_parser("spec", help="print built-in emulator run specs")
    specs_sub = specs.add_subparsers(dest="kind", required=True)
    for name in ("dipole-paper-em1", "beta-paper-em1", "beta-paper-em2", "h2-2d-strength"):
        specs_sub.add_parser(name)

    train = subparsers.add_parser("train", help="run package training entry points")
    train_sub = train.add_subparsers(dest="kind", required=True)
    for name in ("dipole-paper-em1", "beta-paper-em1", "beta-paper-em2"):
        sub = train_sub.add_parser(name, help=f"run {name} training")
        sub.add_argument("training_args", nargs=argparse.REMAINDER, help="arguments forwarded to the trainer")

    diagnose = subparsers.add_parser("diagnose", help="run package diagnostic entry points")
    diagnose_sub = diagnose.add_subparsers(dest="kind", required=True)
    for name in ("dipole-paper-em1", "beta-paper"):
        sub = diagnose_sub.add_parser(name, help=f"run {name} diagnostics")
        sub.add_argument("diagnostic_args", nargs=argparse.REMAINDER, help="arguments forwarded to diagnostics")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "validate" and args.kind == "beta-paper":
        report = validate_paper_beta_data(args.data_dir, args.nucnam)
        _print_report(report, as_json=args.json)
        return 0 if report.ok else 1
    if args.command == "validate" and args.kind == "dipole-paper":
        report = validate_paper_dipole_data(args.strength_dir)
        _print_report(report, as_json=args.json)
        return 0 if report.ok else 1
    if args.command == "validate" and args.kind == "strength-grid":
        parameter_names = tuple(args.parameters) if args.parameters else None
        report = validate_strength_grid(
            args.data_dir,
            args.regex,
            parameter_names=parameter_names,
            min_files=args.min_files,
            min_columns=args.min_columns,
            require_rectangular_grid=not args.no_rectangular_grid,
            allow_negative_strength=args.allow_negative_strength,
        )
        _print_report(report, as_json=args.json)
        return 0 if report.ok else 1
    if args.command == "spec":
        factories = {
            "dipole-paper-em1": paper_dipole_em1_spec,
            "beta-paper-em1": paper_beta_em1_spec,
            "beta-paper-em2": paper_beta_em2_spec,
            "h2-2d-strength": h2_2d_strength_spec,
        }
        print(json.dumps(factories[args.kind]().to_dict(), indent=2, sort_keys=True))
        return 0
    if args.command == "train":
        runners = {
            "dipole-paper-em1": run_dipole_paper_em1,
            "beta-paper-em1": run_beta_paper_em1,
            "beta-paper-em2": run_beta_paper_em2,
        }
        forwarded = list(args.training_args)
        if forwarded and forwarded[0] == "--":
            forwarded = forwarded[1:]
        runners[args.kind](forwarded)
        return 0
    if args.command == "diagnose":
        runners = {
            "dipole-paper-em1": run_dipole_paper_em1_diagnostics,
            "beta-paper": run_beta_diagnostics,
        }
        forwarded = list(args.diagnostic_args)
        if forwarded and forwarded[0] == "--":
            forwarded = forwarded[1:]
        runners[args.kind](forwarded)
        return 0

    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    sys.exit(main())
