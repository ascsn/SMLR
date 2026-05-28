from __future__ import annotations

import argparse
import json
import csv
import sys
from pathlib import Path

import numpy as np

from .serialization import load_emulator, package_existing_emulator, spec_from_selector
from .specs import BUILTIN_SPECS, get_builtin_spec, h2_2d_strength_spec, load_run_spec, paper_beta_em1_spec, paper_beta_em2_spec, paper_dipole_em1_spec, save_run_spec
from .training import strength_only
from .training.paper import run_beta_paper_em1, run_beta_paper_em2, run_dipole_paper_em1
from .validation import (
    GENERIC_2D_PARAMETER_NAMES,
    GENERIC_2D_STRENGTH_REGEX,
    validate_paper_beta_data,
    validate_paper_dipole_data,
    validate_strength_grid,
)
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
    generic.add_argument(
        "--regex",
        default=GENERIC_2D_STRENGTH_REGEX,
        help="filename regex with named parameter groups; defaults to strength_<p1>_<p2>.out",
    )
    generic.add_argument("--parameter", action="append", dest="parameters", help="parameter group name, in grid order")
    generic.add_argument("--min-files", type=int, default=1)
    generic.add_argument("--min-columns", type=int, default=2)
    generic.add_argument("--max-columns", type=int, default=2)
    generic.add_argument("--allow-negative-strength", action="store_true")
    generic.add_argument("--no-rectangular-grid", action="store_true")
    generic.add_argument("--json", action="store_true")

    specs = subparsers.add_parser("spec", help="print built-in emulator run specs")
    specs_sub = specs.add_subparsers(dest="kind", required=True)
    for name in ("dipole-paper-em1", "beta-paper-em1", "beta-paper-em2", "h2-2d-strength"):
        specs_sub.add_parser(name)
    spec_file = specs_sub.add_parser("file", help="print a spec JSON file")
    spec_file.add_argument("path")
    spec_save = specs_sub.add_parser("save", help="save a built-in spec as JSON")
    spec_save.add_argument("name", choices=sorted(BUILTIN_SPECS))
    spec_save.add_argument("path")

    train = subparsers.add_parser("train", help="run package training entry points")
    train_sub = train.add_subparsers(dest="kind", required=True)
    for name in ("dipole-paper-em1", "beta-paper-em1", "beta-paper-em2"):
        sub = train_sub.add_parser(name, help=f"run {name} training")
        sub.add_argument("training_args", nargs=argparse.REMAINDER, help="arguments forwarded to the trainer")
    strength = train_sub.add_parser("strength-only", help="validate/stage generic strength-only training from a spec")
    strength.add_argument("training_args", nargs=argparse.REMAINDER, help="arguments forwarded to the trainer")

    diagnose = subparsers.add_parser("diagnose", help="run package diagnostic entry points")
    diagnose_sub = diagnose.add_subparsers(dest="kind", required=True)
    for name in ("dipole-paper-em1", "beta-paper"):
        sub = diagnose_sub.add_parser(name, help=f"run {name} diagnostics")
        sub.add_argument("diagnostic_args", nargs=argparse.REMAINDER, help="arguments forwarded to diagnostics")

    serialize = subparsers.add_parser("serialize", help="create an emulator.json record for trained params")
    serialize.add_argument("--run-dir", required=True)
    serialize.add_argument("--params-file", required=True)
    serialize.add_argument("--name", required=True)
    serialize.add_argument("--adapter", required=True)
    serialize.add_argument("--spec", default=None, help="built-in spec name or path to spec JSON")
    serialize.add_argument("--retain", type=float, default=None)
    serialize.add_argument("--retention-kind", default="centered")
    serialize.add_argument("--backend", default="tensorflow")
    serialize.add_argument("--metadata-json", default=None)

    predict = subparsers.add_parser("predict", help="run predictions from a serialized emulator")
    predict.add_argument("--emulator", required=True, help="directory containing emulator.json, or path to emulator.json")
    predict.add_argument("--domain", choices=["dipole", "beta-em1", "beta-em2"], required=True)
    predict.add_argument("--points", required=True, help='JSON list of parameter points, e.g. "[[1.0,0.5]]"')
    predict.add_argument("--central-point", default=None, help='Optional JSON point, e.g. "[1.0,0.5]"')
    predict.add_argument("--energy", default=None, help="Optional JSON energy grid for spectra")
    predict.add_argument("--output", default=None, help="Optional CSV/JSON output path")
    predict.add_argument("--json", action="store_true", help="Print JSON instead of a short text summary")

    backend = subparsers.add_parser("backend", help="inspect available optimizer backend names")
    backend_sub = backend.add_subparsers(dest="kind", required=True)
    backend_sub.add_parser("list", help="list known backend names")

    examples = subparsers.add_parser("example", help="list packaged examples")
    examples_sub = examples.add_subparsers(dest="kind", required=True)
    examples_sub.add_parser("list", help="list available example specs/notebooks")

    notebooks = subparsers.add_parser("notebook", help="list package notebooks")
    notebooks_sub = notebooks.add_subparsers(dest="kind", required=True)
    notebooks_sub.add_parser("list", help="list notebooks in the repository")
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
        parameter_names = tuple(args.parameters) if args.parameters else GENERIC_2D_PARAMETER_NAMES
        report = validate_strength_grid(
            args.data_dir,
            args.regex,
            parameter_names=parameter_names,
            min_files=args.min_files,
            min_columns=args.min_columns,
            max_columns=args.max_columns,
            require_rectangular_grid=not args.no_rectangular_grid,
            allow_negative_strength=args.allow_negative_strength,
        )
        _print_report(report, as_json=args.json)
        return 0 if report.ok else 1
    if args.command == "spec":
        if args.kind == "file":
            spec = load_run_spec(args.path)
            print(json.dumps(spec.to_dict(), indent=2, sort_keys=True))
            return 0
        if args.kind == "save":
            save_run_spec(get_builtin_spec(args.name), args.path)
            print(f"Saved spec to {args.path}")
            return 0
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
            "strength-only": strength_only.run,
        }
        forwarded = list(args.training_args)
        if forwarded and forwarded[0] == "--":
            forwarded = forwarded[1:]
        runners[args.kind](forwarded)
        return 0
    if args.command == "serialize":
        from .core.retention import RetainedModePolicy

        metadata = json.loads(args.metadata_json) if args.metadata_json else {}
        retained = (
            RetainedModePolicy(kind=args.retention_kind, retain=args.retain)
            if args.retain is not None
            else None
        )
        record = package_existing_emulator(
            args.run_dir,
            params_file=args.params_file,
            name=args.name,
            adapter=args.adapter,
            spec=spec_from_selector(args.spec),
            retained_modes=retained,
            backend=args.backend,
            metadata=metadata,
        )
        print(json.dumps(record.to_dict(), indent=2, sort_keys=True))
        return 0
    if args.command == "predict":
        loaded = load_emulator(args.emulator)
        points = np.asarray(json.loads(args.points), dtype=float)
        central_point = json.loads(args.central_point) if args.central_point else None
        energy = np.asarray(json.loads(args.energy), dtype=float) if args.energy else None
        if args.domain == "dipole":
            result = loaded.predict_dipole(points, central_point=central_point, energy=energy)
        elif args.domain == "beta-em1":
            result = loaded.predict_beta_em1(points, central_point=central_point, energy=energy)
        else:
            result = loaded.predict_beta_em2(points, central_point=central_point)
        _emit_prediction(result, args.output, as_json=args.json)
        return 0
    if args.command == "backend" and args.kind == "list":
        print(json.dumps({
            "optimizers": ["tensorflow", "torch", "jax"],
            "default": "tensorflow",
            "notes": "Backends share the same scalar loss contract; paper trainers still use TensorFlow until their loops are fully migrated.",
        }, indent=2, sort_keys=True))
        return 0
    if args.command == "example" and args.kind == "list":
        print(json.dumps({
            "specs": sorted(BUILTIN_SPECS),
            "recommended_first_non_nuclear": "h2-2d-strength",
        }, indent=2, sort_keys=True))
        return 0
    if args.command == "notebook" and args.kind == "list":
        notebooks = sorted(str(path) for path in Path("notebooks").glob("*.ipynb"))
        print(json.dumps({"notebooks": notebooks}, indent=2, sort_keys=True))
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


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {key: _json_safe(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_json_safe(val) for val in value]
    return value


def _emit_prediction(result, output, *, as_json: bool):
    if output:
        path = Path(output)
        if path.suffix.lower() == ".json":
            path.write_text(json.dumps(_json_safe(result), indent=2, sort_keys=True))
        else:
            keys = [key for key, val in result.items() if np.asarray(val).ndim == 1]
            with path.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["idx", *keys])
                n = len(np.asarray(result[keys[0]])) if keys else 0
                for i in range(n):
                    writer.writerow([i, *[np.asarray(result[key])[i] for key in keys]])
        print(f"Saved predictions to {path}")
        return
    if as_json:
        print(json.dumps(_json_safe(result), indent=2, sort_keys=True))
        return
    for key, value in result.items():
        arr = np.asarray(value)
        print(f"{key}: shape={arr.shape}")


if __name__ == "__main__":
    sys.exit(main())
