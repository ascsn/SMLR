#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[2]


def load_json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def main() -> int:
    repo_root = repo_root_from_script()
    parser = argparse.ArgumentParser(description="Summarize H2 strength-only sweep results.")
    parser.add_argument("--sweep-dir", type=Path, default=repo_root / "Dipole_polarizability/aaron_H2/sweeps/h2_strength_sweep")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--top", type=int, default=15)
    args = parser.parse_args()

    rows = []
    for run_dir in sorted(args.sweep_dir.iterdir()):
        if not run_dir.is_dir() or run_dir.name == "logs":
            continue
        summary_path = run_dir / "run_summary.json"
        if not summary_path.exists():
            rows.append({"name": run_dir.name, "status": "missing"})
            continue

        summary = load_json(summary_path)
        train_metrics = summary.get("global_best_meta", {}).get("train_metrics") or {}
        cfg = summary.get("config") or {}
        cli_args = summary.get("args") or {}
        rows.append({
            "name": run_dir.name,
            "status": "done",
            "strength_cost": train_metrics.get("strength_cost"),
            "cost": train_metrics.get("cost"),
            "global_best_cost": summary.get("global_best_cost"),
            "seed": summary.get("global_best_meta", {}).get("seed"),
            "best_iter": summary.get("global_best_meta", {}).get("iter"),
            "n": cfg.get("n"),
            "retain": cli_args.get("retain"),
            "fold": cli_args.get("fold"),
            "ansatz": cfg.get("ansatz"),
            "width_model": cfg.get("width_model"),
            "learning_rate": cli_args.get("learning_rate"),
            "run_dir": str(run_dir),
        })

    rows.sort(key=lambda row: (row["status"] != "done", float("inf") if row.get("strength_cost") is None else row["strength_cost"]))

    output = args.output or (args.sweep_dir / "sweep_summary.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "name", "status", "strength_cost", "cost", "global_best_cost",
        "seed", "best_iter", "n", "retain", "fold", "ansatz",
        "width_model", "learning_rate", "run_dir",
    ]
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {output}")
    print("Top runs:")
    for row in rows[:args.top]:
        print(
            f"{row['status']:>7}  strength_cost={row.get('strength_cost')}  "
            f"n={row.get('n')} retain={row.get('retain')} fold={row.get('fold')} "
            f"ansatz={row.get('ansatz')} lr={row.get('learning_rate')}  {row['name']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
