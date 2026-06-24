#!/usr/bin/env python3
"""Plot EM1 predicted vs true strength spectra from a saved run."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "Beta_decay_package" / "src"))
import helper_gpt as helper  # noqa: E402


def load_dipole_helper():
    helper_path = REPO_ROOT / "Dipole_polarizability" / "src" / "helper_gpt.py"
    spec = importlib.util.spec_from_file_location("dipole_helper_gpt", helper_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load dipole helper from {helper_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def rounded_param_key(values):
    return tuple(round(float(v), 4) for v in values)


def sorted_regex_params(match: re.Match) -> tuple[float, ...]:
    groups = match.groupdict()
    if groups:
        return tuple(float(value) for _, value in sorted(groups.items(), key=lambda item: item[0]))
    return tuple(float(value) for value in match.groups())


def split_filename_params(fname: str) -> tuple[float, ...] | None:
    if not fname.startswith("strength_") or not fname.endswith(".out"):
        return None
    raw = fname[len("strength_"):-len(".out")]
    try:
        return tuple(float(part) for part in raw.split("_"))
    except ValueError:
        return None


def load_strength_dataset(data_dir: Path, filename_regex: str | None = None):
    pattern = re.compile(filename_regex) if filename_regex else None
    combined = []
    for fname in sorted(os.listdir(data_dir)):
        if pattern is not None:
            match = pattern.match(fname)
            params = sorted_regex_params(match) if match else None
        else:
            params = split_filename_params(fname)
        if params is not None:
            combined.append((params, str(data_dir / fname)))
    if not combined:
        raise ValueError(f"No strength_*.out files found in {data_dir}")
    return combined


def split_dataset(combined, train_ratio: float = 0.6, cv_ratio: float = 0.1):
    n_total = len(combined)
    n_train = int(n_total * train_ratio)
    n_cv = int(n_total * cv_ratio)
    return {
        "train": combined[:n_train],
        "validation": combined[n_train:n_train + n_cv],
        "test": combined[n_train + n_cv:],
        "all": combined,
    }


def load_saved_split(run_dir: Path, data_dir: Path, split: str):
    split_files = {
        "train": first_existing(run_dir / "train_set.txt", run_dir / "train_param_values.txt"),
        "validation": run_dir / "validation_set.txt",
        "test": first_existing(run_dir / "test_set.txt", run_dir / "test_param_values.txt"),
    }
    split_path = split_files.get(split)
    if split_path is None or not split_path.is_file():
        return None

    metadata = load_run_metadata(run_dir)
    all_entries = load_strength_dataset(data_dir, metadata.get("strength_regex"))
    by_key = {rounded_param_key(helper.dataset_entry_params(entry)): entry for entry in all_entries}
    entries = []
    for line in split_path.read_text().splitlines():
        if not line.strip():
            continue
        parts = line.split(",") if "," in line else line.split()
        params = tuple(float(v) for v in parts)
        key = rounded_param_key(params)
        if key not in by_key:
            raise ValueError(f"Could not match split row {line!r} to a strength file in {data_dir}")
        entries.append(by_key[key])
    return entries


def first_existing(*paths: Path) -> Path:
    for path in paths:
        if path.is_file():
            return path
    return paths[0]


def load_json(path: Path):
    if path.is_file():
        return json.loads(path.read_text())
    return None


def load_run_metadata(run_dir: Path) -> dict:
    summary = load_json(run_dir / "run_summary.json") or {}
    emulator = load_json(run_dir / "emulator.json") or {}
    spec = emulator.get("spec", {})
    strength_spec = spec.get("strength", {})
    model = summary.get("config") or spec.get("model") or {}
    args = summary.get("args", {})
    central = summary.get("central_point") or spec.get("central_point")
    retain = model.get("retain")

    return {
        "summary": summary,
        "emulator": emulator,
        "strength_dir": args.get("strength_dir") or strength_spec.get("data_dir"),
        "strength_regex": args.get("strength_regex") or strength_spec.get("filename_regex"),
        "model": model,
        "central_point": central,
        "retain": retain,
        "adapter": emulator.get("adapter"),
    }


def infer_data_dir(run_dir: Path, explicit_data_dir: Path | None) -> Path:
    if explicit_data_dir is not None:
        return explicit_data_dir
    metadata = load_run_metadata(run_dir)
    if metadata.get("strength_dir"):
        metadata_dir = Path(metadata["strength_dir"])
        if metadata_dir.is_dir():
            return metadata_dir

    name = run_dir.name
    candidates = []
    if "160Yb" in name:
        candidates.append(REPO_ROOT / "data" / "nuclear" / "160Yb_2d" / "total_strength")
        candidates.append(REPO_ROOT / "data" / "dipole_polarizability_160Yb_2d" / "total_strength")
    if "48Ca" in name:
        candidates.append(REPO_ROOT / "data" / "nuclear" / "48Ca_4d" / "total_strength_K0")
        candidates.append(REPO_ROOT / "data" / "gamow_teller_48Ca_4d" / "total_strength_K0")
    if "80Ni" in name:
        candidates.append(REPO_ROOT / "data" / "nuclear" / "80Ni_2d" / "total_strength")

    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return REPO_ROOT / "data" / "nuclear" / "48Ca_4d" / "total_strength_K0"


def infer_n(num_params: int, num_components: int) -> int:
    for n in range(1, 500):
        expected = 2 * n + num_components * (n * (n + 1) // 2) + num_components + 2
        if expected == num_params:
            return n
    raise ValueError(f"Could not infer n from {num_params} parameters and {num_components} components")


def default_params_file(run_dir: Path) -> Path:
    matches = sorted(run_dir.glob("best_params_global.txt"))
    if not matches:
        matches = sorted(run_dir.glob("params_best_n*_retain*.txt"))
    if not matches:
        matches = sorted(run_dir.glob("params.txt"))
    if not matches:
        matches = sorted(run_dir.glob("**/params_n*_retain*_seed*.txt"))
    if not matches:
        raise ValueError(f"No saved parameter file found below {run_dir}")
    return matches[0]


def predict(params, n, num_components, retain, entry, lor_true, central_point, coordinate_scales, fixed_width):
    params_tf = tf.constant(params, dtype=tf.float64)
    D_mod, S_list_mod, v0_mod, eta, width_params = helper.modified_DS_general(
        params_tf, n, num_components
    )
    M_true = helper.linear_matrix(D_mod, S_list_mod, entry, central_point, coordinate_scales)
    eigenvalues, eigenvectors = tf.linalg.eigh(M_true)

    n_i = eigenvalues.shape[0]
    k_keep = int(round(retain * n_i))
    k_keep = max(1, min(k_keep, n_i))
    left = (n_i - k_keep) // 2
    right = left + k_keep
    eigenvalues = eigenvalues[left:right]
    eigenvectors = eigenvectors[:, left:right]

    projections = tf.linalg.matvec(tf.transpose(eigenvectors), v0_mod)
    strengths = tf.square(projections)
    strengths = strengths * tf.cast((eigenvalues > 0) & (eigenvalues < 30), tf.float64)

    x = tf.constant(lor_true[:, 0], dtype=tf.float64)
    if fixed_width is None:
        width = helper.affine_width(eta, width_params, entry, central_point, coordinate_scales)
    else:
        width = tf.constant(float(fixed_width), dtype=tf.float64)
    y_pred = helper.give_me_Lorentzian(x, eigenvalues, strengths, width)
    return x.numpy(), y_pred.numpy(), lor_true[:, 1], eigenvalues.numpy(), strengths.numpy()


def predict_dipole(params, metadata, retain, entry, lor_true):
    dipole_helper = load_dipole_helper()
    model = metadata["model"]
    n = int(model["n"])
    num_params = int(model.get("n_params", len(helper.dataset_entry_params(entry))))
    config = dipole_helper.AnsatzConfig(
        n=n,
        n_params=num_params,
        ansatz=model.get("ansatz", "paper_dipole"),
        width_model=model.get("width_model", "affine"),
        use_vector_terms=bool(model.get("use_vector_terms", True)),
    )
    central_point = np.asarray(metadata["central_point"], dtype=np.float32)
    param_values = np.asarray([helper.dataset_entry_params(entry)], dtype=np.float32)
    params_tf = tf.convert_to_tensor(params, dtype=tf.float32)
    M_batch, v_batch, fwhm_batch, _ = dipole_helper.build_model_matrices_and_vectors(
        params_tf,
        config,
        tf.convert_to_tensor(param_values, dtype=tf.float32),
        tf.convert_to_tensor(central_point, dtype=tf.float32),
    )
    eigenvalues, eigenvectors = tf.linalg.eigh(M_batch)
    eigenvalues = eigenvalues[0]
    eigenvectors = eigenvectors[0]

    n_i = int(eigenvalues.shape[0])
    k_keep = int(round(float(retain) * n_i))
    k_keep = max(1, min(k_keep, n_i))
    left = (n_i - k_keep) // 2
    right = left + k_keep
    eigenvalues = eigenvalues[left:right]
    eigenvectors = eigenvectors[:, left:right]

    projections = tf.linalg.matvec(tf.transpose(eigenvectors), v_batch[0])
    strengths = tf.square(projections)
    x = tf.constant(lor_true[:, 0], dtype=tf.float32)
    y_pred = dipole_helper.give_me_Lorentzian(x, eigenvalues, strengths, 0.5 * fwhm_batch[0])
    return x.numpy(), y_pred.numpy(), lor_true[:, 1], eigenvalues.numpy(), strengths.numpy()


def infer_model_family(run_dir: Path, metadata: dict, explicit: str) -> str:
    if explicit != "auto":
        return explicit
    adapter = metadata.get("adapter") or ""
    if "Dipole" in adapter:
        return "dipole"
    model = metadata.get("model") or {}
    if model.get("ansatz") in {"paper_dipole", "linear", "quadratic", "linear_exp"} and (run_dir / "best_params_global.txt").is_file():
        return "dipole"
    return "beta"


def plot_one(out_dir, split_name, row_index, entry, x, y_pred, y_true, poles, strengths):
    out_dir.mkdir(parents=True, exist_ok=True)
    params_label = ", ".join(f"{v:.4g}" for v in helper.dataset_entry_params(entry))
    out = out_dir / f"pred_vs_true_{split_name}_row{row_index}.png"

    fig, ax = plt.subplots(figsize=(9, 5.5))
    markerline, stemlines, baseline = ax.stem(
        poles, strengths, linefmt="0.55", markerfmt="o", basefmt=" "
    )
    plt.setp(markerline, markersize=3, alpha=0.5)
    plt.setp(stemlines, linewidth=0.8, alpha=0.3)
    ax.plot(x, y_pred, label="pred", lw=1.9)
    ax.plot(x, y_true, label="true", lw=1.5)
    ax.set_xlim(float(np.min(x)), float(np.max(x)))
    ax.set_ylim(0)
    ax.set_xlabel("Energy")
    ax.set_ylabel("Strength")
    ax.set_title(f"{split_name} row {row_index}: p=({params_label})")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)

    denom = np.trapezoid(y_true ** 2, x) + 1e-16
    rel_l2 = np.trapezoid((y_pred - y_true) ** 2, x) / denom
    metrics = out.with_suffix(".txt")
    metrics.write_text(
        "\n".join(
            [
                f"split = {split_name}",
                f"row = {row_index}",
                f"params = {params_label}",
                f"true_max = {float(np.max(y_true)):.16e}",
                f"pred_max = {float(np.max(y_pred)):.16e}",
                f"true_integral = {float(np.trapezoid(y_true, x)):.16e}",
                f"pred_integral = {float(np.trapezoid(y_pred, x)):.16e}",
                f"relative_l2 = {float(rel_l2):.16e}",
            ]
        )
        + "\n"
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=None, help="Strength data directory. Defaults to run metadata or a dataset guess.")
    parser.add_argument("--params", type=Path, default=None, help="Saved params file. Defaults to params_best in --run-dir.")
    parser.add_argument("--split", choices=["train", "validation", "test", "all"], default="train")
    parser.add_argument("--index", type=int, default=1, help="1-based row within --split.")
    parser.add_argument("--all", action="store_true", help="Plot every row in --split.")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--retain", type=float, default=None)
    parser.add_argument("--reference-index", type=int, default=0, help="0-based reference index in sorted full dataset.")
    parser.add_argument("--fixed-width", type=float, default=None, help="Use a fixed width for prediction plots. Omit for affine-width runs.")
    parser.add_argument("--no-coordinate-normalization", action="store_true")
    parser.add_argument("--model-family", choices=["auto", "beta", "dipole"], default="auto")
    args = parser.parse_args()

    run_dir = args.run_dir
    metadata = load_run_metadata(run_dir)
    data_dir = infer_data_dir(run_dir, args.data_dir)
    model_family = infer_model_family(run_dir, metadata, args.model_family)
    retain = float(args.retain if args.retain is not None else (metadata.get("retain") or 1.0))
    params_path = args.params or default_params_file(run_dir)
    params = np.loadtxt(params_path)

    combined = load_strength_dataset(data_dir, metadata.get("strength_regex"))
    if args.split == "all":
        selected = combined
    else:
        selected = load_saved_split(run_dir, data_dir, args.split)
        if selected is None:
            selected = split_dataset(combined)[args.split]
    if not selected:
        raise ValueError(f"Split {args.split!r} is empty")

    num_components = len(helper.dataset_entry_params(combined[0]))
    if model_family == "beta":
        n = infer_n(len(params), num_components)
        combined_ar = np.array([helper.dataset_entry_params(entry) for entry in combined], dtype=float)
        if metadata.get("central_point") is not None:
            central_point = tuple(float(v) for v in metadata["central_point"])
        else:
            central_point = helper.dataset_entry_params(combined[max(0, min(args.reference_index, len(combined) - 1))])
        coordinate_scales = None
        if not args.no_coordinate_normalization:
            ranges = np.ptp(combined_ar, axis=0)
            coordinate_scales = tuple(float(v if v > 0 else 1.0) for v in ranges)
    else:
        n = None
        central_point = None
        coordinate_scales = None

    out_dir = args.out_dir or (run_dir / "pred_vs_true")
    rows = range(1, len(selected) + 1) if args.all else [args.index]
    for row in rows:
        if row < 1 or row > len(selected):
            raise ValueError(f"--index must be between 1 and {len(selected)} for split {args.split}, got {row}")
        entry = selected[row - 1]
        lor_true = np.loadtxt(helper.dataset_entry_path(entry))
        if model_family == "dipole":
            x, y_pred, y_true, poles, strengths = predict_dipole(
                params=params,
                metadata=metadata,
                retain=retain,
                entry=entry,
                lor_true=lor_true,
            )
        else:
            x, y_pred, y_true, poles, strengths = predict(
                params=params,
                n=n,
                num_components=num_components,
                retain=retain,
                entry=entry,
                lor_true=lor_true,
                central_point=central_point,
                coordinate_scales=coordinate_scales,
                fixed_width=args.fixed_width,
            )
        out = plot_one(out_dir, args.split, row, entry, x, y_pred, y_true, poles, strengths)
        print(out)


if __name__ == "__main__":
    main()
