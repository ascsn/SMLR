#!/usr/bin/env python3

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