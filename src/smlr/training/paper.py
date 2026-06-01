from __future__ import annotations

from . import beta_em1, beta_em2, dipole_em1


def run_dipole_2d_example(argv: list[str]) -> None:
    defaults = [
        "--strength-dir",
        "dipole_polarizability_160Yb/total_strength",
        "--alphaD-dir",
        "dipole_polarizability_160Yb/total_alphaD",
        "--strength-regex",
        r"strength_(?P<p2>[0-9.]+)_(?P<p1>[0-9.]+)\.out",
        "--alphaD-regex",
        r"alphaD_(?P<p2>[0-9.]+)_(?P<p1>[0-9.]+)\.out",
        "--filter-ranges",
        '{"p1":[0.4,1.8],"p2":[1.5,4.0]}',
        "--central-point",
        "[1.1,2.75]",
        "--ansatz",
        "linear_exp",
        "--width-model",
        "affine",
        "--save-dir",
        "runs/dp_2d_spectral",
    ]
    dipole_em1.run(defaults + list(argv or []))


def run_dipole_paper_em1(argv: list[str]) -> None:
    dipole_em1.run(argv)


def run_beta_paper_em1(argv: list[str]) -> None:
    beta_em1.run(argv)


def run_beta_paper_em2(argv: list[str]) -> None:
    beta_em2.run(argv)
