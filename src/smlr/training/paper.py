from __future__ import annotations

from . import beta_em1, beta_em2, dipole_em1


def run_dipole_paper_em1(argv: list[str]) -> None:
    dipole_em1.run(argv)


def run_beta_paper_em1(argv: list[str]) -> None:
    beta_em1.run(argv)


def run_beta_paper_em2(argv: list[str]) -> None:
    beta_em2.run(argv)
