from __future__ import annotations

from smlr.training.paper import _run_compat_module


def run_dipole_paper_em1(argv: list[str]) -> None:
    _run_compat_module(
        "Dipole_polarizability.src.diagnostics_general_gpt_styled",
        argv,
        "Dipole_polarizability/src/diagnostics_general_gpt_styled.py",
    )


def run_beta(argv: list[str]) -> None:
    _run_compat_module(
        "Beta_decay_package.src.diagnostics_general_gpt",
        argv,
        "Beta_decay_package/src/diagnostics_general_gpt.py",
    )
