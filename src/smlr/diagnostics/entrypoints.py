from __future__ import annotations

from smlr.training._compat import run_legacy_entrypoint


def run_dipole_paper_em1(argv: list[str]) -> None:
    run_legacy_entrypoint(
        "Dipole_polarizability.src.diagnostics_general_gpt_styled",
        argv,
        "Dipole_polarizability/src/diagnostics_general_gpt_styled.py",
    )


def run_beta(argv: list[str]) -> None:
    run_legacy_entrypoint(
        "Beta_decay_package.src.diagnostics_general_gpt",
        argv,
        "Beta_decay_package/src/diagnostics_general_gpt.py",
    )
