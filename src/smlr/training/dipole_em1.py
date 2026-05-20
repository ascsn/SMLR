from __future__ import annotations

from ._compat import run_legacy_entrypoint


def run(argv: list[str] | None = None) -> None:
    """Run the dipole EM1 paper trainer.

    The command is package-native, while the training internals are still being
    migrated from the historical paper script in small, testable pieces.
    """

    run_legacy_entrypoint(
        "Dipole_polarizability.src.main_gpt2",
        list(argv or []),
        "Dipole_polarizability/src/main_gpt2.py",
    )


def main(argv: list[str] | None = None) -> None:
    run(argv)
