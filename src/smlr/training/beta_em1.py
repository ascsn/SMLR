from __future__ import annotations

from ._compat import run_legacy_entrypoint


def run(argv: list[str] | None = None) -> None:
    """Run the beta-decay EM1 paper trainer."""

    run_legacy_entrypoint(
        "Beta_decay_package.src.main_gpt2",
        list(argv or []),
        "Beta_decay_package/src/main_gpt2.py",
    )


def main(argv: list[str] | None = None) -> None:
    run(argv)
