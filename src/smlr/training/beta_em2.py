from __future__ import annotations

from ._compat import run_legacy_entrypoint


def run(argv: list[str] | None = None) -> None:
    """Run the beta-decay EM2 half-life-only paper trainer."""

    run_legacy_entrypoint(
        "Beta_decay_package.src.main_only_HL_gpt",
        list(argv or []),
        "Beta_decay_package/src/main_only_HL_gpt.py",
    )


def main(argv: list[str] | None = None) -> None:
    run(argv)
