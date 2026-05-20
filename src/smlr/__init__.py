"""SMLR: Surrogate Models for Linear Response.

This package is built with uv and provides the core project metadata
for the SMLR software package.
"""

__version__ = "0.1.0"

from .validation import (
    ValidationIssue,
    ValidationReport,
    validate_paper_beta_data,
    validate_paper_dipole_data,
    validate_strength_grid,
)

__all__ = [
    "BetaDecayAdapter",
    "DipoleAdapter",
    "PaperBetaDecayAdapter",
    "PaperDipoleAdapter",
    "ValidationIssue",
    "ValidationReport",
    "__version__",
    "validate_paper_beta_data",
    "validate_paper_dipole_data",
    "validate_strength_grid",
]


def __getattr__(name):
    if name in {"BetaDecayAdapter", "DipoleAdapter", "PaperBetaDecayAdapter", "PaperDipoleAdapter"}:
        from .domains import BetaDecayAdapter, DipoleAdapter, PaperBetaDecayAdapter, PaperDipoleAdapter

        adapters = {
            "BetaDecayAdapter": BetaDecayAdapter,
            "DipoleAdapter": DipoleAdapter,
            "PaperBetaDecayAdapter": PaperBetaDecayAdapter,
            "PaperDipoleAdapter": PaperDipoleAdapter,
        }
        return adapters[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
