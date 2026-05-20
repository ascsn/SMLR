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
from .specs import (
    EmulatorRunSpec,
    ObservableSpec,
    StrengthGridSpec,
    paper_beta_em1_spec,
    paper_beta_em2_spec,
    paper_dipole_em1_spec,
)

__all__ = [
    "BetaDecayAdapter",
    "DipoleAdapter",
    "EmulatorRunSpec",
    "ObservableSpec",
    "PaperBetaDecayAdapter",
    "PaperDipoleAdapter",
    "StrengthGridSpec",
    "ValidationIssue",
    "ValidationReport",
    "__version__",
    "paper_beta_em1_spec",
    "paper_beta_em2_spec",
    "paper_dipole_em1_spec",
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
