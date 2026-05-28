"""SMLR: Surrogate Models for Linear Response.

This package is built with uv and provides the core project metadata
for the SMLR software package.
"""

__version__ = "0.1.0"

from .core.retention import RetainedModePolicy
from .validation import (
    GENERIC_2D_PARAMETER_NAMES,
    GENERIC_2D_STRENGTH_REGEX,
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
    generic_2d_strength_spec,
    load_run_spec,
    paper_beta_em1_spec,
    paper_beta_em2_spec,
    paper_dipole_em1_spec,
    h2_2d_strength_spec,
)
from .serialization import (
    EmulatorRecord,
    LoadedEmulator,
    load_emulator,
    package_existing_emulator,
    save_emulator,
)

__all__ = [
    "BetaDecayAdapter",
    "DipoleAdapter",
    "EmulatorRecord",
    "EmulatorRunSpec",
    "GENERIC_2D_PARAMETER_NAMES",
    "GENERIC_2D_STRENGTH_REGEX",
    "LoadedEmulator",
    "ObservableSpec",
    "PaperBetaDecayAdapter",
    "PaperDipoleAdapter",
    "RetainedModePolicy",
    "StrengthGridSpec",
    "ValidationIssue",
    "ValidationReport",
    "__version__",
    "generic_2d_strength_spec",
    "load_emulator",
    "h2_2d_strength_spec",
    "load_run_spec",
    "package_existing_emulator",
    "paper_beta_em1_spec",
    "paper_beta_em2_spec",
    "paper_dipole_em1_spec",
    "save_emulator",
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
