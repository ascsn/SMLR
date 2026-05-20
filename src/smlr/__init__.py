"""SMLR: Surrogate Models for Linear Response.

This package is built with uv and provides the core project metadata
for the SMLR software package.
"""

__version__ = "0.1.0"

from .domains import (
    BetaDecayAdapter,
    DipoleAdapter,
    PaperBetaDecayAdapter,
    PaperDipoleAdapter,
)

__all__ = [
    "BetaDecayAdapter",
    "DipoleAdapter",
    "PaperBetaDecayAdapter",
    "PaperDipoleAdapter",
    "__version__",
]
