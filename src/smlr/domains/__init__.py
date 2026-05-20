"""Domain adapters for SMLR emulator workflows."""

from .beta_decay import BetaDecayAdapter, PaperBetaDecayAdapter
from .dipole import DipoleAdapter, PaperDipoleAdapter

__all__ = [
    "BetaDecayAdapter",
    "DipoleAdapter",
    "PaperBetaDecayAdapter",
    "PaperDipoleAdapter",
]

