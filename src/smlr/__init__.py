"""Core SMLR research-code modules."""

__version__ = "0.1.0"

from .core.retention import RetainedModePolicy

__all__ = [
    "BetaDecayAdapter",
    "DipoleAdapter",
    "PaperBetaDecayAdapter",
    "PaperDipoleAdapter",
    "RetainedModePolicy",
    "__version__",
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
