"""Package-native training entry points.

The first paper entry points preserve the CLI contract while delegating to the
current compatibility scripts. The implementation behind these functions will
be migrated into this package module by module.
"""

from . import beta_em1, beta_em2, dipole_em1, strength_only
from .paper import run_beta_paper_em1, run_beta_paper_em2, run_dipole_paper_em1

__all__ = [
    "beta_em1",
    "beta_em2",
    "dipole_em1",
    "strength_only",
    "run_beta_paper_em1",
    "run_beta_paper_em2",
    "run_dipole_paper_em1",
]
