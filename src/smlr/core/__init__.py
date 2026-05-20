"""Core structure-preserving emulator primitives."""

from .ansatz import (
    PackedLayout,
    build_model_matrices_and_vectors,
    compute_ansatz_features,
    compute_trainable_fwhm,
    get_packed_layout,
    make_random_initial_guess,
    sym_from_upper,
    unpack_trainable_parameters,
)
from .fitting import fit_strength_with_tf_lorentzian
from .numerics import (
    centered_keep_indices,
    centered_spectrum_initialization,
    give_me_lorentzian,
    give_me_lorentzian_batched,
)
from .training import make_optimizer, moving_average, set_all_seeds

__all__ = [
    "PackedLayout",
    "build_model_matrices_and_vectors",
    "centered_keep_indices",
    "centered_spectrum_initialization",
    "compute_ansatz_features",
    "compute_trainable_fwhm",
    "fit_strength_with_tf_lorentzian",
    "get_packed_layout",
    "give_me_lorentzian",
    "give_me_lorentzian_batched",
    "make_optimizer",
    "make_random_initial_guess",
    "moving_average",
    "set_all_seeds",
    "sym_from_upper",
    "unpack_trainable_parameters",
]

