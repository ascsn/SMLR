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
from .splitting import (
    ParameterSplit,
    choose_central_parameter_point,
    parse_filter_ranges,
    split_by_parameter_ranges,
)
from .training import make_optimizer, moving_average, set_all_seeds

__all__ = [
    "PackedLayout",
    "ParameterSplit",
    "build_model_matrices_and_vectors",
    "centered_keep_indices",
    "centered_spectrum_initialization",
    "choose_central_parameter_point",
    "compute_ansatz_features",
    "compute_trainable_fwhm",
    "fit_strength_with_tf_lorentzian",
    "get_packed_layout",
    "give_me_lorentzian",
    "give_me_lorentzian_batched",
    "make_optimizer",
    "make_random_initial_guess",
    "moving_average",
    "parse_filter_ranges",
    "set_all_seeds",
    "split_by_parameter_ranges",
    "sym_from_upper",
    "unpack_trainable_parameters",
]
