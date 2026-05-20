from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MatrixAnsatzConfig:
    """Small duck-typed config consumed by ``smlr.core.ansatz``."""

    n: int
    n_params: int
    ansatz: str = "linear"
    width_model: str = "affine"
    use_vector_terms: bool = True

