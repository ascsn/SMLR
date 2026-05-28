from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import tensorflow as tf

from smlr.core import ansatz as core_ansatz
from smlr.core import fitting as core_fitting
from smlr.core import numerics as core_numerics
from .common import MatrixAnsatzConfig


HQC = 197.33
ALPHAD_FAC = 8.0 * np.pi * (7.29735e-3) * HQC / 9.0


@dataclass
class DipoleAdapter:
    """Configurable dipole-like LRT adapter.

    This is intentionally not paper-locked. It exposes the data conventions and
    matrix ansatz as configuration while relying on ``smlr.core`` for shared
    emulator mechanics.
    """

    strength_dir: str
    observable_dir: Optional[str] = None
    strength_regex: str = r"strength_(?P<p2>[0-9.]+)_(?P<p1>[0-9.]+)\.out"
    observable_regex: Optional[str] = None
    n: int = 10
    n_params: int = 2
    retain: float = 0.5
    fold: float = 2.0
    ansatz: str = "linear"
    width_model: str = "affine"
    use_vector_terms: bool = True
    w_strength: float = 1.0
    w_mminus1: float = 1.0
    w_mplus1: float = 0.0
    m1_target: float = 875.0

    def model_config(self) -> MatrixAnsatzConfig:
        return MatrixAnsatzConfig(
            n=self.n,
            n_params=self.n_params,
            ansatz=self.ansatz,
            width_model=self.width_model,
            use_vector_terms=self.use_vector_terms,
        )

    def fit_central_strength(self, omega, strength, n_poles: Optional[int] = None):
        n_poles = n_poles if n_poles is not None else max(1, round(self.retain * self.n))
        return core_fitting.fit_strength_with_tf_lorentzian(
            omega,
            strength,
            n_poles,
            self.fold,
            min_spacing=0.01,
            np_dtype=np.float32,
            tf_dtype=tf.float32,
        )

    def encode_initial_guess(self, initial_guess, fitted_energies, fitted_strengths):
        layout = core_ansatz.get_packed_layout(self.model_config())
        D_full, v0_full, _ = core_numerics.centered_spectrum_initialization(
            fitted_energies,
            fitted_strengths,
            self.n,
            self.retain,
            dtype=np.float32,
        )
        params = np.asarray(initial_guess, dtype=np.float32).copy()
        params[layout.v0_slice] = v0_full
        params[layout.d_diag_slice] = D_full
        return params

    def build_model(self, params, param_values, central_point):
        return core_ansatz.build_model_matrices_and_vectors(
            params=tf.convert_to_tensor(params, dtype=tf.float32),
            config=self.model_config(),
            param_values=tf.convert_to_tensor(param_values, dtype=tf.float32),
            central_point=tf.convert_to_tensor(central_point, dtype=tf.float32),
        )

    def lorentzian(self, energy, poles, strengths, width):
        return core_numerics.give_me_lorentzian(energy, poles, strengths, width, dtype=tf.float32)

    def alphaD_from_poles(self, eigenvalues, strengths):
        eigenvalues = tf.convert_to_tensor(eigenvalues, dtype=tf.float32)
        strengths = tf.convert_to_tensor(strengths, dtype=tf.float32)
        mask = tf.cast(eigenvalues > 1.0, dtype=tf.float32)
        return tf.reduce_sum((strengths * mask) / tf.maximum(eigenvalues, 1e-6)) * tf.constant(ALPHAD_FAC, tf.float32)


@dataclass
class PaperDipoleAdapter(DipoleAdapter):
    """Paper-reproduction dipole adapter with current project defaults."""

    strength_dir: str = "dipole_polarizability_160Yb/total_strength"
    observable_dir: Optional[str] = "dipole_polarizability_160Yb/total_alphaD"
    strength_regex: str = r"strength_(?P<p2>[0-9.]+)_(?P<p1>[0-9.]+)\.out"
    observable_regex: Optional[str] = None
    n: int = 10
    n_params: int = 2
    retain: float = 0.5
    fold: float = 2.0
    ansatz: str = "paper_dipole"
    width_model: str = "affine"
    use_vector_terms: bool = True
    w_strength: float = 1.0
    w_mminus1: float = 2.0
    w_mplus1: float = 0.0
    m1_target: float = 875.0
