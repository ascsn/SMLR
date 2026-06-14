from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf

from smlr.core.ansatz import EDIT_ansatz as core_ansatz
from smlr.core import fitting as core_fitting
from smlr.core import numerics as core_numerics
from .common import MatrixAnsatzConfig


EMASS = 0.511
KAPPA = 6147
DEL_NP = 1.293
DEL_NH = 0.782


@dataclass
class BetaDecayAdapter:
    """Configurable beta-decay-like LRT adapter.

    The default structure is intentionally lightweight and user-configurable.
    Paper-specific constants and training defaults live in
    ``PaperBetaDecayAdapter``.
    """

    data_dir: str
    nucnam: str
    n: int = 8
    n_params: int = 2
    retain: float = 0.9
    ansatz: str = "linear"
    width_model: str = "affine"
    use_vector_terms: bool = False
    g_A: float = 1.0
    emass: float = EMASS
    kappa: float = KAPPA
    del_np: float = DEL_NP
    del_nH: float = DEL_NH

    def model_config(self) -> MatrixAnsatzConfig:
        return MatrixAnsatzConfig(
            n=self.n,
            n_params=self.n_params,
            ansatz=self.ansatz,
            width_model=self.width_model,
            use_vector_terms=self.use_vector_terms,
        )

    def strength_path(self, alpha: str, beta: str) -> Path:
        return Path(self.data_dir) / f"lorm_{self.nucnam}_{beta}_{alpha}.out"

    def excitation_path(self, alpha: str, beta: str) -> Path:
        return Path(self.data_dir) / f"excm_{self.nucnam}_{beta}_{alpha}.out"

    def fit_central_strength(self, omega, strength, n_poles: Optional[int] = None, eta: float = 0.5):
        n_poles = n_poles if n_poles is not None else max(1, round(self.retain * self.n))
        return core_fitting.fit_strength_with_tf_lorentzian(
            omega,
            strength,
            n_poles,
            eta,
            np_dtype=np.float64,
            tf_dtype=tf.float64,
            gap_floor=1e-6,
        )

    def lorentzian(self, energy, poles, strengths, width):
        return core_numerics.give_me_lorentzian(energy, poles, strengths, width, dtype=tf.float64)

    def symmetric_from_upper(self, values, n: Optional[int] = None):
        return core_ansatz.sym_from_upper(values, self.n if n is None else n, dtype=tf.float64)

    def half_life_from_phase_polynomial(self, eigenvalues, strengths, coeffs):
        """Beta half-life observable for a supplied phase-space polynomial."""
        eigenvalues = tf.convert_to_tensor(eigenvalues, dtype=tf.float64)
        strengths = tf.convert_to_tensor(strengths, dtype=tf.float64)
        coeffs = tf.convert_to_tensor(coeffs, dtype=tf.float64)
        W_0 = (tf.constant(self.del_np, dtype=tf.float64) - eigenvalues) / tf.constant(self.emass, dtype=tf.float64)
        valid = tf.greater(W_0, 1.0)
        coefficients = tf.reverse(coeffs, axis=[0])
        x_tensor = W_0 * tf.constant(self.emass, dtype=tf.float64)
        powers = tf.range(tf.shape(coefficients)[0], dtype=tf.float64)
        phase = tf.reduce_sum(coefficients * tf.pow(tf.reshape(x_tensor, [-1, 1]), powers), axis=-1)
        contributions = tf.where(
            valid,
            phase
            * strengths
            * tf.constant(self.g_A**2, dtype=tf.float64)
            * tf.math.log(tf.constant(2.0, dtype=tf.float64))
            / tf.constant(self.kappa, dtype=tf.float64),
            tf.constant(0.0, dtype=tf.float64),
        )
        return tf.math.log(tf.constant(2.0, dtype=tf.float64)) / tf.reduce_sum(contributions)


@dataclass
class PaperBetaDecayAdapter(BetaDecayAdapter):
    """Paper-reproduction beta-decay adapter with Ni-80 defaults."""

    data_dir: str = "beta_decay_80Ni"
    nucnam: str = "Ni_80"
    n: int = 8
    n_params: int = 2
    retain: float = 0.9
    ansatz: str = "paper_beta_decay"
    width_model: str = "paper_affine_quadrature"
    use_vector_terms: bool = False
    g_A: float = 1.2
    A: int = 80
    Z: int = 28
    half_life_weight: float = 1.0

    def model_config(self) -> MatrixAnsatzConfig:
        # The current beta paper workflow uses a custom packed layout rather
        # than the generic matrix ansatz layout. Keep this method explicit so
        # callers do not accidentally treat it as a generalized adapter.
        return MatrixAnsatzConfig(
            n=self.n,
            n_params=self.n_params,
            ansatz=self.ansatz,
            width_model=self.width_model,
            use_vector_terms=self.use_vector_terms,
        )

    def unpack_em1_parameters(self, params, n: Optional[int] = None):
        n = self.n if n is None else int(n)
        params = tf.convert_to_tensor(params, dtype=tf.float64)
        idx = 0
        eta = tf.convert_to_tensor(params[idx])
        idx += 1
        v0 = tf.convert_to_tensor(params[idx:idx + n])
        idx += n
        D = tf.linalg.diag(params[idx:idx + n])
        idx += n
        num_upper = n * (n + 1) // 2
        S1 = core_ansatz.sym_from_upper(params[idx:idx + num_upper], n, dtype=tf.float64)
        idx += num_upper
        S2 = core_ansatz.sym_from_upper(params[idx:idx + num_upper], n, dtype=tf.float64)
        idx += num_upper
        x1 = tf.convert_to_tensor(params[idx])
        x2 = tf.convert_to_tensor(params[idx + 1])
        x3 = tf.convert_to_tensor(params[idx + 2])
        return D, S1, S2, v0, eta, x1, x2, x3

    def unpack_em2_parameters(self, params, n: Optional[int] = None):
        n = self.n if n is None else int(n)
        params = tf.convert_to_tensor(params, dtype=tf.float64)
        D = tf.linalg.diag(params[:n])
        num_upper = n * (n + 1) // 2
        s1_start = n
        s2_start = s1_start + num_upper
        S1 = core_ansatz.sym_from_upper(params[s1_start:s2_start], n, dtype=tf.float64)
        S2 = core_ansatz.sym_from_upper(params[s2_start:s2_start + num_upper], n, dtype=tf.float64)
        return D, S1, S2

    def em1_width(self, eta, x1, x2, x3, alpha: float, beta: float):
        return tf.sqrt(tf.square(eta) + tf.square(x1 + x2 * float(alpha) + x3 * float(beta)))

    def em1_matrix(self, D, S1, S2, alpha: float, beta: float, central_point):
        return D + (float(alpha) - float(central_point[0])) * S1 + (float(beta) - float(central_point[1])) * S2

    def em2_matrix(self, D, S1, S2, alpha: float, beta: float, central_point):
        return D + (float(alpha) - float(central_point[0])) * S1 + (float(beta) - float(central_point[1])) * S2
