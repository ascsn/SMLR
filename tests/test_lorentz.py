import numpy as np

from smlr.lorentz import fit_lorentzian_mixture, lorentzian_sum
from smlr.metrics import normalized_l2


def test_fit_recovers_known_mixture():
    rng = np.random.default_rng(0)
    energy = np.linspace(-2.0, 2.0, 300)
    centers = np.array([-0.8, 0.6])
    strengths = np.array([2.5, 1.4])
    widths = np.array([0.35, 0.25])
    truth = lorentzian_sum(energy, centers, strengths, widths)
    noisy = truth + 0.02 * rng.standard_normal(size=energy.shape)

    fit = fit_lorentzian_mixture(energy, noisy, n_components=2, width_mode="per_component")
    pred = fit.evaluate(energy)

    err = normalized_l2(pred, truth, energy)
    assert err < 0.05, f"normalized L2 too large: {err}"