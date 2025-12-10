import numpy as np

from smlr.data import StrengthDataset, StrengthSample
from smlr.emulator import StrengthEmulator
from smlr.lorentz import LorentzianMixture, lorentzian_sum
from smlr.metrics import normalized_l2


def _make_dataset():
    energy = np.linspace(-3.0, 3.0, 180)
    params_grid = (
        np.stack(np.meshgrid(np.linspace(0, 1, 4), np.linspace(0, 1, 4)), axis=-1)
        .reshape(-1, 2)
    )
    samples = []
    mixtures = []
    for p in params_grid:
        centers = np.array([-1.2 + 0.3 * p[0], 0.4 + 0.2 * p[1]])
        strengths = np.array([2.0 + 0.4 * p[0], 1.0 + 0.5 * p[1]])
        width = np.array([0.3 + 0.1 * p.sum()])
        y = lorentzian_sum(energy, centers, strengths, width)
        samples.append(StrengthSample(params=p, energy=energy, strength=y))
        mixtures.append(LorentzianMixture(centers, strengths, width))
    return StrengthDataset(samples), energy, mixtures


def test_emulator_interpolates_spectrum():
    dataset, energy, mixtures = _make_dataset()
    emu = StrengthEmulator(n_components=2, width_mode="global", random_state=1)
    emu.fit(dataset, mixtures=mixtures, normalize_strengths=False)

    target_params = np.array([0.25, 0.6])
    truth_centers = np.array([-1.2 + 0.3 * target_params[0], 0.4 + 0.2 * target_params[1]])
    truth_strengths = np.array([2.0 + 0.4 * target_params[0], 1.0 + 0.5 * target_params[1]])
    truth_width = np.array([0.3 + 0.1 * target_params.sum()])
    truth = lorentzian_sum(energy, truth_centers, truth_strengths, truth_width)

    _, pred = emu.predict(target_params, energy)
    err = normalized_l2(pred, truth, energy)
    assert err < 0.05, f"emulator error too high: {err}"