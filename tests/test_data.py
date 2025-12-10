import numpy as np

from smlr.data import StrengthDataset, StrengthSample


def test_dataset_to_matrix_interpolates():
    energy_a = np.linspace(-1, 1, 50)
    energy_b = np.linspace(-1, 1, 80)
    params = np.array([[0.0, 0.1], [0.5, 0.2]])
    strength_a = np.sin(energy_a) ** 2
    strength_b = np.cos(energy_b)

    ds = StrengthDataset(
        [
            StrengthSample(params[0], energy_a, strength_a, label="a"),
            StrengthSample(params[1], energy_b, strength_b, label="b"),
        ]
    )

    target_grid = np.linspace(-1, 1, 60)
    pmat, grid, smat = ds.to_matrix(target_grid)
    assert pmat.shape == (2, 2)
    assert grid.shape == target_grid.shape
    assert smat.shape == (2, target_grid.shape[0])
    # Check interpolation kept bounds
    assert np.isclose(smat[0, 0], strength_a[0])
