from __future__ import annotations

import numpy as np
import pytest

from smlr.core.data_io import DatasetPoint, discover_dataset, load_strengths, parse_strength_filename


def write_strength(path, omega=(0.0, 1.0, 2.0), strength=(0.2, 0.4, 0.6)):
    table = np.column_stack([omega, strength])
    np.savetxt(path, table)


def test_parse_strength_filename_accepts_path_or_string():
    np.testing.assert_allclose(
        parse_strength_filename("strength_0.5_-1.25_3e-2.out"),
        np.array([0.5, -1.25, 0.03]),
    )


def test_parse_strength_filename_rejects_non_strength_name():
    with pytest.raises(ValueError, match="Not a strength filename"):
        parse_strength_filename("alphaD_0.5_1.0.out")


def test_discover_dataset_sorts_by_parameter_values(tmp_path):
    write_strength(tmp_path / "strength_2.0_0.0.out")
    write_strength(tmp_path / "strength_1.0_0.0.out")
    write_strength(tmp_path / "notes.out")

    dataset = discover_dataset(tmp_path)

    assert dataset.ndim == 2
    assert [point.path.name for point in dataset.points] == [
        "strength_1.0_0.0.out",
        "strength_2.0_0.0.out",
    ]
    np.testing.assert_allclose(dataset.param_values, np.array([[1.0, 0.0], [2.0, 0.0]]))


def test_discover_dataset_rejects_mixed_parameter_dimensions(tmp_path):
    write_strength(tmp_path / "strength_1.0.out")
    write_strength(tmp_path / "strength_1.0_2.0.out")

    with pytest.raises(ValueError, match="Inconsistent parameter dimension"):
        discover_dataset(tmp_path)


def test_load_strengths_returns_shared_grid_strength_matrix_and_params(tmp_path):
    path_a = tmp_path / "strength_0.0_1.0.out"
    path_b = tmp_path / "strength_1.0_1.0.out"
    write_strength(path_a, strength=(1.0, 2.0, 3.0))
    write_strength(path_b, strength=(4.0, 5.0, 6.0))

    dataset = discover_dataset(tmp_path)
    omega, strengths, params = load_strengths(dataset)

    np.testing.assert_allclose(omega, np.array([0.0, 1.0, 2.0]))
    np.testing.assert_allclose(strengths, np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    np.testing.assert_allclose(params, np.array([[0.0, 1.0], [1.0, 1.0]]))


def test_load_strengths_rejects_mismatched_energy_grids(tmp_path):
    path_a = tmp_path / "strength_0.0.out"
    path_b = tmp_path / "strength_1.0.out"
    write_strength(path_a, omega=(0.0, 1.0, 2.0))
    write_strength(path_b, omega=(0.0, 1.1, 2.0))

    points = (
        DatasetPoint(params=np.array([0.0]), path=path_a),
        DatasetPoint(params=np.array([1.0]), path=path_b),
    )
    with pytest.raises(ValueError, match="Energy grid"):
        load_strengths(points)

