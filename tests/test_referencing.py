import numpy as np

from ecog_decoding.referencing import fit_standardize, grid_neighbors, rereference


def test_grid_neighbors_respect_row_boundaries_and_missing_channels() -> None:
    neighbors = grid_neighbors([1, 2, 3, 9, 10], columns=8)
    assert neighbors[1] == (2, 9)
    assert neighbors[2] == (1, 3, 10)
    assert 9 not in neighbors[2]


def test_car_has_zero_instantaneous_mean() -> None:
    values = np.arange(20, dtype=np.float64).reshape(5, 4)
    transformed, _ = rereference(values, [1, 2, 9, 10], "car")
    np.testing.assert_allclose(transformed.mean(axis=1), 0.0, atol=1e-12)


def test_bipolar_and_laplacian_use_physical_grid_neighbors() -> None:
    values = np.array([[1.0, 3.0, 5.0, 9.0]])
    bipolar, details = rereference(values, [1, 2, 9, 10], "bipolar")
    assert details["bipolar_pairs_one_based"] == [[1, 2], [1, 9], [2, 10], [9, 10]]
    np.testing.assert_allclose(bipolar, [[-2.0, -4.0, -6.0, -4.0]])
    laplacian, details = rereference(values, [1, 2, 9, 10], "laplacian")
    assert details["laplacian_centers_one_based"] == [1, 2, 9, 10]
    np.testing.assert_allclose(laplacian, [[-3.0, -2.0, 0.0, 5.0]])


def test_standardization_fits_training_prefix_only() -> None:
    train = np.array([[0.0, 1.0], [2.0, 3.0], [100.0, 200.0]])
    test = np.array([[4.0, 5.0]])
    normalized_train, normalized_test, mean, scale = fit_standardize(train, test, 2)
    np.testing.assert_allclose(mean, [1.0, 2.0])
    np.testing.assert_allclose(scale, [1.0, 1.0])
    np.testing.assert_allclose(normalized_train[:2].mean(axis=0), 0.0)
    np.testing.assert_allclose(normalized_test, [[3.0, 3.0]])
