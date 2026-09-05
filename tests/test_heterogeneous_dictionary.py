import numpy as np

from scripts.benchmark_event_heterogeneous_dictionary import (
    csp_selection_audit,
    selected_csp_filters,
)
from scripts.prepare_heterogeneous_full_refit import (
    correlations_by_column,
    selected_matrix,
)


def test_csp_filter_fit_uses_requested_active_and_rest_rows() -> None:
    rng = np.random.default_rng(4)
    values = rng.normal(size=(12, 5, 3)).astype(np.float32)
    values[3:6, :, 0] *= 3.0
    active = np.zeros(12, dtype=bool)
    active[3:6] = True
    rest = ~active

    weights, eigenvalues = selected_csp_filters(
        values, np.arange(12), active, rest, components_per_tail=1
    )

    assert weights.shape == (2, 3)
    assert len(eigenvalues) == 2
    np.testing.assert_allclose(np.linalg.norm(weights, axis=1), 1.0, atol=1.0e-5)


def test_joint_selection_audit_separates_wavelet_and_csp_families() -> None:
    # 56 CSP atoms per history bin: seven bands times four own plus four shared.
    selected = np.array([2, 80, 100 + 0, 100 + 4, 100 + 55, 100 + 56])
    audit = csp_selection_audit(
        selected,
        wavelet_features=100,
        csp_features_per_bin=56,
        filters_per_state=4,
    )

    assert audit["ica_wavelet_features"] == 2
    assert audit["csp_designed_band_features"] == 4
    assert audit["csp_selected_by_spatial_contrast"] == {
        "decoded_finger": 2,
        "any_movement": 2,
    }
    assert audit["csp_selected_by_band_hz"]["155-195"] == 1


def test_joint_feature_gather_preserves_global_index_order() -> None:
    wavelet = np.arange(30, dtype=np.float32).reshape(5, 6)
    csp = 100 + np.arange(20, dtype=np.float32).reshape(5, 4)

    observed = selected_matrix(wavelet, csp, np.array([0, 7, 5, 9]))

    np.testing.assert_array_equal(
        observed,
        np.column_stack((wavelet[:, 0], csp[:, 1], wavelet[:, 5], csp[:, 3])),
    )


def test_blockwise_correlations_match_numpy() -> None:
    rng = np.random.default_rng(9)
    features = rng.normal(size=(31, 7)).astype(np.float32)
    target = (0.7 * features[:, 2] - 0.3 * features[:, 5] + rng.normal(size=31)).astype(
        np.float32
    )

    observed = correlations_by_column(features, target, block_columns=3)
    expected = np.corrcoef(np.column_stack((features, target)), rowvar=False)[:-1, -1]

    np.testing.assert_allclose(observed, expected, atol=1.0e-12)
