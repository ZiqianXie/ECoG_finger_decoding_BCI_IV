import numpy as np

from scripts.benchmark_event_heterogeneous_dictionary import (
    csp_selection_audit,
    selected_csp_filters,
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
