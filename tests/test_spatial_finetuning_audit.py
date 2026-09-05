import numpy as np

from scripts.audit_spatial_finetuning import spatial_change_metrics


def test_spatial_change_metrics_reports_rotation_and_selected_rows() -> None:
    initial = np.eye(2)
    trained = np.asarray([[0.0, 1.0], [0.0, 1.1]])

    metrics = spatial_change_metrics(initial, trained, np.asarray([1]))

    assert metrics["rows_over_5_degrees"] == 1
    assert np.isclose(metrics["max_row_angle_degrees"], 90.0)
    assert np.isclose(metrics["selected_row_relative_change_median"], 0.1)
    assert metrics["selected_component_indices"] == [1]
