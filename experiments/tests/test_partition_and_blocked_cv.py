import numpy as np

from scripts.audit_partition_shift import column_correlation, target_summary, vector_cosine
from scripts.benchmark_fixed_lars import named_subset_metrics
from scripts.crossvalidate_selected_ridge import pearson, purged_block_folds


def test_column_correlation_handles_constant_columns() -> None:
    values = np.column_stack((np.arange(6), np.ones(6), -np.arange(6)))
    target = np.arange(6)
    np.testing.assert_allclose(column_correlation(values, target), [1.0, 0.0, -1.0])


def test_vector_cosine_and_pearson_handle_degenerate_inputs() -> None:
    assert vector_cosine(np.zeros(3), np.ones(3)) == 0.0
    assert pearson(np.ones(4), np.arange(4)) == 0.0


def test_target_summary_reports_activity() -> None:
    summary = target_summary(np.array([0.0, 0.1, 0.2, 0.3]), threshold=0.2)
    assert summary["active_fraction"] == 0.5
    assert np.isclose(summary["mean"], 0.15)


def test_purged_blocks_cover_validation_once_and_remove_neighbors() -> None:
    folds = purged_block_folds(sample_count=20, fold_count=4, purge=2)
    validation = np.concatenate([held_out for _, held_out in folds])
    np.testing.assert_array_equal(np.sort(validation), np.arange(20))
    for training, held_out in folds:
        lower = max(0, held_out[0] - 2)
        upper = min(20, held_out[-1] + 3)
        assert not np.isin(training, np.arange(lower, upper)).any()


def test_named_subset_metrics_preserves_requested_finger_name() -> None:
    target = np.arange(8, dtype=np.float32)[:, None]
    metrics = named_subset_metrics(target.copy(), target, ["little"])
    assert set(metrics["pearson_by_finger"]) == {"little"}
    assert np.isclose(metrics["pearson_by_finger"]["little"], 1.0)
