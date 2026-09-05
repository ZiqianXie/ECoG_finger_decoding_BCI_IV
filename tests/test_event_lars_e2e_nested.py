import numpy as np

from scripts.train_event_grouped_lars_e2e_nested import event_grouped_cv_splits
from scripts.train_event_grouped_lars_lstm import indices_from_intervals


def test_event_grouped_subfolds_are_disjoint_and_complete() -> None:
    intervals = [[0, 9], [14, 29], [35, 46], [52, 61], [70, 83], [90, 98]]
    training_indices = indices_from_intervals(intervals)
    splits = event_grouped_cv_splits(intervals, training_indices, folds=3)
    validation_seen: list[int] = []
    for training, validation in splits:
        assert np.intersect1d(training, validation).size == 0
        assert np.union1d(training, validation).size == training_indices.size
        validation_seen.extend(validation.tolist())
    assert np.array_equal(np.sort(validation_seen), np.arange(training_indices.size))
