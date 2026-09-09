import numpy as np

from scripts.apply_validation_latent_gate import event_balanced_folds


def test_event_balanced_folds_partition_rows_and_keep_active_runs_together() -> None:
    state = np.asarray([0] * 20 + [1] * 5 + [0] * 20 + [1] * 4 + [2] * 6)
    folds = event_balanced_folds(state, folds=3)
    assert np.all(np.sum(np.stack(folds), axis=0) == 1)
    for start, stop in ((20, 25), (45, 49), (49, 55)):
        assert sum(bool(np.all(fold[start:stop])) for fold in folds) == 1
