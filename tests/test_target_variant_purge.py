from pathlib import Path

import numpy as np

from scripts.benchmark_event_target_variants import (
    purge_near_validation,
    target_path,
    target_support_bins,
)


def test_target_paths_include_raw_glove_special_case() -> None:
    root = Path("prepared")
    assert target_path(root, "raw_25hz") == root / "train_glove_25hz_raw.npy"
    assert target_path(root, "local_w2_q10") == root / "train_glove_local_w2_q10.npy"


def test_target_support_includes_percentile_and_gaussian_radius() -> None:
    assert target_support_bins("raw_25hz") == 0
    assert target_support_bins("paper_baseline_only") is None
    assert target_support_bins("local_w1_q10") == 32
    assert target_support_bins("local_w6_q30") == 95


def test_extra_target_purge_removes_rows_near_validation() -> None:
    training = np.r_[np.arange(0, 30), np.arange(40, 100)]
    purged = purge_near_validation(training, [[30, 40]], row_count=100, margin=5)
    assert np.array_equal(purged, np.r_[np.arange(0, 25), np.arange(45, 100)])
