import importlib.util
import sys
from pathlib import Path

import numpy as np


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "summarize_single_wavelet_oof_ensemble.py"
)
SPEC = importlib.util.spec_from_file_location("summarize_single_wavelet_oof_ensemble", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_collapse_rule_does_not_exclude_a_low_pcc_finite_seed() -> None:
    raw = np.arange(6, dtype=np.float32)
    target = raw.copy()
    initialized = raw * 0.5
    predictions = {
        0: raw.copy(),
        1: raw[::-1].copy(),
        2: np.ones_like(raw),
    }

    report, ensemble = MODULE.summarize_predictions(
        initialized, target, raw, predictions, collapse_sd=1.0e-8
    )

    assert report["included_seed_count"] == 2
    assert report["collapsed_seed_count"] == 1
    assert [item["eligible_without_target"] for item in report["members"]] == [
        True,
        True,
        False,
    ]
    np.testing.assert_allclose(ensemble, np.full(6, 2.5, dtype=np.float32))


def test_finite_intervals_split_at_oof_gaps() -> None:
    first = np.asarray([np.nan, 1.0, 2.0, np.nan, 3.0, 4.0])
    second = np.asarray([np.nan, 1.0, 2.0, np.nan, 3.0, np.nan])

    assert MODULE.finite_intervals(first, second) == [[1, 3], [4, 5]]
