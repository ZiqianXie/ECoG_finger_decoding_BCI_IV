from __future__ import annotations

import pytest

from scripts.build_event_stratified_folds import selection_stop


def test_selection_stop_preserves_original_model_fit_scope() -> None:
    assert selection_stop(
        raw_target_rows=10_000,
        model_fit_stop=6_666,
        selection_scope="model-fit",
    ) == 6_666


def test_selection_stop_can_include_complete_development_recording() -> None:
    assert selection_stop(
        raw_target_rows=10_000,
        model_fit_stop=6_666,
        selection_scope="full-development",
    ) == 10_000


def test_selection_stop_rejects_unknown_scope() -> None:
    with pytest.raises(ValueError, match="unknown selection scope"):
        selection_stop(
            raw_target_rows=10_000,
            model_fit_stop=6_666,
            selection_scope="test",
        )
