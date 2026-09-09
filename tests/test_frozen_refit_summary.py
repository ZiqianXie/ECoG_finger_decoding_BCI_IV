from pathlib import Path

from summarize_frozen_full_refit import resolve_refit_root, training_only_collapse


def test_training_only_collapse_uses_only_training_metrics() -> None:
    summary = {
        "initialized_full_train_raw_pcc": 0.45,
        "fitted_full_train_raw_pcc": 0.40,
        "released_test_metrics": {"raw_pcc": 0.99},
    }
    collapsed, change = training_only_collapse(summary, 0.02)
    assert collapsed
    assert abs(change + 0.05) < 1.0e-12


def test_training_only_collapse_retains_improved_and_unscreened_seeds() -> None:
    summary = {
        "initialized_full_train_raw_pcc": 0.45,
        "fitted_full_train_raw_pcc": 0.46,
    }
    assert training_only_collapse(summary, 0.02)[0] is False
    assert training_only_collapse(summary, None)[0] is False


def test_refit_root_can_be_selected_per_finger() -> None:
    fallback = Path("outputs/default")
    assert resolve_refit_root({}, fallback) == fallback
    assert resolve_refit_root(
        {"refit_root": "outputs/sevenband"}, fallback
    ) == Path("outputs/sevenband")
