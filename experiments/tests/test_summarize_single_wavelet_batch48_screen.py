from __future__ import annotations

from experiments.scripts.summarize_single_wavelet_batch48_screen import summarize_report


def test_summarize_report_computes_gain_and_diagnostic_means() -> None:
    metric = {
        "event_macro_nmse": 0.4,
        "rest_false_positive_rms": 0.1,
        "movement_rmse": 0.2,
        "amplitude_slope": 0.7,
        "velocity_pcc": 0.3,
        "movement_balanced_accuracy": 0.8,
        "raw_pcc": 0.6,
    }
    report = {
        "stitched_initialized_raw_pcc": 0.4,
        "stitched_selected_raw_pcc": 0.6,
        "runtime_seconds": 12.0,
        "outer_folds": [
            {
                "selected_schedule": "frozen_25",
                "initialized_outer_metrics": metric,
                "selected_outer_metrics": metric,
            }
        ],
    }

    observed = summarize_report(report)

    assert observed["gain"] == 0.19999999999999996
    assert observed["selected_schedules"] == ["frozen_25"]
    assert observed["mean_selected_outer_diagnostics"]["velocity_pcc"] == 0.3
