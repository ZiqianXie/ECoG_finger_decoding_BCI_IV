from argparse import Namespace
from pathlib import Path

from run_single_wavelet_cv import command


def base_args(**overrides) -> Namespace:
    values = {
        "output_root": Path("outputs/new"),
        "model_rate": 1000,
        "tap_resample_up": 5,
        "tap_resample_down": 2,
        "fold_root": Path("outputs/folds"),
        "head_learning_rate": 1.0e-4,
        "spatial_learning_rate": 3.0e-6,
        "wavelet_learning_rate": 3.0e-6,
        "head_initialization": "lars_linear_regime",
        "recurrent_cell": "paper_equations",
        "csp_mode": "movement_1",
        "lars_candidate_scale": 1.0,
        "lars_near_zero_std": 1.0e-3,
        "lars_open_gate_bias": 3.0,
        "lars_forget_gate_bias": -3.0,
        "sequence_steps": 100,
        "sequence_stride": 25,
        "sampler_mode": "dense_sequences",
        "batch_size": 24,
        "output_activation": "softplus",
        "softplus_beta": 10.0,
        "selection_metric": "raw_pcc",
        "selection_rule": "best",
        "purge_bins": 95,
        "frozen_update_grid": (20, 40),
        "unfreeze_after": 40,
        "unfrozen_update_grid": (20, 40),
        "seed": 2026,
        "require_lstm_update": True,
        "reuse_from_root": None,
        "initialization_from_root": Path("outputs/standard"),
        "reuse_inner_metrics_from_root": None,
        "ica_cache_root": None,
        "compile": True,
    }
    values.update(overrides)
    return Namespace(**values)


def option_value(values: list[str], option: str) -> str:
    return values[values.index(option) + 1]


def test_paper_cell_reuses_initialization_without_reusing_training_curve() -> None:
    values = command(base_args(), 1, "little")
    assert option_value(values, "--recurrent-cell") == "paper_equations"
    assert option_value(values, "--lars-open-gate-bias") == "3.0"
    assert option_value(values, "--initialization-cache-root") == (
        "outputs/standard/sub1/little"
    )
    assert "--reuse-inner-metrics-from" not in values
