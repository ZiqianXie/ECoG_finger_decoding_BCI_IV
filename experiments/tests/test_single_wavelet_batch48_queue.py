from __future__ import annotations

import json

import pytest

from experiments.scripts.run_single_wavelet_batch48_queue import (
    build_command,
    completed_summary,
    parse_spec,
)


def test_parse_spec_validates_all_fields() -> None:
    assert parse_spec("3:little:2") == (3, "little", 2)
    with pytest.raises(Exception):
        parse_spec("4:little:2")
    with pytest.raises(Exception):
        parse_spec("3:toe:2")


def test_completed_summary_requires_requested_single_fold(tmp_path) -> None:
    path = tmp_path / "summary.json"
    path.write_text(json.dumps({"outer_folds": [{"outer_fold": 1}]}))
    assert completed_summary(path, 1)
    assert not completed_summary(path, 0)


def test_command_keeps_single_branch_residual_configuration(tmp_path) -> None:
    command = build_command(
        "python",
        tmp_path / "cross_validate.py",
        2,
        "ring",
        1,
        tmp_path / "output",
        tmp_path / "initialization",
        tmp_path / "ica",
    )
    joined = " ".join(command)
    assert "--recurrent-cell residual_lstm" in joined
    assert "--residual-input current_candidate" in joined
    assert "--residual-include-direct" in command
    assert "--frozen-only" in command
    assert "--csp-band-mode joint_hhl_hhh" in joined
    assert "--csp-mode movement_1" in joined
    assert "--residual-input-width 64" in joined
    assert "--correlation-loss-weight 0.0" in joined
    assert "--derivative-correlation-weight 0.0" in joined
    assert "--raw-movement-correlation-weight 0.0" in joined
    assert "--raw-movement-derivative-correlation-weight 0.0" in joined
    assert "--lasso-backend torch_fista" in joined
    assert "--wavelet-frontend depth3" in joined
    assert "--frozen-update-grid 10 25 50 100 200" in joined
    assert "--wavelet-interlevel-skip" not in command
    assert "--wavelet-interlevel-normalization" not in command


def test_command_can_request_two_csp_components_per_band(tmp_path) -> None:
    command = build_command(
        "python",
        tmp_path / "cross_validate.py",
        3,
        "ring",
        0,
        tmp_path / "output",
        tmp_path / "initialization",
        tmp_path / "ica",
        csp_band_mode="separate_50_100_hhl_hhh",
        csp_mode="movement_2",
    )
    joined = " ".join(command)
    assert "--csp-mode movement_2" in joined
    assert "--csp-band-mode separate_50_100_hhl_hhh" in joined


def test_command_can_extend_the_frozen_training_schedule(tmp_path) -> None:
    command = build_command(
        "python",
        tmp_path / "cross_validate.py",
        1,
        "thumb",
        2,
        tmp_path / "output",
        tmp_path / "initialization",
        tmp_path / "ica",
        frozen_update_grid=(10, 25, 50, 100, 200, 400, 800),
    )
    joined = " ".join(command)
    assert "--frozen-update-grid 10 25 50 100 200 400 800" in joined


def test_command_can_request_one_overcomplete_wavelet_tree(tmp_path) -> None:
    command = build_command(
        "python",
        tmp_path / "cross_validate.py",
        1,
        "middle",
        0,
        tmp_path / "output",
        tmp_path / "initialization",
        tmp_path / "ica",
        wavelet_frontend="overcomplete_depth3_depth4",
    )
    joined = " ".join(command)
    assert "--wavelet-frontend overcomplete_depth3_depth4" in joined
    assert "--csp-band-mode joint_hhl_hhh" in joined
    assert "--recurrent-cell residual_lstm" in joined


def test_command_can_request_depth5_overcomplete_wavelet_tree(tmp_path) -> None:
    command = build_command(
        "python",
        tmp_path / "cross_validate.py",
        1,
        "middle",
        0,
        tmp_path / "output",
        tmp_path / "initialization",
        tmp_path / "ica",
        wavelet_frontend="overcomplete_depth3_depth4_depth5",
    )
    joined = " ".join(command)
    assert "--wavelet-frontend overcomplete_depth3_depth4_depth5" in joined
    assert "--recurrent-cell residual_lstm" in joined


def test_command_can_fit_separate_gamma_csp_rows(tmp_path) -> None:
    command = build_command(
        "python",
        tmp_path / "cross_validate.py",
        3,
        "middle",
        2,
        tmp_path / "output",
        tmp_path / "initialization",
        tmp_path / "ica",
        "separate_hhl_hhh",
        96,
        0.1,
        0.1,
        0.2,
        0.3,
        "torch_fista",
    )
    joined = " ".join(command)
    assert "--csp-band-mode separate_hhl_hhh" in joined
    assert "--residual-input-width 96" in joined
    assert "--correlation-loss-weight 0.1" in joined
    assert "--derivative-correlation-weight 0.1" in joined
    assert "--raw-movement-correlation-weight 0.2" in joined
    assert "--raw-movement-derivative-correlation-weight 0.3" in joined
    assert "--lasso-backend torch_fista" in joined


def test_command_can_request_designed_band_csp_rows(tmp_path) -> None:
    command = build_command(
        "python",
        tmp_path / "cross_validate.py",
        1,
        "middle",
        0,
        tmp_path / "output",
        tmp_path / "initialization",
        tmp_path / "ica",
        csp_band_mode="designed_seven",
        csp_mode="tails_2x2",
        residual_input_width=256,
        wavelet_frontend="overcomplete_depth3_depth4",
    )
    joined = " ".join(command)
    assert "--csp-band-mode designed_seven" in joined
    assert "--csp-mode tails_2x2" in joined
    assert "--residual-input-width 256" in joined
    assert "--wavelet-frontend overcomplete_depth3_depth4" in joined


def test_command_can_randomize_near_zero_residual_output(tmp_path) -> None:
    command = build_command(
        "python",
        tmp_path / "cross_validate.py",
        1,
        "middle",
        0,
        tmp_path / "output",
        tmp_path / "initialization",
        tmp_path / "ica",
        residual_output_init_std=0.001,
    )
    joined = " ".join(command)
    assert "--residual-output-init-std 0.001" in joined


def test_command_can_lower_the_head_learning_rate(tmp_path) -> None:
    command = build_command(
        "python",
        tmp_path / "cross_validate.py",
        1,
        "middle",
        0,
        tmp_path / "output",
        tmp_path / "initialization",
        tmp_path / "ica",
        head_learning_rate=1.0e-4,
    )
    joined = " ".join(command)
    assert "--head-learning-rate 0.0001" in joined


def test_command_can_use_gpu_vectorized_leaky_residual_state(tmp_path) -> None:
    command = build_command(
        "python",
        tmp_path / "cross_validate.py",
        1,
        "middle",
        0,
        tmp_path / "output",
        tmp_path / "initialization",
        tmp_path / "ica",
        residual_dynamics="leaky_velocity",
        residual_decay=0.95,
    )
    joined = " ".join(command)
    assert "--residual-dynamics leaky_velocity" in joined
    assert "--residual-decay 0.95" in joined


def test_command_can_blend_split_local_raw_trajectory_target(tmp_path) -> None:
    command = build_command(
        "python",
        tmp_path / "cross_validate.py",
        3,
        "middle",
        1,
        tmp_path / "output",
        tmp_path / "initialization",
        tmp_path / "ica",
        raw_trajectory_blend=0.25,
    )
    assert "--raw-trajectory-blend 0.25" in " ".join(command)
