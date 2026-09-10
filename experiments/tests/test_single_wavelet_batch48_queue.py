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
    assert "--wavelet-interlevel-skip" not in command
    assert "--wavelet-interlevel-normalization" not in command


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
