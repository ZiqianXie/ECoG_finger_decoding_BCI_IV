import json
from pathlib import Path

import numpy as np
import torch
import yaml

from prepare_perfinger_ica_csp_wavelet_features import fit_csp_rows
from train_event_grouped_lars_e2e_nested import (
    build_model,
    compact_spatial_dictionary,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FINGERS = ("thumb", "index", "middle", "ring", "little")


def test_seven_band_csp_initializes_one_spatial_bank() -> None:
    rng = np.random.default_rng(29)
    target = np.zeros((30, 5), dtype=np.float32)
    target[5:10, 2] = 1.0
    target[12:16, 0] = 1.0
    filtered = rng.normal(size=(7, 54, 16, 4)).astype(np.float32)
    filtered[:, 29:34, :, 0] *= 2.0
    rows, audit = fit_csp_rows(
        filtered,
        target,
        np.arange(20, dtype=np.int64),
        2,
        (0, -1),
        include_any_movement=True,
    )

    assert rows.shape == (7 * 2 * 2, 4)
    assert len(audit) == 7
    assert all("decoded_finger" in record for record in audit)
    assert all("any_movement" in record for record in audit)
    np.testing.assert_allclose(np.linalg.norm(rows, axis=1), 1.0, atol=1.0e-5)


def test_compaction_keeps_one_spatial_wavelet_feature_path() -> None:
    weights = np.arange(20, dtype=np.float32).reshape(4, 5)
    selected = np.asarray([0, 2, 7, 10], dtype=np.int64)
    compact_weights, remapped, used_rows = compact_spatial_dictionary(
        weights, selected, full_feature_count=12
    )

    np.testing.assert_array_equal(used_rows, [0, 2, 3])
    np.testing.assert_array_equal(compact_weights, weights[used_rows])
    full = np.arange(24, dtype=np.float32).reshape(2, 12)
    compact = full.reshape(2, 4, 3)[:, used_rows].reshape(2, -1)
    np.testing.assert_array_equal(compact[:, remapped], full[:, selected])


def test_final_decoder_has_no_parallel_or_beta_gate_branch() -> None:
    model = build_model(
        input_channels=2,
        ica=np.eye(2, dtype=np.float32),
        selected=np.asarray([0, 1], dtype=np.int64),
        mean=np.zeros(2, dtype=np.float32),
        scale=np.ones(2, dtype=np.float32),
        coefficients=np.asarray([0.2, -0.1], dtype=np.float32),
        intercept=0.03,
        hidden_size=4,
        near_zero_std=1.0e-3,
        output_activation="softplus",
        device=torch.device("cpu"),
    )

    assert model.lstm.input_size == 2
    assert hasattr(model, "spatial")
    assert hasattr(model, "wavelet")
    assert not hasattr(model, "beta_gate_enabled")
    assert not hasattr(model, "beta_context_gru")
    assert not hasattr(model, "parallel_branches")


def test_final_map_covers_all_fifteen_whole_models() -> None:
    config = yaml.safe_load(
        (REPOSITORY_ROOT / "configs/final_single_branch_oof_selected.yaml").read_text()
    )
    default_root = config["default"]["refit_root"]
    routes: list[str] = []
    for subject in (1, 2, 3):
        subject_options = config.get("subjects", {}).get(subject, {})
        for finger in FINGERS:
            routes.append(subject_options.get(finger, {}).get("refit_root", default_root))

    assert len(routes) == 15
    assert routes.count(default_root) == 11
    assert len(routes) - routes.count(default_root) == 4

    result = json.loads(
        (REPOSITORY_ROOT / "docs/results/final-single-branch-six-seed.json").read_text()
    )
    assert result["architecture"]["parallel_decoder_branches"] is False
    assert result["architecture"]["output_stacking"] is False
    assert result["ensemble"]["retained_members"] == 90
    assert result["summary"]["subjects_beating_paper_macro5"] == 3
