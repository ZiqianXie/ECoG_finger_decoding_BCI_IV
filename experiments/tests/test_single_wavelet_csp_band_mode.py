from __future__ import annotations

import numpy as np
import torch

from cross_validate_single_wavelet import fit_csp_band_rows, resolve_seed_roles
from ecog_decoding.models import WaveletPacketEnergy
from single_wavelet_support import (
    OFFSET,
    csp_candidate_union,
    linear_gamma_leaf_signals,
    linear_lower_high_gamma_leaf_signals,
    linear_wavelet_leaf_signals,
)


def synthetic_inputs():
    rng = np.random.default_rng(7)
    rows = 40
    target = np.zeros((rows, 5), dtype=np.float32)
    target[20:, 4] = 1.0
    hhl = rng.normal(size=(OFFSET + rows, 4, 3)).astype(np.float32)
    hhh = rng.normal(size=(OFFSET + rows, 4, 3)).astype(np.float32)
    hhl[OFFSET + 20 :, :, 0] *= 3.0
    hhh[OFFSET + 20 :, :, 1] *= 4.0
    return target, np.arange(rows), hhl, hhh


def test_separate_gamma_mode_adds_one_row_per_leaf() -> None:
    target, training, hhl, hhh = synthetic_inputs()
    weights, audit = fit_csp_band_rows(
        joint_bins=np.concatenate((hhl, hhh), axis=1),
        hhl_bins=hhl,
        hhh_bins=hhh,
        target=target,
        training=training,
        finger_index=4,
        component_indices=(-1,),
        csp_band_mode="separate_hhl_hhh",
    )
    assert weights.shape == (2, 3)
    assert audit["band_mode"] == "separate_hhl_hhh"
    assert audit["spatial_rows"] == 2
    assert not np.allclose(weights[0], weights[1])


def test_joint_gamma_mode_retains_one_row() -> None:
    target, training, hhl, hhh = synthetic_inputs()
    weights, audit = fit_csp_band_rows(
        joint_bins=np.concatenate((hhl, hhh), axis=1),
        hhl_bins=hhl,
        hhh_bins=hhh,
        target=target,
        training=training,
        finger_index=4,
        component_indices=(-1,),
        csp_band_mode="joint_hhl_hhh",
    )
    assert weights.shape == (1, 3)
    assert audit["band_mode"] == "joint_hhl_hhh"


def test_dual_csp_contrast_retains_rest_and_other_movement_rows() -> None:
    rng = np.random.default_rng(17)
    rows = 60
    target = np.zeros((rows, 5), dtype=np.float32)
    target[20:40, 4] = 1.0
    target[40:, 0] = 1.0
    joint = rng.normal(size=(OFFSET + rows, 8, 3)).astype(np.float32)
    joint[OFFSET + 20 : OFFSET + 40, :, 0] *= 4.0
    joint[OFFSET + 40 :, :, 2] *= 3.0

    weights, audit = fit_csp_band_rows(
        joint_bins=joint,
        hhl_bins=None,
        hhh_bins=None,
        target=target,
        training=np.arange(rows),
        finger_index=4,
        component_indices=(-1,),
        csp_band_mode="joint_hhl_hhh",
        csp_contrast_mode="dual_rest_other",
    )

    assert weights.shape == (2, 3)
    assert audit["contrast_mode"] == "dual_rest_other"
    assert audit["contrasts"]["common_rest"]["rest_bins"] == 20
    assert audit["contrasts"]["other_movement"]["other_movement_bins"] == 20
    assert not np.allclose(weights[0], weights[1])


def test_triple_csp_contrast_adds_training_only_amplitude_row() -> None:
    rng = np.random.default_rng(23)
    rows = 100
    target = np.zeros((rows, 5), dtype=np.float32)
    target[20:40, 4] = np.linspace(0.21, 0.45, 20)
    target[40:60, 4] = np.linspace(0.75, 1.0, 20)
    target[60:80, 0] = 1.0
    target[80:, 4] = 100.0
    joint = rng.normal(size=(OFFSET + rows, 8, 3)).astype(np.float32)
    joint[OFFSET + 20 : OFFSET + 40, :, 0] *= 2.0
    joint[OFFSET + 40 : OFFSET + 60, :, 1] *= 5.0
    joint[OFFSET + 60 : OFFSET + 80, :, 2] *= 3.0

    weights, audit = fit_csp_band_rows(
        joint_bins=joint,
        hhl_bins=None,
        hhh_bins=None,
        target=target,
        training=np.arange(80),
        finger_index=4,
        component_indices=(-1,),
        csp_band_mode="joint_hhl_hhh",
        csp_contrast_mode="triple_rest_other_amplitude",
    )

    amplitude = audit["contrasts"]["lower_target_movement"]
    assert weights.shape == (3, 3)
    assert amplitude["active_bins"] == 10
    assert amplitude["lower_target_movement_bins"] == 20
    assert 0.85 < amplitude["high_movement_threshold"] < 0.90
    assert 0.45 < amplitude["lower_movement_ceiling"] < 0.75
    assert not np.allclose(weights[0], weights[2])


def test_continuous_amplitude_filter_is_split_safe_and_tracks_power() -> None:
    rng = np.random.default_rng(29)
    rows = 80
    training_rows = 60
    target = np.zeros((rows, 5), dtype=np.float32)
    target[:training_rows, 1] = np.linspace(0.0, 1.0, training_rows)
    target[training_rows:, 1] = 100.0
    joint = rng.normal(size=(OFFSET + rows, 12, 3)).astype(np.float32)
    training_scale = 0.5 + 3.0 * target[:training_rows, 1]
    joint[OFFSET : OFFSET + training_rows, :, 0] *= training_scale[:, None]
    joint[OFFSET + training_rows :, :, 2] *= 100.0

    weights, audit = fit_csp_band_rows(
        joint_bins=joint,
        hhl_bins=None,
        hhh_bins=None,
        target=target,
        training=np.arange(training_rows),
        finger_index=1,
        component_indices=(-1,),
        csp_band_mode="joint_hhl_hhh",
        csp_contrast_mode="continuous_amplitude",
    )

    assert weights.shape == (1, 3)
    assert audit["contrast_mode"] == "continuous_amplitude"
    assert audit["training_target_max"] == 1.0
    assert abs(weights[0, 0]) > 0.9
    assert abs(weights[0, 2]) < 0.2


def test_dual_rest_amplitude_retains_binary_and_continuous_rows() -> None:
    rng = np.random.default_rng(31)
    rows = 60
    target = np.zeros((rows, 5), dtype=np.float32)
    target[20:, 1] = np.linspace(0.21, 1.0, 40)
    joint = rng.normal(size=(OFFSET + rows, 12, 3)).astype(np.float32)
    joint[OFFSET + 20 :, :, 0] *= 2.0
    joint[OFFSET + 20 :, :, 1] *= 1.0 + 3.0 * target[20:, 1, None]

    weights, audit = fit_csp_band_rows(
        joint_bins=joint,
        hhl_bins=None,
        hhh_bins=None,
        target=target,
        training=np.arange(rows),
        finger_index=1,
        component_indices=(-1,),
        csp_band_mode="joint_hhl_hhh",
        csp_contrast_mode="dual_rest_amplitude",
    )

    assert weights.shape == (2, 3)
    assert set(audit["contrasts"]) == {"common_rest", "continuous_amplitude"}
    assert audit["contrasts"]["common_rest"]["rest_bins"] == 20
    assert audit["contrasts"]["continuous_amplitude"]["training_target_max"] == 1.0
    assert not np.allclose(weights[0], weights[1])


def test_triple_rest_other_continuous_amplitude_adds_three_rows() -> None:
    rng = np.random.default_rng(37)
    rows = 80
    target = np.zeros((rows, 5), dtype=np.float32)
    target[20:60, 1] = np.linspace(0.21, 1.0, 40)
    target[60:, 3] = 1.0
    joint = rng.normal(size=(OFFSET + rows, 12, 3)).astype(np.float32)
    joint[OFFSET + 20 : OFFSET + 60, :, 0] *= 2.0
    joint[OFFSET + 20 : OFFSET + 60, :, 1] *= 1.0 + 3.0 * target[20:60, 1, None]
    joint[OFFSET + 60 :, :, 2] *= 3.0

    weights, audit = fit_csp_band_rows(
        joint_bins=joint,
        hhl_bins=None,
        hhh_bins=None,
        target=target,
        training=np.arange(rows),
        finger_index=1,
        component_indices=(-1,),
        csp_band_mode="joint_hhl_hhh",
        csp_contrast_mode="triple_rest_other_continuous_amplitude",
    )

    assert weights.shape == (3, 3)
    assert set(audit["contrasts"]) == {
        "common_rest",
        "other_movement",
        "continuous_amplitude",
    }
    assert audit["contrasts"]["other_movement"]["other_movement_bins"] == 20


def test_lower_high_gamma_mode_adds_one_spatial_row_to_same_bank() -> None:
    target, training, hhl, hhh = synthetic_inputs()
    lower = np.random.default_rng(9).normal(size=hhl.shape).astype(np.float32)
    lower[OFFSET + 20 :, :, 2] *= 5.0
    weights, audit = fit_csp_band_rows(
        joint_bins=np.concatenate((hhl, hhh), axis=1),
        hhl_bins=hhl,
        hhh_bins=hhh,
        lower_high_gamma_bins=lower,
        target=target,
        training=training,
        finger_index=4,
        component_indices=(-1,),
        csp_band_mode="separate_50_100_hhl_hhh",
    )
    assert weights.shape == (3, 3)
    assert audit["spatial_rows"] == 3
    assert "wavelet_50_100_hz" in audit


def test_designed_seven_mode_fits_each_band_inside_one_spatial_bank() -> None:
    target, training, hhl, hhh = synthetic_inputs()
    designed = np.stack(
        [hhl * (1.0 + 0.1 * band) for band in range(7)], axis=0
    )
    weights, audit = fit_csp_band_rows(
        joint_bins=np.concatenate((hhl, hhh), axis=1),
        hhl_bins=hhl,
        hhh_bins=hhh,
        designed_band_bins=designed,
        target=target,
        training=training,
        finger_index=4,
        component_indices=(0, -1),
        csp_band_mode="designed_seven",
    )
    assert weights.shape == (14, 3)
    assert audit["spatial_rows"] == 14
    assert audit["source_band_indices"] == [
        band for band in range(7) for _ in range(2)
    ]
    assert len(audit["bands"]) == 7


def test_candidate_union_can_keep_only_band_aligned_csp_atoms() -> None:
    rng = np.random.default_rng(12)
    features = rng.normal(size=(6, 30)).astype(np.float32)
    target = rng.normal(size=(6, 5)).astype(np.float32)
    candidates, audit = csp_candidate_union(
        features,
        target,
        np.arange(6),
        per_bin=10,
        ica_per_bin=4,
        ica_prescreen=2,
        finger_index=4,
        csp_per_bin_positions=np.asarray([5, 8]),
    )
    csp = candidates[candidates % 10 >= 4]
    assert set((csp % 10).tolist()) == {5, 8}
    assert csp.size == 6
    assert audit["aligned_csp_positions_per_bin"] == 2


def test_seed_roles_can_hold_data_order_fixed() -> None:
    assert resolve_seed_roles(4, 2026, 2026) == (4, 2026, 2026)
    assert resolve_seed_roles(4, None, None) == (4, 4, 4)


def test_named_leaf_helpers_match_physical_frequency_positions() -> None:
    frontend = WaveletPacketEnergy(
        wavelet="bior6.8",
        levels=3,
        kernel_size=17,
        trainable=False,
        padding_mode="constant",
        energy_window_samples=40,
        energy_stride_samples=40,
        tap_resample_up=5,
        tap_resample_down=2,
    ).eval()
    ecog = np.random.default_rng(11).normal(size=(240, 3)).astype(np.float32)

    leaves = linear_wavelet_leaf_signals(
        ecog, frontend, torch.device("cpu"), (2, 3, 4, 5)
    )
    lower = linear_lower_high_gamma_leaf_signals(
        ecog, frontend, torch.device("cpu")
    )
    gamma = linear_gamma_leaf_signals(ecog, frontend, torch.device("cpu"))

    for expected, observed in zip(leaves, lower + gamma, strict=True):
        np.testing.assert_allclose(observed, expected)
