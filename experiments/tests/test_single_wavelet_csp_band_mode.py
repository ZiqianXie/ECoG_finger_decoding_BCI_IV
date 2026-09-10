from __future__ import annotations

import numpy as np
import torch

from cross_validate_single_wavelet import fit_csp_band_rows, resolve_seed_roles
from ecog_decoding.models import WaveletPacketEnergy
from single_wavelet_support import (
    OFFSET,
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
