from __future__ import annotations

import numpy as np

from cross_validate_single_wavelet import fit_csp_band_rows, resolve_seed_roles
from single_wavelet_support import OFFSET


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


def test_seed_roles_can_hold_data_order_fixed() -> None:
    assert resolve_seed_roles(4, 2026, 2026) == (4, 2026, 2026)
    assert resolve_seed_roles(4, None, None) == (4, 4, 4)
