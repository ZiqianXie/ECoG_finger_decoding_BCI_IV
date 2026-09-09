from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from calibrate_prediction_constrained import calibrate, finger_metrics  # noqa: E402


def test_calibration_preserves_floor_and_applies_deadzone_gain() -> None:
    prediction = np.array([-0.1, 0.0, 0.05, 0.10, 0.20], dtype=np.float32)
    observed = calibrate(prediction, floor=0.5, gain=4.0, deadzone=0.1)
    expected = np.array([-0.05, 0.0, 0.025, 0.05, 0.5], dtype=np.float32)
    np.testing.assert_allclose(observed, expected)


def test_finger_metrics_identify_perfect_state_and_shape() -> None:
    target = np.array([0.0, 0.0, 0.2, 0.8, 0.2, 0.0], dtype=np.float32)
    metrics = finger_metrics(target.copy(), target, threshold=0.1)
    assert metrics["pcc"] == pytest.approx(1.0, abs=1.0e-6)
    assert metrics["derivative_pcc"] == pytest.approx(1.0, abs=1.0e-6)
    assert metrics["state_precision"] == 1.0
    assert metrics["state_recall"] == 1.0
    assert metrics["state_f1"] == 1.0
    assert metrics["movement_peak_ratio"] == 1.0
    assert metrics["rest_rms"] == 0.0
