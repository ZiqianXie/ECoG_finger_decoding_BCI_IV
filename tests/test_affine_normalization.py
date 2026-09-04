from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from normalize_prediction_affine import apply_affine, fit_positive_affine  # noqa: E402


def test_affine_normalization_recovers_scale_and_offset() -> None:
    prediction = np.linspace(-2.0, 2.0, 101)
    target = 0.25 * prediction + 0.4
    scale, offset = fit_positive_affine(prediction, target)
    assert scale == pytest.approx(0.25)
    assert offset == pytest.approx(0.4)
    np.testing.assert_allclose(
        apply_affine(prediction, scale, offset), target, atol=1.0e-12
    )


def test_positive_affine_does_not_reverse_an_anticorrelated_trace() -> None:
    prediction = np.arange(8.0)
    target = -prediction
    scale, offset = fit_positive_affine(prediction, target)
    assert scale == 0.0
    assert offset == pytest.approx(target.mean())
