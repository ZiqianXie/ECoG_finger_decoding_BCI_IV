import numpy as np

from scripts.select_saved_event_validation_ensemble import event_oof


def test_event_oof_fits_affine_calibration_without_validation_rows() -> None:
    values = np.arange(8, dtype=float)
    target = 2.0 * values + 3.0
    folds = [
        np.asarray([1, 1, 0, 0, 0, 0, 0, 0], dtype=bool),
        np.asarray([0, 0, 1, 1, 1, 1, 1, 1], dtype=bool),
    ]
    calibrated = event_oof(values, target, folds)
    assert np.allclose(calibrated, target)
