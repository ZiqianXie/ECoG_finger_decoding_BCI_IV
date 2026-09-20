#!/usr/bin/env python3
"""Measure fixed zero-phase smoothing of an existing OOF trajectory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage

from build_event_stratified_folds import runs
from ecog_decoding.training import FINGER_NAMES
from single_wavelet_support import OFFSET, pearson


def shift_interval_prediction(
    prediction: np.ndarray, observed: np.ndarray, shift_bins: int
) -> np.ndarray:
    """Shift each finite interval without borrowing values across boundaries."""
    shifted = prediction.copy()
    for start, stop in runs(observed):
        interval = prediction[start:stop]
        if shift_bins > 0:
            shifted[start:stop] = np.concatenate(
                (np.repeat(interval[:1], shift_bins), interval[:-shift_bins])
            ) if shift_bins < interval.size else interval[0]
        elif shift_bins < 0:
            advance = -shift_bins
            shifted[start:stop] = np.concatenate(
                (interval[advance:], np.repeat(interval[-1:], advance))
            ) if advance < interval.size else interval[-1]
    return shifted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--subject", type=int, required=True)
    parser.add_argument("--finger", choices=tuple(FINGER_NAMES), required=True)
    parser.add_argument("--fixed-reference-pcc", type=float, required=True)
    parser.add_argument("--prepared-root", type=Path, default=Path("outputs/preprocessed_v2"))
    args = parser.parse_args()

    prediction = np.load(args.prediction)
    finger_index = list(FINGER_NAMES).index(args.finger)
    raw = np.load(
        args.prepared_root / f"sub{args.subject}" / "train_glove_25hz_raw.npy",
        mmap_mode="r",
    )[OFFSET : OFFSET + prediction.size, finger_index]
    observed = np.isfinite(prediction)
    result = {}
    for sigma in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0):
        smoothed = prediction.copy()
        if sigma:
            for start, stop in runs(observed):
                smoothed[start:stop] = ndimage.gaussian_filter1d(
                    prediction[start:stop], sigma=sigma, mode="nearest"
                )
        pcc = pearson(smoothed[observed], raw[observed])
        result[str(sigma)] = {
            "pcc": pcc,
            "gain_over_fixed_reference": pcc - args.fixed_reference_pcc,
        }
    temporal = []
    for shift_bins in range(-8, 9):
        shifted = shift_interval_prediction(prediction, observed, shift_bins)
        for sigma in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0):
            transformed = shifted.copy()
            if sigma:
                for start, stop in runs(observed):
                    transformed[start:stop] = ndimage.gaussian_filter1d(
                        shifted[start:stop], sigma=sigma, mode="nearest"
                    )
            pcc = pearson(transformed[observed], raw[observed])
            temporal.append(
                {
                    "shift_bins": shift_bins,
                    "sigma": sigma,
                    "pcc": pcc,
                    "gain_over_fixed_reference": pcc - args.fixed_reference_pcc,
                }
            )
    result["outer_diagnostic_best_shift_smoothing"] = {
        **max(temporal, key=lambda item: item["pcc"]),
        "selection_warning": "outer-target upper-bound diagnostic only",
    }
    best = None
    for floor in (0.0, 0.25, 0.5, 0.75, 1.0):
        for gain in (1.0, 2.0, 4.0, 8.0):
            for deadzone in (0.025, 0.05, 0.075, 0.1, 0.15, 0.2):
                calibrated = floor * prediction + gain * np.maximum(
                    prediction - deadzone, 0.0
                )
                pcc = pearson(calibrated[observed], raw[observed])
                candidate = (pcc, floor, gain, deadzone)
                if best is None or candidate[0] > best[0]:
                    best = candidate
    assert best is not None
    result["outer_diagnostic_best_hinge"] = {
        "pcc": best[0],
        "gain_over_fixed_reference": best[0] - args.fixed_reference_pcc,
        "floor": best[1],
        "gain": best[2],
        "deadzone": best[3],
        "selection_warning": "outer-target upper-bound diagnostic only",
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
