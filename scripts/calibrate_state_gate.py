#!/usr/bin/env python3
"""Calibrate state-aware predictions for morphology rather than PCC alone."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ecog_decoding.training import FINGER_NAMES, trajectory_metrics


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    xc = x - x.mean()
    yc = y - y.mean()
    denominator = np.linalg.norm(xc) * np.linalg.norm(yc)
    return float(xc @ yc / denominator) if denominator > 0 else 0.0


def remove_short_runs(mask: np.ndarray, minimum: int = 2) -> np.ndarray:
    result = mask.copy()
    changes = np.flatnonzero(np.diff(np.r_[False, result, False]))
    for start, stop in changes.reshape(-1, 2):
        if stop - start < minimum:
            result[start:stop] = False
    return result


def fit_gain(value: np.ndarray, target: np.ndarray) -> float:
    denominator = float(value @ value)
    if denominator <= 1e-12:
        return 1.0
    return float(np.clip((value @ target) / denominator, 0.25, 8.0))


def gain_candidates(
    value: np.ndarray, target: np.ndarray, movement_threshold: float
) -> dict[str, float]:
    moving = target >= movement_threshold
    candidates = {"global_ls": fit_gain(value, target)}
    if moving.any():
        candidates["movement_ls"] = fit_gain(value[moving], target[moving])
        predicted_peak = float(np.quantile(value[moving], 0.95))
        target_peak = float(np.quantile(target[moving], 0.95))
        candidates["peak_q95"] = float(
            np.clip(target_peak / max(predicted_peak, 1e-6), 0.25, 8.0)
        )
    return candidates


def quality(value: np.ndarray, target: np.ndarray, movement_threshold: float) -> dict[str, float]:
    moving = target >= movement_threshold
    predicted_moving = value >= movement_threshold
    tp = np.sum(moving & predicted_moving)
    fp = np.sum(~moving & predicted_moving)
    fn = np.sum(moving & ~predicted_moving)
    precision = float(tp / max(tp + fp, 1))
    recall = float(tp / max(tp + fn, 1))
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    rest_rms = float(np.sqrt(np.mean(value[~moving] ** 2)))
    movement_rms = float(np.sqrt(np.mean(target[moving] ** 2)))
    peak_ratio = float(
        np.quantile(value[moving], 0.95) / max(np.quantile(target[moving], 0.95), 1e-12)
    )
    pcc = pearson(value, target)
    derivative_pcc = pearson(np.diff(value), np.diff(target))
    score = (
        0.45 * pcc
        + 0.20 * derivative_pcc
        + 0.20 * f1
        - 0.10 * rest_rms / max(movement_rms, 1e-12)
        - 0.05 * abs(np.log(max(peak_ratio, 1e-4)))
    )
    return {
        "quality_score": score,
        "pcc_cleaned": pcc,
        "derivative_pcc": derivative_pcc,
        "state_f1": f1,
        "rest_rms": rest_rms,
        "peak_ratio": peak_ratio,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, default=3)
    parser.add_argument("--prepared-root", type=Path, default=Path("outputs/preprocessed_v2"))
    parser.add_argument("--state-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", default="local_w4_q20")
    parser.add_argument("--history", type=int, default=25)
    parser.add_argument("--movement-threshold", type=float, default=0.1)
    parser.add_argument(
        "--gate-modes",
        nargs="+",
        choices=("continuous", "soft", "binary"),
        default=("continuous", "soft", "binary"),
    )
    args = parser.parse_args()

    root = args.prepared_root / f"sub{args.subject}"
    metadata = json.loads((root / "metadata.json").read_text())
    split = int(metadata["target_fit_samples_25hz"])
    offset = args.history - 1
    validation_target = np.load(root / f"train_glove_{args.target}.npy")[split:]
    test_target = np.load(root / f"test_glove_{args.target}.npy")[offset:]
    validation_raw = np.load(root / "train_glove_25hz_raw.npy")[split:]
    test_raw = np.load(root / "test_glove_25hz_raw.npy")[offset:]
    validation_amplitude = np.load(args.state_output / "validation_amplitude.npy")
    test_amplitude = np.load(args.state_output / "test_amplitude.npy")
    validation_probability = np.load(args.state_output / "validation_state_probability.npy")
    test_probability = np.load(args.state_output / "test_state_probability.npy")
    validation_prediction = np.zeros_like(validation_amplitude)
    test_prediction = np.zeros_like(test_amplitude)
    selections: dict[str, object] = {}

    for finger, name in enumerate(FINGER_NAMES):
        candidates: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        if "continuous" in args.gate_modes:
            candidates["continuous"] = (
                validation_amplitude[:, finger], test_amplitude[:, finger]
            )
        if "soft" in args.gate_modes:
            for floor in (0.0, 0.1, 0.25, 0.5):
                for exponent in (0.5, 1.0, 2.0):
                    validation_gate = floor + (1.0 - floor) * np.power(
                        validation_probability[:, finger], exponent
                    )
                    test_gate = floor + (1.0 - floor) * np.power(
                        test_probability[:, finger], exponent
                    )
                    candidates[f"soft_f{floor:.2f}_g{exponent:.1f}"] = (
                        validation_amplitude[:, finger] * validation_gate,
                        test_amplitude[:, finger] * test_gate,
                    )
        if "binary" in args.gate_modes:
            for threshold in np.arange(0.2, 0.81, 0.05):
                validation_gate = remove_short_runs(validation_probability[:, finger] >= threshold)
                test_gate = remove_short_runs(test_probability[:, finger] >= threshold)
                candidates[f"binary_t{threshold:.2f}"] = (
                    validation_amplitude[:, finger] * validation_gate,
                    test_amplitude[:, finger] * test_gate,
                )
        audits: dict[str, object] = {}
        calibrated: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for mode, (validation_value, test_value) in candidates.items():
            for gain_name, gain in gain_candidates(
                validation_value,
                validation_target[:, finger],
                args.movement_threshold,
            ).items():
                candidate_name = f"{mode}__{gain_name}"
                calibrated_validation = gain * validation_value
                calibrated_test = gain * test_value
                metrics = quality(
                    calibrated_validation,
                    validation_target[:, finger],
                    args.movement_threshold,
                )
                audits[candidate_name] = {"gain": gain, **metrics}
                calibrated[candidate_name] = calibrated_validation, calibrated_test
        winner = max(audits, key=lambda mode: audits[mode]["quality_score"])
        validation_prediction[:, finger] = calibrated[winner][0]
        test_prediction[:, finger] = calibrated[winner][1]
        selections[name] = {"selected_mode": winner, "candidates": audits}

    report = {
        "subject": args.subject,
        "source": str(args.state_output),
        "selection_rule": "validation-only morphology score combining cleaned-target PCC, derivative PCC, state F1, rest RMS, and peak calibration",
        "target": args.target,
        "movement_threshold": args.movement_threshold,
        "gate_modes": args.gate_modes,
        "selection": selections,
        "validation_cleaned_metrics": trajectory_metrics(validation_prediction, validation_target),
        "test_cleaned_metrics": trajectory_metrics(test_prediction, test_target),
        "validation_raw_metrics": trajectory_metrics(validation_prediction, validation_raw),
        "test_raw_metrics": trajectory_metrics(test_prediction, test_raw),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "validation_prediction.npy", validation_prediction, allow_pickle=False)
    np.save(args.output / "test_prediction.npy", test_prediction, allow_pickle=False)
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
