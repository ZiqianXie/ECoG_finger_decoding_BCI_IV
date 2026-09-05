#!/usr/bin/env python3
"""Aggregate event-fold seed ensembles and render out-of-fold trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ecog_decoding.training import FINGER_NAMES


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(x @ y / denominator) if denominator > 0 else 0.0


def concordance(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    covariance = float(np.mean((x - x.mean()) * (y - y.mean())))
    denominator = float(x.var() + y.var() + (x.mean() - y.mean()) ** 2)
    return 2.0 * covariance / denominator if denominator > 0 else 0.0


def binary_f1(predicted: np.ndarray, observed: np.ndarray) -> float:
    true_positive = int(np.sum(predicted & observed))
    false_positive = int(np.sum(predicted & ~observed))
    false_negative = int(np.sum(~predicted & observed))
    denominator = 2 * true_positive + false_positive + false_negative
    return 2.0 * true_positive / denominator if denominator else 0.0


def morphology_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    groups: list[dict[str, int]],
    movement_threshold: float = 0.08,
    rest_threshold: float = 0.04,
) -> dict[str, float]:
    moving = target >= movement_threshold
    resting = target <= rest_threshold
    same_group = np.zeros(target.size - 1, dtype=bool)
    for group in groups:
        start, stop = int(group["start"]), int(group["stop"])
        same_group[start : max(start, stop - 1)] = True
    target_velocity = np.diff(target)[same_group]
    predicted_velocity = np.diff(prediction)[same_group]
    event_ratios: list[float] = []
    for group in groups:
        start, stop = int(group["start"]), int(group["stop"])
        target_amplitude = float(np.max(target[start:stop]))
        if target_amplitude >= movement_threshold:
            event_ratios.append(
                float(np.max(prediction[start:stop])) / max(target_amplitude, 1.0e-8)
            )
    return {
        "cleaned_pcc": pearson(prediction, target),
        "cleaned_ccc": concordance(prediction, target),
        "rmse": float(np.sqrt(np.mean(np.square(prediction - target)))),
        "movement_rmse": float(np.sqrt(np.mean(np.square(prediction[moving] - target[moving])))),
        "rest_rms": float(np.sqrt(np.mean(np.square(prediction[resting])))),
        "movement_state_f1": binary_f1(prediction >= movement_threshold, moving),
        "velocity_pcc": pearson(predicted_velocity, target_velocity),
        "prediction_to_target_sd_ratio": float(np.std(prediction)) / max(float(np.std(target)), 1.0e-8),
        "negative_prediction_fraction": float(np.mean(prediction < 0.0)),
        "median_event_peak_ratio": float(np.median(event_ratios)),
    }


def plot_event_windows(
    path: Path,
    groups: list[dict[str, int]],
    target: np.ndarray,
    prediction: np.ndarray,
    title: str,
) -> None:
    selected = sorted(
        groups,
        key=lambda group: float(
            np.max(target[int(group["start"]):int(group["stop"])])
        ),
        reverse=True,
    )[:12]
    figure, axes = plt.subplots(4, 3, figsize=(14, 10))
    for axis, group in zip(axes.flat, selected):
        start, stop = int(group["start"]), int(group["stop"])
        time = np.arange(start, stop) / 25.0
        axis.plot(time, target[start:stop], color="black", linewidth=0.9, label="target")
        axis.plot(time, prediction[start:stop], color="#2563eb", linewidth=0.9, label="OOF")
        axis.set_title(f"{start / 25:.1f}-{stop / 25:.1f} s", fontsize=9)
    for axis in axes.flat[len(selected):]:
        axis.set_visible(False)
    axes.flat[0].legend(frameon=False, ncol=2, fontsize=8)
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def plot_hurdle_event_windows(
    path: Path,
    groups: list[dict[str, int]],
    target: np.ndarray,
    prediction: np.ndarray,
    probability: np.ndarray,
    amplitude: np.ndarray,
    title: str,
) -> None:
    selected = sorted(
        groups,
        key=lambda group: float(
            np.max(target[int(group["start"]):int(group["stop"])])
        ),
        reverse=True,
    )[:12]
    figure, axes = plt.subplots(4, 3, figsize=(14, 10))
    for axis, group in zip(axes.flat, selected):
        start, stop = int(group["start"]), int(group["stop"])
        time = np.arange(start, stop) / 25.0
        axis.plot(time, target[start:stop], color="black", linewidth=0.9, label="target")
        axis.plot(time, prediction[start:stop], color="#2563eb", linewidth=0.9, label="gate × amp")
        axis.plot(time, amplitude[start:stop], color="#7c3aed", linewidth=0.7, linestyle="--", label="amplitude")
        axis.plot(time, probability[start:stop], color="#db2777", linewidth=0.7, alpha=0.8, label="P(move)")
        axis.set_title(f"{start / 25:.1f}-{stop / 25:.1f} s", fontsize=9)
    for axis in axes.flat[len(selected):]:
        axis.set_visible(False)
    axes.flat[0].legend(frameon=False, ncol=2, fontsize=7)
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=Path("outputs/event_lars_lstm_wavelet_v1"))
    parser.add_argument("--fold-root", type=Path, default=Path("outputs/event_stratified_folds_v1"))
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1))
    parser.add_argument("--subjects", type=int, nargs="+", default=(1, 2, 3))
    parser.add_argument("--fingers", nargs="+", choices=tuple(FINGER_NAMES), default=tuple(FINGER_NAMES))
    parser.add_argument("--output", type=Path, default=Path("outputs/event_lars_lstm_wavelet_v1/summary.json"))
    args = parser.parse_args()

    report: dict[str, object] = {
        "protocol": "per-finger purged event-grouped out-of-fold seed ensemble",
        "released_test_touched": False,
        "official_final_validation_touched": False,
        "subjects": {},
    }
    for subject in args.subjects:
        subject_report: dict[str, object] = {"per_finger": {}}
        figure, axes = plt.subplots(len(args.fingers), 1, figsize=(16, 2.5 * len(args.fingers) + 1), sharex=True)
        axes = np.atleast_1d(axes)
        for finger_index, finger in enumerate(args.fingers):
            definition = json.loads(
                (args.fold_root / f"sub{subject}" / finger / "folds.json").read_text()
            )
            rows = int(definition["training_rows"])
            raw = np.full(rows, np.nan, dtype=np.float32)
            cleaned = np.full(rows, np.nan, dtype=np.float32)
            seed_predictions = {
                seed: np.full(rows, np.nan, dtype=np.float32) for seed in args.seeds
            }
            reference_root = (
                args.input_root / f"sub{subject}" / finger / "fold0"
                / f"seed{args.seeds[0]}"
            )
            has_hurdle = (reference_root / "validation_movement_probability.npy").exists()
            seed_probabilities = (
                {seed: np.full(rows, np.nan, dtype=np.float32) for seed in args.seeds}
                if has_hurdle else {}
            )
            seed_amplitudes = (
                {seed: np.full(rows, np.nan, dtype=np.float32) for seed in args.seeds}
                if has_hurdle else {}
            )
            fold_scores: dict[str, list[float]] = {str(seed): [] for seed in args.seeds}
            best_epochs: dict[str, list[int]] = {str(seed): [] for seed in args.seeds}
            for fold in range(3):
                reference = (
                    args.input_root / f"sub{subject}" / finger / f"fold{fold}" / f"seed{args.seeds[0]}"
                )
                indices = np.load(reference / "validation_indices.npy")
                target = np.load(reference / "validation_raw_target.npy")
                cleaned_target = np.load(reference / "validation_cleaned_target.npy")
                if np.any(np.isfinite(raw[indices])):
                    raise RuntimeError(f"overlapping OOF rows for S{subject} {finger}")
                raw[indices] = target
                cleaned[indices] = cleaned_target
                for seed in args.seeds:
                    root = args.input_root / f"sub{subject}" / finger / f"fold{fold}" / f"seed{seed}"
                    prediction = np.load(root / "validation_prediction.npy")
                    seed_predictions[seed][indices] = prediction
                    if has_hurdle:
                        seed_probabilities[seed][indices] = np.load(
                            root / "validation_movement_probability.npy"
                        )
                        seed_amplitudes[seed][indices] = np.load(
                            root / "validation_conditional_amplitude.npy"
                        )
                    summary = json.loads((root / "summary.json").read_text())
                    fold_scores[str(seed)].append(pearson(prediction, target))
                    best_epochs[str(seed)].append(
                        int(summary.get("best_epoch", summary.get("selected_epoch", 0)))
                    )
            if not np.isfinite(raw).all() or not np.isfinite(cleaned).all() or any(
                not np.isfinite(values).all() for values in seed_predictions.values()
            ):
                raise RuntimeError(f"incomplete OOF coverage for S{subject} {finger}")
            seed_scores = {
                str(seed): pearson(values, raw)
                for seed, values in seed_predictions.items()
            }
            target_std = max(float(np.std(cleaned)), 1.0e-8)
            collapsed_seeds = [
                seed
                for seed, values in seed_predictions.items()
                if not np.isfinite(values).all()
                or float(np.std(values)) < max(1.0e-4, 0.05 * target_std)
            ]
            included_seeds = [seed for seed in args.seeds if seed not in collapsed_seeds]
            if not included_seeds:
                included_seeds = list(args.seeds)
            ensemble = np.mean(
                np.stack([seed_predictions[seed] for seed in included_seeds]), axis=0
            )
            ensemble_score = pearson(ensemble, raw)
            ensemble_cleaned_score = pearson(ensemble, cleaned)
            seed_sd = float(np.std(list(seed_scores.values()), ddof=1)) if len(seed_scores) > 1 else 0.0
            subject_report["per_finger"][finger] = {
                "seed_oof_pcc": seed_scores,
                "seed_sd": seed_sd,
                "included_seeds": included_seeds,
                "collapsed_seeds": collapsed_seeds,
                "ensemble_oof_pcc": ensemble_score,
                "ensemble_cleaned_target_oof_pcc": ensemble_cleaned_score,
                "morphology": morphology_metrics(
                    ensemble, cleaned, definition["groups"]
                ),
                "fold_pcc": fold_scores,
                "best_epochs": best_epochs,
            }
            destination = args.input_root / f"sub{subject}"
            destination.mkdir(parents=True, exist_ok=True)
            if has_hurdle:
                if any(
                    not np.isfinite(values).all()
                    for values in (*seed_probabilities.values(), *seed_amplitudes.values())
                ):
                    raise RuntimeError(f"incomplete hurdle OOF coverage for S{subject} {finger}")
                probability = np.mean(
                    np.stack([seed_probabilities[seed] for seed in included_seeds]), axis=0
                )
                amplitude = np.mean(
                    np.stack([seed_amplitudes[seed] for seed in included_seeds]), axis=0
                )
                moving = cleaned >= 0.08
                predicted_moving = probability >= 0.5
                subject_report["per_finger"][finger]["hurdle"] = {
                    "movement_state_f1": binary_f1(predicted_moving, moving),
                    "mean_rest_probability": float(np.mean(probability[~moving])),
                    "mean_movement_probability": float(np.mean(probability[moving])),
                    "conditional_amplitude_movement_rmse": float(
                        np.sqrt(np.mean(np.square(amplitude[moving] - cleaned[moving])))
                    ),
                }
                np.save(destination / f"{finger}_oof_movement_probability.npy", probability)
                np.save(destination / f"{finger}_oof_conditional_amplitude.npy", amplitude)
                plot_hurdle_event_windows(
                    destination / f"{finger}_oof_hurdle_event_windows.png",
                    definition["groups"], cleaned, ensemble, probability, amplitude,
                    f"Subject {subject} {finger}: hurdle components by event",
                )
            plot_event_windows(
                destination / f"{finger}_oof_event_windows.png",
                definition["groups"],
                cleaned,
                ensemble,
                f"Subject {subject} {finger}: highest-amplitude event groups",
            )
            time = np.arange(rows) / 25.0
            axes[finger_index].plot(time, cleaned, color="black", linewidth=0.8, label="cleaned movement target")
            axes[finger_index].plot(time, ensemble, color="#2563eb", linewidth=0.8, label="OOF ensemble")
            axes[finger_index].set_ylabel(finger)
            axes[finger_index].text(
                0.995,
                0.92,
                f"raw PCC={ensemble_score:.3f}; cleaned PCC={ensemble_cleaned_score:.3f}",
                transform=axes[finger_index].transAxes,
                ha="right",
                va="top",
            )
        axes[0].legend(frameon=False, ncol=2)
        axes[-1].set_xlabel("training-partition time (s)")
        figure.suptitle(f"Subject {subject}: purged event-fold out-of-fold ensemble")
        figure.tight_layout()
        figure.savefig(destination / "oof_ensemble_trajectories.png", dpi=160)
        plt.close(figure)
        report["subjects"][str(subject)] = subject_report
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
