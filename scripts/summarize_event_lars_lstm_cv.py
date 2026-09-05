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
    return float(np.corrcoef(np.asarray(x), np.asarray(y))[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=Path("outputs/event_lars_lstm_wavelet_v1"))
    parser.add_argument("--fold-root", type=Path, default=Path("outputs/event_stratified_folds_v1"))
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1))
    parser.add_argument("--subjects", type=int, nargs="+", default=(1, 2, 3))
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
        figure, axes = plt.subplots(5, 1, figsize=(16, 13), sharex=True)
        for finger_index, finger in enumerate(FINGER_NAMES):
            definition = json.loads(
                (args.fold_root / f"sub{subject}" / finger / "folds.json").read_text()
            )
            rows = int(definition["training_rows"])
            raw = np.full(rows, np.nan, dtype=np.float32)
            cleaned = np.full(rows, np.nan, dtype=np.float32)
            seed_predictions = {
                seed: np.full(rows, np.nan, dtype=np.float32) for seed in args.seeds
            }
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
                    summary = json.loads((root / "summary.json").read_text())
                    fold_scores[str(seed)].append(pearson(prediction, target))
                    best_epochs[str(seed)].append(
                        int(summary.get("best_epoch", summary.get("selected_epoch", 0)))
                    )
            if not np.isfinite(raw).all() or not np.isfinite(cleaned).all() or any(
                not np.isfinite(values).all() for values in seed_predictions.values()
            ):
                raise RuntimeError(f"incomplete OOF coverage for S{subject} {finger}")
            ensemble = np.mean(np.stack(list(seed_predictions.values())), axis=0)
            seed_scores = {
                str(seed): pearson(values, raw)
                for seed, values in seed_predictions.items()
            }
            ensemble_score = pearson(ensemble, raw)
            ensemble_cleaned_score = pearson(ensemble, cleaned)
            seed_sd = float(np.std(list(seed_scores.values()), ddof=1)) if len(seed_scores) > 1 else 0.0
            subject_report["per_finger"][finger] = {
                "seed_oof_pcc": seed_scores,
                "seed_sd": seed_sd,
                "ensemble_oof_pcc": ensemble_score,
                "ensemble_cleaned_target_oof_pcc": ensemble_cleaned_score,
                "fold_pcc": fold_scores,
                "best_epochs": best_epochs,
            }
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
        destination = args.input_root / f"sub{subject}"
        destination.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination / "oof_ensemble_trajectories.png", dpi=160)
        plt.close(figure)
        report["subjects"][str(subject)] = subject_report
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
