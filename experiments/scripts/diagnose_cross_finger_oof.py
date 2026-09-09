#!/usr/bin/env python3
"""Diagnose cross-finger interference from per-finger OOF predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ecog_decoding.training import FINGER_NAMES


TARGETS = {1: "local_w2_q10", 2: "local_w1_q10", 3: "local_w4_q20"}


def pearson_matrix(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    output = np.empty((prediction.shape[1], target.shape[1]), dtype=np.float64)
    for decoder in range(prediction.shape[1]):
        for observed in range(target.shape[1]):
            output[decoder, observed] = np.corrcoef(
                prediction[:, decoder], target[:, observed]
            )[0, 1]
    return output


def contiguous_events(active: np.ndarray, minimum_samples: int = 3) -> list[tuple[int, int]]:
    padded = np.r_[False, active, False]
    starts = np.flatnonzero(~padded[:-1] & padded[1:])
    stops = np.flatnonzero(padded[:-1] & ~padded[1:])
    return [
        (int(start), int(stop))
        for start, stop in zip(starts, stops)
        if stop - start >= minimum_samples
    ]


def reconstruct_oof(
    input_root: Path,
    fold_root: Path,
    subject: int,
    seeds: list[int],
) -> np.ndarray:
    rows = None
    columns: list[np.ndarray] = []
    for finger in FINGER_NAMES:
        definition = json.loads(
            (fold_root / f"sub{subject}" / finger / "folds.json").read_text()
        )
        finger_rows = int(definition["training_rows"])
        if rows is None:
            rows = finger_rows
        elif rows != finger_rows:
            raise RuntimeError("per-finger fold definitions disagree on row count")
        per_seed = [np.full(finger_rows, np.nan, dtype=np.float32) for _ in seeds]
        for fold in range(3):
            reference = input_root / f"sub{subject}" / finger / f"fold{fold}"
            indices = np.load(reference / f"seed{seeds[0]}" / "validation_indices.npy")
            for seed_index, seed in enumerate(seeds):
                prediction = np.load(
                    reference / f"seed{seed}" / "validation_prediction.npy"
                )
                per_seed[seed_index][indices] = prediction
        if any(not np.isfinite(values).all() for values in per_seed):
            raise RuntimeError(f"incomplete OOF predictions for S{subject} {finger}")
        columns.append(np.mean(np.stack(per_seed), axis=0))
    return np.column_stack(columns)


def robust_evidence(prediction: np.ndarray, rest: np.ndarray) -> np.ndarray:
    baseline = np.median(prediction[rest], axis=0)
    mad = np.median(np.abs(prediction[rest] - baseline), axis=0)
    scale = np.maximum(1.4826 * mad, 1e-4)
    return (prediction - baseline) / scale


def render_heatmaps(
    path: Path,
    raw_pcc: np.ndarray,
    cleaned_pcc: np.ndarray,
    confusion: np.ndarray,
    subject: int,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    panels = (
        (raw_pcc, "OOF decoder vs raw glove PCC", "coolwarm", -1.0, 1.0),
        (cleaned_pcc, "OOF decoder vs cleaned target PCC", "coolwarm", -1.0, 1.0),
        (confusion, "Event winner fraction", "Blues", 0.0, 1.0),
    )
    for axis, (matrix, title, cmap, lower, upper) in zip(axes, panels):
        image = axis.imshow(matrix, cmap=cmap, vmin=lower, vmax=upper)
        axis.set_xticks(range(5), FINGER_NAMES, rotation=35, ha="right")
        axis.set_yticks(range(5), FINGER_NAMES)
        axis.set_title(title)
        axis.set_xlabel("observed/dominant glove finger")
        axis.set_ylabel("neural decoder / winning decoder")
        for row in range(5):
            for column in range(5):
                axis.text(
                    column,
                    row,
                    f"{matrix[row, column]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if abs(matrix[row, column]) > 0.55 else "black",
                )
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.suptitle(f"Subject {subject}: cross-finger OOF diagnosis")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def render_traces(
    path: Path,
    cleaned: np.ndarray,
    evidence: np.ndarray,
    subject: int,
) -> None:
    target_scale = np.maximum(np.quantile(cleaned, 0.99, axis=0), 1e-4)
    normalized_target = cleaned / target_scale
    figure, axes = plt.subplots(5, 1, figsize=(17, 12), sharex=True)
    time = np.arange(cleaned.shape[0]) / 25.0
    for finger, axis in enumerate(axes):
        other = evidence.copy()
        other[:, finger] = -np.inf
        competitor = np.argmax(other, axis=1)
        competitor_strength = np.max(other, axis=1)
        axis.plot(time, normalized_target[:, finger], color="black", linewidth=0.8, label="cleaned movement")
        axis.plot(time, evidence[:, finger], color="#2563eb", linewidth=0.7, label="own OOF evidence")
        axis.plot(time, competitor_strength, color="#dc2626", linewidth=0.55, alpha=0.65, label="strongest competing evidence")
        axis.set_ylim(-3, min(15, max(5, float(np.nanquantile(evidence, 0.995)))))
        axis.set_ylabel(FINGER_NAMES[finger])
        if finger == 0:
            axis.legend(frameon=False, ncol=3)
    axes[-1].set_xlabel("training-partition time (s)")
    figure.suptitle(f"Subject {subject}: target and robustly standardized OOF neural evidence")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", type=int, nargs="+", default=(1, 2, 3))
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1))
    parser.add_argument("--input-root", type=Path, default=Path("outputs/event_lars_lstm_wavelet_v1"))
    parser.add_argument("--fold-root", type=Path, default=Path("outputs/event_stratified_folds_v1"))
    parser.add_argument("--prepared-root", type=Path, default=Path("outputs/preprocessed_v2"))
    args = parser.parse_args()

    for subject in args.subjects:
        prediction = reconstruct_oof(
            args.input_root, args.fold_root, subject, list(args.seeds)
        )
        rows = prediction.shape[0]
        prepared = args.prepared_root / f"sub{subject}"
        raw = np.load(prepared / "train_glove_25hz_raw.npy")[24 : 24 + rows]
        cleaned = np.load(
            prepared / f"train_glove_{TARGETS[subject]}.npy"
        )[24 : 24 + rows]
        thresholds = np.maximum(0.08, 0.20 * np.quantile(cleaned, 0.99, axis=0))
        active_by_finger = cleaned > thresholds
        rest = ~active_by_finger.any(axis=1)
        evidence = robust_evidence(prediction, rest)

        union_active = active_by_finger.any(axis=1)
        # Bridge two-sample gaps so one flexion burst is treated as one event.
        for lag in (1, 2):
            union_active[lag:] |= union_active[:-lag]
            union_active[:-lag] |= union_active[lag:]
        events = contiguous_events(union_active)
        confusion_counts = np.zeros((5, 5), dtype=np.int64)
        event_records: list[dict[str, object]] = []
        for start, stop in events:
            glove_strength = np.max(cleaned[start:stop], axis=0) / np.maximum(
                np.quantile(cleaned, 0.99, axis=0), 1e-4
            )
            neural_strength = np.quantile(evidence[start:stop], 0.90, axis=0)
            dominant = int(np.argmax(glove_strength))
            winner = int(np.argmax(neural_strength))
            ordered = np.sort(neural_strength)
            confusion_counts[winner, dominant] += 1
            event_records.append(
                {
                    "start": start,
                    "stop": stop,
                    "dominant_glove_finger": FINGER_NAMES[dominant],
                    "winning_decoder": FINGER_NAMES[winner],
                    "winner_margin": float(ordered[-1] - ordered[-2]),
                    "glove_strength": glove_strength.tolist(),
                    "neural_strength": neural_strength.tolist(),
                }
            )
        confusion = confusion_counts / np.maximum(confusion_counts.sum(axis=0, keepdims=True), 1)
        raw_pcc = pearson_matrix(prediction, raw)
        cleaned_pcc = pearson_matrix(prediction, cleaned)
        destination = args.input_root / f"sub{subject}" / "cross_finger_diagnosis"
        destination.mkdir(parents=True, exist_ok=True)
        render_heatmaps(
            destination / "cross_finger_matrices.png",
            raw_pcc,
            cleaned_pcc,
            confusion,
            subject,
        )
        render_traces(
            destination / "oof_evidence_traces.png", cleaned, evidence, subject
        )
        report = {
            "subject": subject,
            "protocol": "diagnostic only; seed-ensemble out-of-fold decoder evidence",
            "raw_glove_pcc_decoder_by_observed": raw_pcc.tolist(),
            "cleaned_target_pcc_decoder_by_observed": cleaned_pcc.tolist(),
            "event_winner_fraction_decoder_by_dominant_glove": confusion.tolist(),
            "event_count_by_dominant_glove": confusion_counts.sum(axis=0).tolist(),
            "rest_rows": int(rest.sum()),
            "events": event_records,
        }
        (destination / "diagnosis.json").write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "saved": str(destination),
                    "subject": subject,
                    "raw_glove_pcc_decoder_by_observed": raw_pcc.tolist(),
                    "cleaned_target_pcc_decoder_by_observed": cleaned_pcc.tolist(),
                    "event_winner_fraction_decoder_by_dominant_glove": confusion.tolist(),
                    "event_count_by_dominant_glove": confusion_counts.sum(axis=0).tolist(),
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
