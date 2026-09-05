#!/usr/bin/env python3
"""Gate retrospective trajectories with a direct ECoG movement classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from apply_validation_latent_gate import event_balanced_folds
from ecog_decoding.training import FINGER_NAMES
from summarize_event_lars_lstm_cv import pearson
from summarize_frozen_full_refit import PAPER_PCC


def correlation_screen(
    features: np.ndarray, label: np.ndarray, mask: np.ndarray, count: int
) -> np.ndarray:
    values = np.asarray(features[mask], dtype=np.float64)
    target = np.asarray(label[mask], dtype=np.float64)
    values -= values.mean(axis=0, keepdims=True)
    target -= target.mean()
    numerator = target @ values
    denominator = np.linalg.norm(target) * np.linalg.norm(values, axis=0)
    score = np.divide(
        numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0
    )
    return np.argsort(np.abs(score))[-min(count, score.size):]


def fit_classifier(
    features: np.ndarray, label: np.ndarray, mask: np.ndarray,
    selected: np.ndarray, c_value: float,
) -> tuple[StandardScaler, LogisticRegression]:
    scaler = StandardScaler()
    values = scaler.fit_transform(np.asarray(features[mask][:, selected]))
    model = LogisticRegression(
        C=c_value, class_weight="balanced", max_iter=1000, solver="liblinear"
    )
    model.fit(values, label[mask])
    return scaler, model


def probability(
    scaler: StandardScaler, model: LogisticRegression,
    features: np.ndarray, selected: np.ndarray,
) -> np.ndarray:
    values = scaler.transform(np.asarray(features[:, selected]))
    return model.predict_proba(values)[:, 1]


def smooth(values: np.ndarray, bins: int) -> np.ndarray:
    if bins == 1:
        return values
    kernel = np.bartlett(2 * bins + 1)[1:-1]
    kernel /= kernel.sum()
    return np.convolve(values, kernel, mode="same")


def normalized_gate(
    values: np.ndarray, movement_reference: float, strength: float
) -> np.ndarray:
    state = np.clip(values / max(movement_reference, 1.0e-3), 0.0, 1.5)
    return (1.0 - strength) + strength * state


def select_spec(
    features: np.ndarray, base: np.ndarray, raw_target: np.ndarray,
    cleaned_target: np.ndarray, finger: int, threshold: float,
    screen_count: int, minimum_gain: float,
) -> tuple[float, int, float, dict[str, object]]:
    label = cleaned_target[:, finger] >= threshold
    folds = event_balanced_folds(label.astype(np.int64))
    c_values = (0.01, 0.1, 1.0, 10.0)
    smooth_bins = (1, 3, 5, 9)
    strengths = (0.0, 0.25, 0.5, 0.75, 1.0)
    scores = {
        (c_value, bins, strength): []
        for c_value in c_values for bins in smooth_bins for strength in strengths
    }
    auc_accuracy = []
    for validation_mask in folds:
        training_mask = ~validation_mask
        selected = correlation_screen(features, label, training_mask, screen_count)
        for c_value in c_values:
            scaler, model = fit_classifier(
                features, label, training_mask, selected, c_value
            )
            raw_probability = probability(scaler, model, features, selected)
            auc_accuracy.append(float(np.mean((raw_probability[validation_mask] >= 0.5) == label[validation_mask])))
            movement_reference = float(np.mean(raw_probability[training_mask & label]))
            for bins in smooth_bins:
                smoothed = smooth(raw_probability, bins)
                for strength in strengths:
                    estimate = base[validation_mask, finger] * normalized_gate(
                        smoothed[validation_mask], movement_reference, strength
                    )
                    scores[(c_value, bins, strength)].append(
                        pearson(estimate, raw_target[validation_mask, finger])
                    )
    means = {key: float(np.mean(value)) for key, value in scores.items()}
    baseline = max(means[(c_value, bins, 0.0)] for c_value in c_values for bins in smooth_bins)
    best = max(means, key=means.get)
    if means[best] < baseline + minimum_gain:
        best = (1.0, 1, 0.0)
    return *best, {
        "baseline_validation_cv_pcc": baseline,
        "selected_validation_cv_pcc": means[best],
        "selected_c": best[0],
        "selected_smoothing_bins": best[1],
        "selected_strength": best[2],
        "mean_binary_accuracy": float(np.mean(auc_accuracy)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", type=int, nargs="+", default=(1, 3))
    parser.add_argument("--fingers", nargs="+", choices=tuple(FINGER_NAMES), default=("thumb", "little"))
    parser.add_argument("--base-map", type=Path, default=Path("configs/retrospective_base_predictions.yaml"))
    parser.add_argument("--target-map", type=Path, default=Path("configs/targetsafe_conservative_targets.yaml"))
    parser.add_argument("--prepared-root", type=Path, default=Path("outputs/preprocessed_v2"))
    parser.add_argument("--feature-root", type=Path, default=Path("outputs/windowed_ica_wavelet_asymmetric_v1"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/validation_ecog_state_gate_v1"))
    parser.add_argument("--screen-count", type=int, default=256)
    parser.add_argument("--movement-threshold", type=float, default=0.08)
    parser.add_argument("--minimum-validation-cv-gain", type=float, default=0.005)
    args = parser.parse_args()
    base_map = yaml.safe_load(args.base_map.read_text())
    target_map = yaml.safe_load(args.target_map.read_text())
    report: dict[str, object] = {
        "protocol": "direct ECoG movement classifier and gate selected by event-balanced validation CV",
        "released_test_used_for_selection": False,
        "subjects": {},
    }
    for subject in args.subjects:
        prepared = args.prepared_root / f"sub{subject}"
        metadata = json.loads((prepared / "metadata.json").read_text())
        split = int(metadata["target_fit_samples_25hz"])
        feature_split = split - 24
        train_features = np.load(
            args.feature_root / f"sub{subject}" / "train_initialized_window_features.npy",
            mmap_mode="r",
        )
        test_features = np.load(
            args.feature_root / f"sub{subject}" / "test_initialized_window_features.npy",
            mmap_mode="r",
        )
        validation_features = train_features[feature_split:]
        raw_train = np.load(prepared / "train_glove_25hz_raw.npy")
        raw_validation = raw_train[split:]
        raw_test = np.load(prepared / "test_glove_25hz_raw.npy")[24:]
        subject_targets = target_map.get(subject, target_map.get(str(subject)))
        cleaned_validation = np.column_stack([
            np.load(prepared / f"train_glove_{subject_targets[name]}.npy")[split:, finger]
            for finger, name in enumerate(FINGER_NAMES)
        ])
        full_targets = np.column_stack([
            np.load(prepared / f"train_glove_{subject_targets[name].removesuffix('_split_safe')}.npy")[24:, finger]
            for finger, name in enumerate(FINGER_NAMES)
        ])
        base_root = Path(str(base_map.get(subject, base_map.get(str(subject)))))
        base_validation = np.load(base_root / "validation_prediction.npy")
        base_test = np.load(base_root / "test_prediction.npy")
        prediction = base_test.astype(np.float64, copy=True)
        per_finger = {}
        for name in args.fingers:
            finger = list(FINGER_NAMES).index(name)
            c_value, bins, strength, audit = select_spec(
                validation_features, base_validation, raw_validation,
                cleaned_validation, finger, args.movement_threshold,
                args.screen_count, args.minimum_validation_cv_gain,
            )
            if strength:
                full_label = full_targets[:, finger] >= args.movement_threshold
                full_mask = np.ones(full_label.size, dtype=bool)
                selected = correlation_screen(
                    train_features, full_label, full_mask, args.screen_count
                )
                scaler, model = fit_classifier(
                    train_features, full_label, full_mask, selected, c_value
                )
                train_probability = probability(
                    scaler, model, train_features, selected
                )
                test_probability = smooth(
                    probability(scaler, model, test_features, selected), bins
                )
                movement_reference = float(np.mean(train_probability[full_label]))
                prediction[:, finger] *= normalized_gate(
                    test_probability, movement_reference, strength
                )
            score = pearson(prediction[:, finger], raw_test[:, finger])
            per_finger[name] = {
                **audit,
                "test_raw_pcc": score,
                "paper_raw_pcc": PAPER_PCC[subject][finger],
                "delta_vs_paper": score - PAPER_PCC[subject][finger],
            }
        destination = args.output_root / f"sub{subject}"
        destination.mkdir(parents=True, exist_ok=True)
        np.save(destination / "test_prediction.npy", prediction)
        report["subjects"][str(subject)] = {
            "base": str(base_root), "per_finger": per_finger,
        }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
