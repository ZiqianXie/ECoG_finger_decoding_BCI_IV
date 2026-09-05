#!/usr/bin/env python3
"""Learn a fixed-wavelet ECoG residual on top of a saved trajectory decoder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from apply_validation_latent_gate import event_balanced_folds
from ecog_decoding.training import FINGER_NAMES
from fit_validation_ecog_state_gate import correlation_screen
from summarize_event_lars_lstm_cv import pearson
from summarize_frozen_full_refit import PAPER_PCC


def fit(
    features: np.ndarray, residual: np.ndarray, mask: np.ndarray,
    selected: np.ndarray, alpha: float,
) -> tuple[StandardScaler, Ridge]:
    scaler = StandardScaler()
    values = scaler.fit_transform(np.asarray(features[mask][:, selected]))
    model = Ridge(alpha=alpha, solver="cholesky")
    model.fit(values, residual[mask])
    return scaler, model


def predict(
    scaler: StandardScaler, model: Ridge, features: np.ndarray,
    selected: np.ndarray,
) -> np.ndarray:
    return np.asarray(
        model.predict(scaler.transform(np.asarray(features[:, selected]))),
        dtype=np.float64,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", type=int, nargs="+", default=(1, 3))
    parser.add_argument("--fingers", nargs="+", choices=tuple(FINGER_NAMES), default=("thumb", "little"))
    parser.add_argument("--base-map", type=Path, default=Path("configs/retrospective_base_predictions.yaml"))
    parser.add_argument("--target-map", type=Path, default=Path("configs/targetsafe_conservative_targets.yaml"))
    parser.add_argument("--prepared-root", type=Path, default=Path("outputs/preprocessed_v2"))
    parser.add_argument("--feature-root", type=Path, default=Path("outputs/windowed_ica_wavelet_asymmetric_v1"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/validation_ecog_residual_v1"))
    parser.add_argument("--screen-count", type=int, default=512)
    parser.add_argument("--movement-threshold", type=float, default=0.08)
    parser.add_argument("--minimum-validation-cv-gain", type=float, default=0.005)
    args = parser.parse_args()
    base_map = yaml.safe_load(args.base_map.read_text())
    target_map = yaml.safe_load(args.target_map.read_text())
    alphas = tuple(float(value) for value in np.logspace(-2, 5, 8))
    strengths = (0.0, 0.25, 0.5, 0.75, 1.0)
    report: dict[str, object] = {
        "protocol": "fixed-wavelet ECoG residual selected by complete-event validation CV",
        "released_test_used_for_selection": False,
        "subjects": {},
    }
    for subject in args.subjects:
        prepared = args.prepared_root / f"sub{subject}"
        metadata = json.loads((prepared / "metadata.json").read_text())
        split = int(metadata["target_fit_samples_25hz"])
        feature_split = split - 24
        validation_features = np.load(
            args.feature_root / f"sub{subject}" / "train_initialized_window_features.npy",
            mmap_mode="r",
        )[feature_split:]
        test_features = np.load(
            args.feature_root / f"sub{subject}" / "test_initialized_window_features.npy",
            mmap_mode="r",
        )
        raw_train = np.load(prepared / "train_glove_25hz_raw.npy")
        raw_validation = raw_train[split:]
        raw_test = np.load(prepared / "test_glove_25hz_raw.npy")[24:]
        subject_targets = target_map.get(subject, target_map.get(str(subject)))
        cleaned_validation = np.column_stack([
            np.load(prepared / f"train_glove_{subject_targets[name]}.npy")[split:, finger]
            for finger, name in enumerate(FINGER_NAMES)
        ])
        base_root = Path(str(base_map.get(subject, base_map.get(str(subject)))))
        base_validation = np.load(base_root / "validation_prediction.npy")
        base_test = np.load(base_root / "test_prediction.npy")
        output = base_test.astype(np.float64, copy=True)
        per_finger = {}
        for name in args.fingers:
            finger = list(FINGER_NAMES).index(name)
            residual = raw_validation[:, finger] - base_validation[:, finger]
            state = (cleaned_validation[:, finger] >= args.movement_threshold).astype(np.int64)
            folds = event_balanced_folds(state)
            scores = {(alpha, strength): [] for alpha in alphas for strength in strengths}
            for validation in folds:
                training = ~validation
                selected = correlation_screen(
                    validation_features, residual, training, args.screen_count
                )
                for alpha in alphas:
                    scaler, model = fit(
                        validation_features, residual, training, selected, alpha
                    )
                    correction = predict(
                        scaler, model, validation_features[validation], selected
                    )
                    for strength in strengths:
                        estimate = base_validation[validation, finger] + strength * correction
                        scores[(alpha, strength)].append(
                            pearson(estimate, raw_validation[validation, finger])
                        )
            means = {key: float(np.mean(value)) for key, value in scores.items()}
            baseline = max(means[(alpha, 0.0)] for alpha in alphas)
            best = max(means, key=means.get)
            if means[best] < baseline + args.minimum_validation_cv_gain:
                best = (alphas[-1], 0.0)
            alpha, strength = best
            if strength:
                full_mask = np.ones(residual.size, dtype=bool)
                selected = correlation_screen(
                    validation_features, residual, full_mask, args.screen_count
                )
                scaler, model = fit(
                    validation_features, residual, full_mask, selected, alpha
                )
                output[:, finger] += strength * predict(
                    scaler, model, test_features, selected
                )
            score = pearson(output[:, finger], raw_test[:, finger])
            per_finger[name] = {
                "baseline_validation_cv_pcc": baseline,
                "selected_validation_cv_pcc": means[best],
                "selected_alpha": alpha,
                "selected_strength": strength,
                "test_raw_pcc": score,
                "paper_raw_pcc": PAPER_PCC[subject][finger],
                "delta_vs_paper": score - PAPER_PCC[subject][finger],
            }
        destination = args.output_root / f"sub{subject}"
        destination.mkdir(parents=True, exist_ok=True)
        np.save(destination / "test_prediction.npy", output)
        report["subjects"][str(subject)] = {
            "base": str(base_root), "per_finger": per_finger,
        }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
