#!/usr/bin/env python3
"""Fit a small state-aware residual stack on validation predictions only.

The trajectory path remains the selected retrospective base decoder.  A
diverse bank of saved decoders supplies both latent-state evidence and
candidate residual atoms.  Candidate choice, ridge strength, and residual
step size are evaluated with event-balanced folds before the final fit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from apply_validation_latent_gate import event_balanced_folds
from apply_validation_multibase_latent_gate import select_diverse_candidates
from ecog_decoding.training import FINGER_NAMES
from fit_oof_latent_movement_gate import (
    fit_classifier,
    intended_state,
    state_probabilities,
    temporal_features,
)
from select_validation_latent_gate_candidate import discover
from summarize_event_lars_lstm_cv import pearson
from summarize_frozen_full_refit import PAPER_PCC


ALPHAS = tuple(float(value) for value in np.logspace(-2, 5, 8))
STRENGTHS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25)


def atom_features(
    base: np.ndarray,
    candidates: list[dict[str, object]],
    probability: np.ndarray,
    finger: int,
    split: str,
) -> np.ndarray:
    bank = np.column_stack([
        np.asarray(candidate[split], dtype=np.float64)[:, finger]
        for candidate in candidates
    ])
    center = base[:, [finger]]
    delta = bank - center
    own = probability[:, [finger + 1]]
    rest = probability[:, [0]]
    return np.concatenate(
        [delta, delta * own, delta * rest, base, probability], axis=1
    )


def fit_ridge(
    features: np.ndarray,
    residual: np.ndarray,
    mask: np.ndarray,
    alpha: float,
) -> tuple[StandardScaler, Ridge]:
    scaler = StandardScaler()
    values = scaler.fit_transform(features[mask])
    model = Ridge(alpha=alpha)
    model.fit(values, residual[mask])
    return scaler, model


def predict_ridge(
    scaler: StandardScaler, model: Ridge, features: np.ndarray
) -> np.ndarray:
    return np.asarray(model.predict(scaler.transform(features)), dtype=np.float64)


def select_spec(
    *, candidates: list[dict[str, object]], base: np.ndarray,
    raw_target: np.ndarray, cleaned_target: np.ndarray, finger: int,
    candidate_count: int, correlation_ceiling: float, threshold: float,
    minimum_gain: float,
) -> tuple[float, float, dict[str, object]]:
    state = intended_state(cleaned_target, threshold)
    folds = event_balanced_folds(state)
    scores = {(alpha, strength): [] for alpha in ALPHAS for strength in STRENGTHS}
    selected_paths: list[list[str]] = []
    for validation_mask in folds:
        training_mask = ~validation_mask
        selected = select_diverse_candidates(
            candidates,
            raw_target,
            candidate_count,
            correlation_ceiling,
            mask=training_mask,
        )
        selected_paths.append([str(item["path"]) for item in selected])
        evidence = temporal_features(np.concatenate([
            np.asarray(item["validation"], dtype=np.float64) for item in selected
        ], axis=1))
        classifier_scaler, classifier = fit_classifier(evidence, state, training_mask)
        probability = state_probabilities(classifier_scaler, classifier, evidence)
        features = atom_features(base, selected, probability, finger, "validation")
        residual = raw_target[:, finger] - base[:, finger]
        for alpha in ALPHAS:
            scaler, model = fit_ridge(features, residual, training_mask, alpha)
            correction = predict_ridge(scaler, model, features[validation_mask])
            for strength in STRENGTHS:
                estimate = base[validation_mask, finger] + strength * correction
                scores[(alpha, strength)].append(
                    pearson(estimate, raw_target[validation_mask, finger])
                )
    means = {key: float(np.mean(value)) for key, value in scores.items()}
    baseline = max(means[(alpha, 0.0)] for alpha in ALPHAS)
    best = max(means, key=means.get)
    if means[best] < baseline + minimum_gain:
        best = (ALPHAS[-1], 0.0)
    alpha, strength = best
    return alpha, strength, {
        "baseline_validation_cv_pcc": baseline,
        "selected_validation_cv_pcc": means[best],
        "selected_alpha": alpha,
        "selected_strength": strength,
        "fold_candidate_paths": selected_paths,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", type=int, nargs="+", default=(1, 2, 3))
    parser.add_argument("--root", type=Path, default=Path("outputs"))
    parser.add_argument("--base-map", type=Path, default=Path("configs/retrospective_base_predictions.yaml"))
    parser.add_argument("--target-map", type=Path, default=Path("configs/targetsafe_conservative_targets.yaml"))
    parser.add_argument("--prepared-root", type=Path, default=Path("outputs/preprocessed_v2"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/validation_state_aware_residual_v1"))
    parser.add_argument("--candidate-count", type=int, default=16)
    parser.add_argument("--candidate-correlation-ceiling", type=float, default=0.995)
    parser.add_argument("--movement-threshold", type=float, default=0.08)
    parser.add_argument("--minimum-validation-cv-gain", type=float, default=0.005)
    args = parser.parse_args()
    base_map = yaml.safe_load(args.base_map.read_text())
    target_map = yaml.safe_load(args.target_map.read_text())
    report: dict[str, object] = {
        "protocol": "retrospective state-aware residual stack selected with event-balanced validation CV",
        "released_test_used_for_selection": False,
        "subjects": {},
    }
    for subject in args.subjects:
        prepared = args.prepared_root / f"sub{subject}"
        metadata = json.loads((prepared / "metadata.json").read_text())
        split = int(metadata["target_fit_samples_25hz"])
        raw_train = np.load(prepared / "train_glove_25hz_raw.npy")
        raw_validation = raw_train[split:]
        raw_test = np.load(prepared / "test_glove_25hz_raw.npy")[24:]
        subject_targets = target_map.get(subject, target_map.get(str(subject)))
        cleaned_validation = np.column_stack([
            np.load(prepared / f"train_glove_{subject_targets[name]}.npy")[-raw_validation.shape[0]:, finger]
            for finger, name in enumerate(FINGER_NAMES)
        ])
        base_root = Path(str(base_map.get(subject, base_map.get(str(subject)))))
        base_validation = np.load(base_root / "validation_prediction.npy")
        base_test = np.load(base_root / "test_prediction.npy")
        candidates = discover(args.root, subject, raw_validation.shape, raw_test.shape)
        final_candidates = select_diverse_candidates(
            candidates, raw_validation, args.candidate_count,
            args.candidate_correlation_ceiling,
        )
        final_evidence_validation = temporal_features(np.concatenate([
            np.asarray(item["validation"], dtype=np.float64) for item in final_candidates
        ], axis=1))
        final_evidence_test = temporal_features(np.concatenate([
            np.asarray(item["test"], dtype=np.float64) for item in final_candidates
        ], axis=1))
        state = intended_state(cleaned_validation, args.movement_threshold)
        full_mask = np.ones(state.size, dtype=bool)
        classifier_scaler, classifier = fit_classifier(
            final_evidence_validation, state, full_mask
        )
        validation_probability = state_probabilities(
            classifier_scaler, classifier, final_evidence_validation
        )
        test_probability = state_probabilities(
            classifier_scaler, classifier, final_evidence_test
        )
        prediction = base_test.astype(np.float64, copy=True)
        per_finger = {}
        for finger, name in enumerate(FINGER_NAMES):
            alpha, strength, audit = select_spec(
                candidates=candidates,
                base=base_validation,
                raw_target=raw_validation,
                cleaned_target=cleaned_validation,
                finger=finger,
                candidate_count=args.candidate_count,
                correlation_ceiling=args.candidate_correlation_ceiling,
                threshold=args.movement_threshold,
                minimum_gain=args.minimum_validation_cv_gain,
            )
            if strength:
                validation_features = atom_features(
                    base_validation, final_candidates, validation_probability,
                    finger, "validation",
                )
                test_features = atom_features(
                    base_test, final_candidates, test_probability, finger, "test"
                )
                residual = raw_validation[:, finger] - base_validation[:, finger]
                scaler, model = fit_ridge(
                    validation_features, residual, full_mask, alpha
                )
                prediction[:, finger] += strength * predict_ridge(
                    scaler, model, test_features
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
            "base": str(base_root),
            "final_candidate_paths": [str(item["path"]) for item in final_candidates],
            "per_finger": per_finger,
            "test_raw_pcc": [per_finger[name]["test_raw_pcc"] for name in FINGER_NAMES],
        }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
