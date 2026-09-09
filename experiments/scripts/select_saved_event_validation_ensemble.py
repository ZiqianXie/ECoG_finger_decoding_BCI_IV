#!/usr/bin/env python3
"""Select equal-weight saved-decoder ensembles with event-balanced CV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from apply_validation_latent_gate import event_balanced_folds
from ecog_decoding.training import FINGER_NAMES
from select_validation_latent_gate_candidate import discover
from summarize_event_lars_lstm_cv import pearson
from summarize_frozen_full_refit import PAPER_PCC


def affine(
    fit_values: np.ndarray, fit_target: np.ndarray, values: np.ndarray
) -> np.ndarray:
    source_sd = float(np.std(fit_values))
    scale = float(np.std(fit_target)) / source_sd if source_sd > 1.0e-8 else 1.0
    offset = float(np.mean(fit_target) - scale * np.mean(fit_values))
    return scale * values + offset


def event_oof(
    values: np.ndarray, target: np.ndarray, folds: list[np.ndarray]
) -> np.ndarray:
    output = np.empty_like(values, dtype=np.float64)
    for validation in folds:
        training = ~validation
        output[validation] = affine(
            values[training], target[training], values[validation]
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", type=int, nargs="+", default=(1, 2, 3))
    parser.add_argument("--root", type=Path, default=Path("outputs"))
    parser.add_argument("--prepared-root", type=Path, default=Path("outputs/preprocessed_v2"))
    parser.add_argument("--target-map", type=Path, default=Path("configs/targetsafe_conservative_targets.yaml"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/saved_event_validation_ensemble_v1"))
    parser.add_argument("--max-members", type=int, default=12)
    parser.add_argument("--minimum-oof-gain", type=float, default=5.0e-4)
    parser.add_argument("--residual-correlation-ceiling", type=float, default=0.995)
    parser.add_argument("--movement-threshold", type=float, default=0.08)
    args = parser.parse_args()
    target_map = yaml.safe_load(args.target_map.read_text())
    report: dict[str, object] = {
        "protocol": "equal-weight candidate ensemble selected by complete-event validation CV",
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
            np.load(prepared / f"train_glove_{subject_targets[name]}.npy")[split:, finger]
            for finger, name in enumerate(FINGER_NAMES)
        ])
        candidates = discover(args.root, subject, raw_validation.shape, raw_test.shape)
        validation_output = np.empty_like(raw_validation, dtype=np.float64)
        test_output = np.empty_like(raw_test, dtype=np.float64)
        per_finger = {}
        for finger, name in enumerate(FINGER_NAMES):
            target = raw_validation[:, finger]
            state = (cleaned_validation[:, finger] >= args.movement_threshold).astype(np.int64)
            folds = event_balanced_folds(state)
            pool = []
            for candidate in candidates:
                values = np.asarray(candidate["validation"][:, finger], dtype=np.float64)
                oof = event_oof(values, target, folds)
                pool.append({
                    "path": str(candidate["path"]),
                    "oof": oof,
                    "validation": values,
                    "test": np.asarray(candidate["test"][:, finger], dtype=np.float64),
                    "oof_pcc": pearson(oof, target),
                })
            pool.sort(key=lambda item: item["oof_pcc"], reverse=True)
            selected = [pool[0]]
            current = np.asarray(pool[0]["oof"])
            current_score = pearson(current, target)
            trace = []
            available = pool[1:]
            while available and len(selected) < args.max_members:
                residual = current - target
                proposals = []
                for candidate in available:
                    proposal = np.mean(
                        np.stack([*[item["oof"] for item in selected], candidate["oof"]]),
                        axis=0,
                    )
                    score = pearson(proposal, target)
                    residual_correlation = pearson(candidate["oof"] - target, residual)
                    proposals.append((score - current_score, -residual_correlation, candidate, proposal, score, residual_correlation))
                gain, _, candidate, proposal, score, residual_correlation = max(
                    proposals, key=lambda value: value[:2]
                )
                accepted = gain >= args.minimum_oof_gain and residual_correlation <= args.residual_correlation_ceiling
                trace.append({
                    "path": candidate["path"], "oof_gain": gain,
                    "residual_correlation": residual_correlation,
                    "accepted": accepted,
                })
                if not accepted:
                    break
                selected.append(candidate)
                available.remove(candidate)
                current = proposal
                current_score = score
            calibrated_validation = []
            calibrated_test = []
            for candidate in selected:
                calibrated_validation.append(
                    affine(candidate["validation"], target, candidate["validation"])
                )
                calibrated_test.append(
                    affine(candidate["validation"], target, candidate["test"])
                )
            final_validation = np.mean(np.stack(calibrated_validation), axis=0)
            final_test = np.mean(np.stack(calibrated_test), axis=0)
            validation_output[:, finger] = final_validation
            test_output[:, finger] = final_test
            score = pearson(final_test, raw_test[:, finger])
            per_finger[name] = {
                "selected_members": [item["path"] for item in selected],
                "selection_trace": trace,
                "event_oof_pcc": current_score,
                "validation_fit_pcc": pearson(final_validation, target),
                "test_raw_pcc": score,
                "paper_raw_pcc": PAPER_PCC[subject][finger],
                "delta_vs_paper": score - PAPER_PCC[subject][finger],
            }
        destination = args.output_root / f"sub{subject}"
        destination.mkdir(parents=True, exist_ok=True)
        np.save(destination / "validation_prediction.npy", validation_output)
        np.save(destination / "test_prediction.npy", test_output)
        report["subjects"][str(subject)] = {
            "candidate_count": len(candidates),
            "per_finger": per_finger,
            "test_raw_pcc": [per_finger[name]["test_raw_pcc"] for name in FINGER_NAMES],
        }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        subject: report["subjects"][str(subject)]["test_raw_pcc"]
        for subject in args.subjects
    }, indent=2))


if __name__ == "__main__":
    main()
