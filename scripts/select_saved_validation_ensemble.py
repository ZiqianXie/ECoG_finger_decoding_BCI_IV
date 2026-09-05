#!/usr/bin/env python3
"""Greedily select diverse saved candidates using validation labels only."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from ecog_decoding.training import FINGER_NAMES
from summarize_event_lars_lstm_cv import pearson


SUBJECT = re.compile(r"(?:^|/)sub([123])(?:/|$)")


def affine_to_target(
    validation: np.ndarray, test: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    source_sd = float(np.std(validation))
    scale = float(np.std(target)) / source_sd if source_sd > 1.0e-8 else 1.0
    offset = float(np.mean(target) - scale * np.mean(validation))
    return scale * validation + offset, scale * test + offset


def fold_score(prediction: np.ndarray, target: np.ndarray, folds: int = 5) -> float:
    return float(np.mean([
        pearson(prediction[index], target[index])
        for index in np.array_split(np.arange(target.size), folds)
    ]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("outputs"))
    parser.add_argument("--prepared-root", type=Path, default=Path("outputs/preprocessed_v2"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/saved_validation_ensemble_v1"))
    parser.add_argument("--max-members", type=int, default=10)
    parser.add_argument("--minimum-fold-gain", type=float, default=5.0e-4)
    parser.add_argument("--residual-correlation-ceiling", type=float, default=0.995)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {
        "protocol": "retrospective broad candidate search; membership selected on chronological validation only",
        "released_test_used_for_selection": False,
        "subjects": {},
    }
    for subject in (1, 2, 3):
        prepared = args.prepared_root / f"sub{subject}"
        metadata = json.loads((prepared / "metadata.json").read_text())
        split = int(metadata["target_fit_samples_25hz"])
        raw_train = np.load(prepared / "train_glove_25hz_raw.npy")
        raw_test = np.load(prepared / "test_glove_25hz_raw.npy")[24:]
        validation_target = raw_train[split:]
        candidates: list[dict[str, object]] = []
        for test_path in args.root.rglob("test_prediction*.npy"):
            match = SUBJECT.search(test_path.as_posix())
            if match is None or int(match.group(1)) != subject:
                continue
            validation_path = test_path.with_name(
                test_path.name.replace("test_", "validation_", 1)
            )
            if not validation_path.exists():
                continue
            try:
                test = np.load(test_path)
                validation = np.load(validation_path)
            except (OSError, ValueError):
                continue
            if (
                test.shape != raw_test.shape
                or validation.shape != validation_target.shape
                or test.ndim != 2
                or test.shape[1] != 5
            ):
                continue
            candidates.append({
                "test_path": test_path.as_posix(),
                "validation": validation,
                "test": test,
            })
        validation_output = np.empty_like(validation_target, dtype=np.float64)
        test_output = np.empty_like(raw_test, dtype=np.float64)
        per_finger = {}
        for finger, name in enumerate(FINGER_NAMES):
            target = validation_target[:, finger]
            pool = []
            for candidate in candidates:
                validation, test = affine_to_target(
                    candidate["validation"][:, finger],
                    candidate["test"][:, finger],
                    target,
                )
                if float(np.std(validation)) < 1.0e-5:
                    continue
                pool.append({
                    "path": candidate["test_path"],
                    "validation": validation,
                    "test": test,
                    "individual_full_pcc": pearson(validation, target),
                    "individual_fold_pcc": fold_score(validation, target),
                })
            pool.sort(key=lambda item: (item["individual_fold_pcc"], item["individual_full_pcc"]), reverse=True)
            selected = [pool[0]]
            current_validation = np.asarray(pool[0]["validation"])
            current_full = pearson(current_validation, target)
            current_fold = fold_score(current_validation, target)
            trace = []
            available = pool[1:]
            while available and len(selected) < args.max_members:
                residual = current_validation - target
                proposals = []
                for candidate in available:
                    estimate = np.mean(
                        np.stack([*[item["validation"] for item in selected], candidate["validation"]]),
                        axis=0,
                    )
                    full = pearson(estimate, target)
                    blocked = fold_score(estimate, target)
                    residual_correlation = pearson(
                        candidate["validation"] - target, residual
                    )
                    proposals.append((
                        blocked - current_fold,
                        full - current_full,
                        -residual_correlation,
                        candidate,
                        estimate,
                        full,
                        blocked,
                        residual_correlation,
                    ))
                proposal = max(proposals, key=lambda value: value[:3])
                fold_gain, full_gain, _, candidate, estimate, full, blocked, residual_correlation = proposal
                accepted = (
                    fold_gain >= args.minimum_fold_gain
                    and full_gain >= 0.0
                    and residual_correlation <= args.residual_correlation_ceiling
                )
                trace.append({
                    "path": candidate["path"],
                    "fold_gain": fold_gain,
                    "full_gain": full_gain,
                    "residual_correlation": residual_correlation,
                    "accepted": accepted,
                })
                if not accepted:
                    break
                selected.append(candidate)
                available.remove(candidate)
                current_validation = estimate
                current_full = full
                current_fold = blocked
            final_validation = np.mean(
                np.stack([item["validation"] for item in selected]), axis=0
            )
            final_test = np.mean(np.stack([item["test"] for item in selected]), axis=0)
            validation_output[:, finger] = final_validation
            test_output[:, finger] = final_test
            per_finger[name] = {
                "selected_members": [item["path"] for item in selected],
                "selection_trace": trace,
                "validation_raw_pcc": pearson(final_validation, target),
                "validation_fold_pcc": fold_score(final_validation, target),
                "test_raw_pcc": pearson(final_test, raw_test[:, finger]),
            }
        subject_output = args.output_root / f"sub{subject}"
        subject_output.mkdir(parents=True, exist_ok=True)
        np.save(subject_output / "validation_prediction.npy", validation_output)
        np.save(subject_output / "test_prediction.npy", test_output)
        report["subjects"][str(subject)] = {
            "candidate_count": len(candidates),
            "per_finger": per_finger,
            "test_raw_pcc": [per_finger[name]["test_raw_pcc"] for name in FINGER_NAMES],
        }
    (args.output_root / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        subject: report["subjects"][str(subject)]["test_raw_pcc"]
        for subject in (1, 2, 3)
    }, indent=2))


if __name__ == "__main__":
    main()
