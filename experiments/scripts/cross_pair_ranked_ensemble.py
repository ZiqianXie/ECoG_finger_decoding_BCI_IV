#!/usr/bin/env python3
"""Select an equal OOF ensemble using performance on other decoding pairs only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from ecog_decoding.training import FINGER_NAMES
from single_wavelet_support import OFFSET


ALLOWED_FILENAMES = {"initialized_oof.npy", "selected_oof.npy"}


def pearson(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.corrcoef(np.asarray(first), np.asarray(second))[0, 1])


def raw_target(prepared_root: Path, subject: int, finger: str, rows: int) -> np.ndarray:
    finger_index = list(FINGER_NAMES).index(finger)
    return np.asarray(
        np.load(
            prepared_root / f"sub{subject}" / "train_glove_25hz_raw.npy",
            mmap_mode="r",
        )[OFFSET : OFFSET + rows, finger_index],
        dtype=np.float32,
    )


def scalar_prediction(path: Path, finger: str) -> np.ndarray:
    prediction = np.asarray(np.load(path, mmap_mode="r"), dtype=np.float32)
    if prediction.ndim == 2:
        prediction = prediction[:, list(FINGER_NAMES).index(finger)]
    if prediction.ndim != 1:
        raise ValueError(f"unsupported prediction shape {prediction.shape} at {path}")
    return prediction


def source_path(target_path: Path, subject: int, finger: str) -> Path:
    parts = list(target_path.parts)
    marker = next(
        index
        for index in range(len(parts) - 1)
        if parts[index].startswith("sub") and parts[index + 1] in FINGER_NAMES
    )
    parts[marker] = f"sub{subject}"
    parts[marker + 1] = finger
    return Path(*parts)


def candidate_score(
    target_path: Path,
    target_subject: int,
    target_finger: str,
    prepared_root: Path,
) -> tuple[float, list[dict[str, object]]]:
    records = []
    for subject in (1, 2, 3):
        for finger in FINGER_NAMES:
            if subject == target_subject and finger == target_finger:
                continue
            path = source_path(target_path, subject, finger)
            if not path.is_file():
                continue
            try:
                prediction = scalar_prediction(path, finger)
                target = raw_target(prepared_root, subject, finger, prediction.size)
            except (OSError, ValueError, IndexError):
                continue
            observed = np.isfinite(prediction) & np.isfinite(target)
            if observed.sum() < 100 or np.std(prediction[observed]) <= 1.0e-8:
                continue
            pcc = pearson(prediction[observed], target[observed])
            if not np.isfinite(pcc):
                continue
            records.append({"subject": subject, "finger": finger, "pcc": pcc})
    if not records:
        return float("-inf"), records
    clipped = np.clip([record["pcc"] for record in records], -0.999, 0.999)
    return float(np.mean(np.arctanh(clipped))), records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--finger", choices=FINGER_NAMES, required=True)
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--prepared-root", type=Path, default=Path("outputs/preprocessed_v2")
    )
    parser.add_argument("--minimum-source-pairs", type=int, default=8)
    parser.add_argument("--ensemble-size", type=int, default=8)
    parser.add_argument("--fixed-reference-pcc", type=float, required=True)
    parser.add_argument("--gain-threshold", type=float, default=0.095)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    pattern_root = args.outputs_root
    target_fragment = (f"sub{args.subject}", args.finger)
    candidates = []
    for path in pattern_root.rglob("*.npy"):
        if path.name not in ALLOWED_FILENAMES:
            continue
        parts = path.parts
        if any(part.startswith("outer") and part[5:].isdigit() for part in parts):
            continue
        if not any(
            parts[index : index + 2] == target_fragment
            for index in range(len(parts) - 1)
        ):
            continue
        score, source_records = candidate_score(
            path, args.subject, args.finger, args.prepared_root
        )
        if len(source_records) < args.minimum_source_pairs:
            continue
        prediction = scalar_prediction(path, args.finger)
        observed = np.isfinite(prediction)
        if observed.sum() < 100 or np.std(prediction[observed]) <= 1.0e-8:
            continue
        normalized = np.full_like(prediction, np.nan, dtype=np.float32)
        normalized[observed] = (
            prediction[observed] - prediction[observed].mean()
        ) / prediction[observed].std()
        digest = hashlib.sha256(normalized[observed].tobytes()).hexdigest()
        candidates.append(
            {
                "path": path,
                "score": score,
                "source_records": source_records,
                "normalized": normalized,
                "digest": digest,
            }
        )

    candidates.sort(key=lambda item: item["score"], reverse=True)
    unique = []
    seen = set()
    for candidate in candidates:
        if candidate["digest"] in seen:
            continue
        seen.add(candidate["digest"])
        unique.append(candidate)
    selected = unique[: args.ensemble_size]
    if len(selected) < args.ensemble_size:
        raise RuntimeError(
            f"only {len(selected)} unique candidates meet source-pair requirement"
        )
    masks = [np.isfinite(candidate["normalized"]) for candidate in selected]
    observed = np.logical_and.reduce(masks)
    ensemble = np.full_like(selected[0]["normalized"], np.nan, dtype=np.float32)
    ensemble[observed] = np.mean(
        np.column_stack(
            [candidate["normalized"][observed] for candidate in selected]
        ),
        axis=1,
    )
    target = raw_target(
        args.prepared_root, args.subject, args.finger, ensemble.size
    )
    pcc = pearson(ensemble[observed], target[observed])
    gain = pcc - args.fixed_reference_pcc
    report = {
        "protocol": (
            "candidate families ranked by mean Fisher-z PCC on other subject-finger "
            "pairs only; fixed top-K prediction-z-scored equal ensemble applied once "
            "to held target pair; released test untouched"
        ),
        "released_test_touched": False,
        "target_pair_used_for_selection": False,
        "subject": args.subject,
        "finger": args.finger,
        "minimum_source_pairs": args.minimum_source_pairs,
        "ensemble_size": args.ensemble_size,
        "discovered_candidates": len(candidates),
        "unique_candidates": len(unique),
        "selected": [
            {
                "path": str(candidate["path"]),
                "source_mean_fisher_z": candidate["score"],
                "source_pair_count": len(candidate["source_records"]),
            }
            for candidate in selected
        ],
        "selected_oof_raw_pcc": pcc,
        "fixed_pre_ablation_reference_pcc": args.fixed_reference_pcc,
        "gain_over_fixed_reference": gain,
        "required_gain_over_fixed_reference": args.gain_threshold,
        "passes_working_threshold": gain >= args.gain_threshold,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "ensemble_oof.npy", ensemble, allow_pickle=False)
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
