#!/usr/bin/env python3
"""Cross-validate a ridge stack over already out-of-fold base predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ecog_decoding.training import FINGER_NAMES
from single_wavelet_support import OFFSET


ALPHAS = tuple(float(value) for value in np.logspace(-3, 5, 9))
OUTER_DIRECTORY = re.compile(r"outer\d+$")


def rows_from_intervals(intervals: list[list[int]]) -> np.ndarray:
    return np.concatenate(
        [np.arange(start, stop, dtype=np.int64) for start, stop in intervals]
    )


def pearson(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.corrcoef(np.asarray(first), np.asarray(second))[0, 1])


def is_prediction_path(path: Path, subject: int, finger: str) -> bool:
    lowered = path.name.lower()
    if "oof" not in lowered or "target" in lowered or "cleaned" in lowered:
        return False
    if any(OUTER_DIRECTORY.fullmatch(part) for part in path.parts):
        return False
    parts = path.parts
    return any(
        parts[index : index + 2] == (f"sub{subject}", finger)
        for index in range(len(parts) - 1)
    )


def load_scalar(path: Path, finger: str, rows: int) -> np.ndarray | None:
    try:
        prediction = np.asarray(np.load(path, mmap_mode="r"), dtype=np.float32)
    except (OSError, ValueError):
        return None
    if prediction.ndim == 2 and prediction.shape[1] == len(FINGER_NAMES):
        prediction = prediction[:, list(FINGER_NAMES).index(finger)]
    if prediction.ndim != 1 or prediction.size != rows:
        return None
    return prediction


def discover_predictions(
    outputs_root: Path,
    subject: int,
    finger: str,
    rows: int,
    required: np.ndarray,
) -> list[tuple[Path, np.ndarray]]:
    unique: list[tuple[Path, np.ndarray]] = []
    seen = set()
    for path in outputs_root.rglob("*.npy"):
        if not is_prediction_path(path, subject, finger):
            continue
        prediction = load_scalar(path, finger, rows)
        if prediction is None or not np.isfinite(prediction[required]).all():
            continue
        values = prediction[required]
        if np.std(values) <= 1.0e-8:
            continue
        normalized = (values - values.mean()) / values.std()
        digest = hashlib.sha256(normalized.astype(np.float32).tobytes()).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        unique.append((path, prediction))
    return unique


def select_columns(
    features: np.ndarray,
    target: np.ndarray,
    rows: np.ndarray,
    count: int,
) -> np.ndarray:
    values = features[rows]
    centered = values - values.mean(axis=0, keepdims=True)
    centered_target = target[rows] - target[rows].mean()
    denominator = np.sqrt(np.sum(centered * centered, axis=0)) * np.sqrt(
        np.sum(centered_target * centered_target)
    )
    correlations = np.divide(
        centered.T @ centered_target,
        denominator,
        out=np.zeros(features.shape[1], dtype=np.float32),
        where=denominator > 0,
    )
    count = min(count, features.shape[1])
    selected = np.argpartition(np.abs(correlations), -count)[-count:]
    return selected[np.argsort(np.abs(correlations[selected]))[::-1]]


def choose_alpha(
    features: np.ndarray,
    target: np.ndarray,
    training_fold_rows: list[np.ndarray],
) -> tuple[float, dict[str, float]]:
    scores = {alpha: [] for alpha in ALPHAS}
    for validation_index in range(len(training_fold_rows)):
        validation = training_fold_rows[validation_index]
        training = np.concatenate(
            [
                rows
                for index, rows in enumerate(training_fold_rows)
                if index != validation_index
            ]
        )
        for alpha in ALPHAS:
            model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
            model.fit(features[training], target[training])
            estimate = np.maximum(model.predict(features[validation]), 0.0)
            scores[alpha].append(pearson(estimate, target[validation]))
    means = {alpha: float(np.mean(values)) for alpha, values in scores.items()}
    selected = max(means, key=means.get)
    return selected, {str(alpha): value for alpha, value in means.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--finger", choices=FINGER_NAMES, required=True)
    parser.add_argument("--split-cache-root", type=Path, required=True)
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--prepared-root", type=Path, default=Path("outputs/preprocessed_v2")
    )
    parser.add_argument("--top-features", type=int, default=32)
    parser.add_argument("--fixed-reference-pcc", type=float, required=True)
    parser.add_argument("--gain-threshold", type=float, default=0.095)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.top_features <= 0:
        parser.error("top-features must be positive")

    finger_index = list(FINGER_NAMES).index(args.finger)
    rows = int(
        np.load(
            args.split_cache_root / "cache" / "outer0" / "outer" / "target.npy",
            mmap_mode="r",
        ).shape[0]
    )
    raw = np.asarray(
        np.load(
            args.prepared_root
            / f"sub{args.subject}"
            / "train_glove_25hz_raw.npy",
            mmap_mode="r",
        )[OFFSET : OFFSET + rows, finger_index],
        dtype=np.float32,
    )
    fold_rows = []
    for outer_fold in range(3):
        split = json.loads(
            (
                args.split_cache_root
                / "cache"
                / f"outer{outer_fold}"
                / "outer"
                / "split.json"
            ).read_text()
        )
        fold_rows.append(rows_from_intervals(split["validation_intervals"]))
    required = np.unique(np.concatenate(fold_rows))
    candidates = discover_predictions(
        args.outputs_root,
        args.subject,
        args.finger,
        rows,
        required,
    )
    if len(candidates) < 2:
        raise RuntimeError("fewer than two complete unique OOF predictions found")
    features = np.column_stack([prediction for _, prediction in candidates])
    stitched = np.full(rows, np.nan, dtype=np.float32)
    outer_records = []
    for outer_fold, validation in enumerate(fold_rows):
        training_folds = [
            fold for index, fold in enumerate(fold_rows) if index != outer_fold
        ]
        training = np.concatenate(training_folds)
        selected = select_columns(
            features, raw, training, args.top_features
        )
        selected_features = features[:, selected]
        alpha, curve = choose_alpha(selected_features, raw, training_folds)
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        model.fit(selected_features[training], raw[training])
        estimate = np.maximum(model.predict(selected_features[validation]), 0.0)
        stitched[validation] = estimate
        outer_records.append(
            {
                "outer_fold": outer_fold,
                "selected_alpha": alpha,
                "inner_meta_pcc_curve": curve,
                "selected_candidates": [str(candidates[index][0]) for index in selected],
                "outer_raw_pcc": pearson(estimate, raw[validation]),
            }
        )

    observed = np.isfinite(stitched) & np.isfinite(raw)
    pcc = pearson(stitched[observed], raw[observed])
    gain = pcc - args.fixed_reference_pcc
    report = {
        "protocol": (
            "three-fold cross-validated ridge meta-model over complete OOF base "
            "predictions; candidate correlation screen and ridge fitting use only "
            "the other two outer folds; released test untouched"
        ),
        "released_test_touched": False,
        "outer_validation_used_for_meta_fitting": False,
        "subject": args.subject,
        "finger": args.finger,
        "discovered_unique_oof_candidates": len(candidates),
        "top_features": args.top_features,
        "outer_records": outer_records,
        "selected_oof_raw_pcc": pcc,
        "fixed_pre_ablation_reference_pcc": args.fixed_reference_pcc,
        "gain_over_fixed_reference": gain,
        "required_gain_over_fixed_reference": args.gain_threshold,
        "passes_working_threshold": gain >= args.gain_threshold,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "selected_oof.npy", stitched, allow_pickle=False)
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
