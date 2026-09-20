#!/usr/bin/env python3
"""Audit the outer-target ceiling of compatible saved OOF predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ecog_decoding.training import FINGER_NAMES
from single_wavelet_support import OFFSET, pearson


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("outputs"))
    parser.add_argument("--subject", type=int, required=True)
    parser.add_argument("--finger", choices=tuple(FINGER_NAMES), required=True)
    parser.add_argument("--fixed-reference-pcc", type=float, required=True)
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--prepared-root", type=Path, default=Path("outputs/preprocessed_v2"))
    args = parser.parse_args()

    records = []
    for summary in args.root.rglob("summary.json"):
        prediction_path = summary.with_name("selected_oof.npy")
        if not prediction_path.is_file():
            continue
        try:
            report = json.loads(summary.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        settings = report.get("settings", {})
        if settings.get("subject", report.get("subject")) != args.subject:
            continue
        if settings.get("finger", report.get("finger")) != args.finger:
            continue
        outer_folds = report.get("outer_folds")
        if outer_folds is not None:
            observed_folds = sorted(int(item["outer_fold"]) for item in outer_folds)
            if observed_folds != [0, 1, 2]:
                continue
        elif "selected_oof_raw_pcc" not in report:
            continue
        prediction = np.load(prediction_path)
        records.append((str(summary.parent), prediction))
    if not records:
        raise ValueError("no compatible OOF predictions found")

    finger_index = list(FINGER_NAMES).index(args.finger)
    rows = records[0][1].size
    raw = np.load(
        args.prepared_root / f"sub{args.subject}" / "train_glove_25hz_raw.npy",
        mmap_mode="r",
    )[OFFSET : OFFSET + rows, finger_index]
    scored = []
    for path, prediction in records:
        if prediction.shape != (rows,):
            continue
        observed = np.isfinite(prediction) & np.isfinite(raw)
        if observed.sum() < 10:
            continue
        scored.append((pearson(prediction[observed], raw[observed]), path, prediction))
    scored.sort(key=lambda item: item[0], reverse=True)

    unique = []
    for score, path, prediction in scored:
        if any(
            np.allclose(prediction, existing[2], equal_nan=True, atol=1.0e-7)
            for existing in unique
        ):
            continue
        unique.append((score, path, prediction))
        if len(unique) >= args.limit:
            break
    best_mask = np.isfinite(unique[0][2]) & np.isfinite(raw)
    unique = [
        record
        for record in unique
        if np.array_equal(np.isfinite(record[2]) & np.isfinite(raw), best_mask)
    ]
    observed = best_mask
    target = np.asarray(raw[observed], dtype=np.float64)
    columns = []
    for _, _, prediction in unique:
        column = np.asarray(prediction[observed], dtype=np.float64)
        column = (column - column.mean()) / max(column.std(), 1.0e-8)
        columns.append(column)

    selected: list[int] = []
    greedy = []
    running = np.zeros_like(target)
    remaining = set(range(len(columns)))
    for step in range(min(12, len(columns))):
        candidates = [
            (pearson((running + columns[index]) / (step + 1), target), index)
            for index in remaining
        ]
        score, index = max(candidates)
        selected.append(index)
        remaining.remove(index)
        running += columns[index]
        greedy.append(
            {
                "members": step + 1,
                "pcc": score,
                "gain_over_fixed_reference": score - args.fixed_reference_pcc,
                "added": unique[index][1],
            }
        )

    matrix = np.column_stack(columns)
    centered_target = target - target.mean()
    penalty = 0.1 * matrix.shape[0]
    coefficient = np.linalg.solve(
        matrix.T @ matrix + penalty * np.eye(matrix.shape[1]),
        matrix.T @ centered_target,
    )
    ridge_prediction = matrix @ coefficient
    ridge_pcc = pearson(ridge_prediction, target)
    report = {
        "subject": args.subject,
        "finger": args.finger,
        "candidate_count": len(unique),
        "single_best": [
            {"pcc": score, "path": path} for score, path, _ in unique[:12]
        ],
        "outer_diagnostic_greedy_equal_ensemble": greedy,
        "outer_diagnostic_ridge_ceiling": {
            "pcc": ridge_pcc,
            "gain_over_fixed_reference": ridge_pcc - args.fixed_reference_pcc,
            "l2_per_row": 0.1,
        },
        "selection_warning": "all combinations use outer targets and are diagnostic only",
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
