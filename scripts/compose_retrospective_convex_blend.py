#!/usr/bin/env python3
"""Fit one explicitly test-informed convex blend as a diagnostic upper bound."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ecog_decoding.training import FINGER_NAMES
from summarize_event_lars_lstm_cv import pearson


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, required=True, choices=(1, 2, 3))
    parser.add_argument("--finger", required=True, choices=tuple(FINGER_NAMES))
    parser.add_argument("--left-root", type=Path, required=True)
    parser.add_argument("--right-root", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, default=Path("outputs/preprocessed_v2"))
    parser.add_argument("--grid-size", type=int, default=1001)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.grid_size < 2:
        parser.error("--grid-size must be at least 2")

    finger = list(FINGER_NAMES).index(args.finger)
    left = np.load(args.left_root / f"sub{args.subject}" / "test_prediction.npy")
    right = np.load(args.right_root / f"sub{args.subject}" / "test_prediction.npy")
    if left.shape != right.shape or left.ndim != 2 or left.shape[1] != len(FINGER_NAMES):
        raise ValueError(f"candidate shapes must match five fingers: {left.shape}, {right.shape}")
    target = np.load(
        args.prepared_root / f"sub{args.subject}" / "test_glove_25hz_raw.npy"
    )[24 : 24 + left.shape[0], finger]

    weights = np.linspace(0.0, 1.0, args.grid_size)
    scores = np.asarray(
        [pearson((1.0 - weight) * left[:, finger] + weight * right[:, finger], target)
         for weight in weights]
    )
    selected = int(np.argmax(scores))
    right_weight = float(weights[selected])
    output = np.asarray(left, dtype=np.float32).copy()
    output[:, finger] = (
        (1.0 - right_weight) * left[:, finger] + right_weight * right[:, finger]
    )

    destination = args.output_root / f"sub{args.subject}"
    destination.mkdir(parents=True, exist_ok=True)
    np.save(destination / "test_prediction.npy", output)
    report = {
        "protocol": "released-test-informed convex blend for retrospective diagnosis only",
        "released_test_used_for_weight_selection": True,
        "suitable_as_confirmatory_benchmark": False,
        "subject": args.subject,
        "finger": args.finger,
        "left_root": str(args.left_root),
        "right_root": str(args.right_root),
        "grid_size": args.grid_size,
        "right_weight": right_weight,
        "left_pcc": pearson(left[:, finger], target),
        "right_pcc": pearson(right[:, finger], target),
        "blend_pcc": float(scores[selected]),
    }
    (args.output_root / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
