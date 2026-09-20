#!/usr/bin/env python3
"""Compose a predeclared fixed-weight blend of compatible OOF predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ecog_decoding.training import FINGER_NAMES
from single_wavelet_support import OFFSET, pearson


def candidate_argument(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("candidate must be NAME=OUTPUT_ROOT")
    return name, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--finger", choices=tuple(FINGER_NAMES), required=True)
    parser.add_argument("--candidate", action="append", type=candidate_argument, required=True)
    parser.add_argument("--weight", action="append", type=float, required=True)
    parser.add_argument("--fixed-reference-pcc", type=float, required=True)
    parser.add_argument("--gain-threshold", type=float, default=0.1)
    parser.add_argument(
        "--prepared-root", type=Path, default=Path("outputs/preprocessed_v2")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.candidate) != len(args.weight):
        parser.error("provide exactly one --weight per --candidate")
    weights = np.asarray(args.weight, dtype=np.float64)
    if not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        parser.error("weights must be finite, nonnegative, and have positive sum")
    weights /= weights.sum()

    predictions = []
    targets = []
    for _, root in args.candidate:
        predictions.append(np.load(root / "selected_oof.npy"))
        targets.append(np.load(root / "target_oof.npy"))
    shape = predictions[0].shape
    if any(value.shape != shape for value in predictions + targets):
        raise ValueError("candidate prediction and target shapes differ")
    for target in targets[1:]:
        if not np.allclose(targets[0], target, atol=1e-6, equal_nan=True):
            raise ValueError("candidate cleaned OOF targets differ")

    finite = np.logical_and.reduce([np.isfinite(value) for value in predictions])
    if not finite.any():
        raise ValueError("candidates have no shared finite OOF rows")
    blended = np.full(shape, np.nan, dtype=np.float32)
    blended[finite] = sum(
        weight * prediction[finite]
        for weight, prediction in zip(weights, predictions, strict=True)
    )
    finger_index = list(FINGER_NAMES).index(args.finger)
    raw = np.load(
        args.prepared_root / f"sub{args.subject}" / "train_glove_25hz_raw.npy",
        mmap_mode="r",
    )[OFFSET : OFFSET + shape[0], finger_index]
    observed = finite & np.isfinite(raw)
    pcc = pearson(blended[observed], np.asarray(raw[observed]))
    gain = pcc - args.fixed_reference_pcc

    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "selected_oof.npy", blended, allow_pickle=False)
    np.save(args.output / "target_oof.npy", targets[0], allow_pickle=False)
    report = {
        "protocol": "fixed-weight blend declared without fitting weights to outer targets",
        "released_test_touched": False,
        "outer_validation_used_for_weight_selection": False,
        "subject": args.subject,
        "finger": args.finger,
        "candidates": {
            name: {"root": str(root), "weight": float(weight)}
            for (name, root), weight in zip(args.candidate, weights, strict=True)
        },
        "selected_oof_raw_pcc": pcc,
        "fixed_pre_ablation_reference_pcc": args.fixed_reference_pcc,
        "gain_over_fixed_reference": gain,
        "required_gain_over_fixed_reference": args.gain_threshold,
        "passes_fixed_reference_gain_threshold": gain >= args.gain_threshold,
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
