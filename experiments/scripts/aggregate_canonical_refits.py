#!/usr/bin/env python3
"""Aggregate final refits only when every member has one identical structure."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from single_wavelet_support import pearson


def normalized_structure(value: dict[str, Any]) -> str:
    # Randomization roles may differ between ensemble members by design; they
    # do not change the computation graph, objective, or optimization policy.
    structural = {
        key: item
        for key, item in value.items()
        if key not in {"seed", "split_seed", "sampler_seed"}
    }
    return json.dumps(structural, sort_keys=True, separators=(",", ":"))


def aggregate(member_paths: list[Path], output: Path, structure_id: str) -> dict[str, Any]:
    if not member_paths:
        raise ValueError("at least one final refit member is required")
    reports = [json.loads((path / "summary.json").read_text()) for path in member_paths]
    if any(report.get("released_test_used_for_selection") for report in reports):
        raise ValueError("a member used released-test labels for selection")
    subjects = {int(report["subject"]) for report in reports}
    fingers = {str(report["finger"]) for report in reports}
    if len(subjects) != 1 or len(fingers) != 1:
        raise ValueError("all members must decode one identical subject/finger pair")
    structures = {
        normalized_structure(report["structure_settings"]) for report in reports
    }
    if len(structures) != 1:
        raise ValueError("heterogeneous model structures cannot be aggregated")

    targets = [np.load(path / "test_raw_target.npy") for path in member_paths]
    predictions = [np.load(path / "test_prediction_raw_scale.npy") for path in member_paths]
    reference_shape = targets[0].shape
    if any(target.shape != reference_shape for target in targets + predictions):
        raise ValueError("member prediction and target shapes differ")
    if any(not np.allclose(targets[0], target, equal_nan=True) for target in targets[1:]):
        raise ValueError("member released-test targets differ")
    prediction = np.mean(np.stack(predictions), axis=0).astype(np.float32)
    target = np.asarray(targets[0], dtype=np.float32)
    member_pcc = [pearson(item, target) for item in predictions]
    structure_text = next(iter(structures))
    report = {
        "protocol": "equal-weight final ensemble of one identical model structure",
        "subject": next(iter(subjects)),
        "finger": next(iter(fingers)),
        "structure_id": structure_id,
        "structure_sha256": hashlib.sha256(structure_text.encode()).hexdigest(),
        "member_count": len(member_paths),
        "member_paths": [str(path) for path in member_paths],
        "member_seeds": [int(item["seed"]) for item in reports],
        "member_structure_count": 1,
        "heterogeneous_model_soup": False,
        "released_test_used_for_selection": False,
        "released_test_member_pcc_descriptive_only": member_pcc,
        "released_test_ensemble_pcc_descriptive_only": pearson(prediction, target),
        "released_test_ensemble_rmse_descriptive_only": float(
            np.sqrt(np.mean((prediction - target) ** 2))
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "test_prediction_raw_scale.npy", prediction, allow_pickle=False)
    np.save(output / "test_raw_target.npy", target, allow_pickle=False)
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--member", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--structure-id", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            aggregate(args.member, args.output, args.structure_id), indent=2
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
