#!/usr/bin/env python3
"""Collect exact-window learning-rate runs without using test scores to select."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_run(value: str) -> tuple[str, Path]:
    label, separator, path = value.partition("=")
    if not separator or not label or not path:
        raise argparse.ArgumentTypeError("runs must have the form LABEL=SUMMARY_JSON")
    return label, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, required=True)
    parser.add_argument("--finger", required=True)
    parser.add_argument("--baseline-validation", type=float, required=True)
    parser.add_argument("--baseline-test", type=float, required=True)
    parser.add_argument("--run", action="append", type=parse_run, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    runs: dict[str, object] = {}
    for label, path in args.run:
        source = json.loads(path.read_text())
        result = source["per_finger"][args.finger]
        runs[label] = {
            "summary": str(path),
            "optimization": source.get("optimization"),
            "initialized_validation_raw_r": result.get("initialized_validation_raw_r"),
            "best_epoch": result["best_epoch"],
            "validation_raw_r": result["validation_raw_r"],
            "test_raw_r_descriptive_only": result["test_raw_r"],
        }

    selected = max(runs, key=lambda label: runs[label]["validation_raw_r"])
    report = {
        "subject": args.subject,
        "finger": args.finger,
        "selection_rule": "maximum chronological validation PCC; released test is descriptive only",
        "baseline": {
            "validation_raw_r": args.baseline_validation,
            "test_raw_r_descriptive_only": args.baseline_test,
        },
        "runs": runs,
        "selected_by_validation": selected,
        "selected_validation_raw_r": runs[selected]["validation_raw_r"],
        "released_test_used_for_selection": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
