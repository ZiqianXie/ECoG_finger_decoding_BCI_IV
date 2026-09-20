#!/usr/bin/env python3
"""Rank existing stitched OOF reports for one subject-finger pair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("outputs"))
    parser.add_argument("--subject", type=int, required=True)
    parser.add_argument("--finger", required=True)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    records = []
    for path in args.root.rglob("summary.json"):
        try:
            report = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(report, dict):
            continue
        settings = report.get("settings", {})
        subject = settings.get("subject", report.get("subject"))
        finger = settings.get("finger", report.get("finger"))
        if subject != args.subject or finger != args.finger:
            continue
        pcc = report.get("stitched_selected_raw_pcc")
        if pcc is None:
            continue
        outer_folds = report.get("outer_folds", [])
        if sorted(int(item["outer_fold"]) for item in outer_folds) != [0, 1, 2]:
            continue
        records.append(
            {
                "pcc": float(pcc),
                "initialized_pcc": report.get("stitched_initialized_raw_pcc"),
                "gain_over_fixed_reference": report.get("gain_over_fixed_reference"),
                "path": str(path),
            }
        )
    records.sort(key=lambda item: item["pcc"], reverse=True)
    print(json.dumps(records[: args.limit], indent=2))


if __name__ == "__main__":
    main()
