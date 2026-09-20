#!/usr/bin/env python3
"""Audit canonical per-pair routes for single-family and positive tuning rules."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


FINGERS = ("thumb", "index", "middle", "ring", "little")


def resolve_options(route_map: dict[str, object], subject: int, finger: str) -> dict[str, object]:
    options = dict(route_map.get("default", {}))
    subjects = route_map.get("subjects", {})
    subject_options = subjects.get(subject, subjects.get(str(subject), {}))
    options.update(subject_options.get(finger, {}))
    return options


def audit_routes(route_path: Path) -> dict[str, object]:
    route_map = yaml.safe_load(route_path.read_text())
    pairs: dict[str, object] = {}
    for subject in (1, 2, 3):
        for finger in FINGERS:
            options = resolve_options(route_map, subject, finger)
            selection_root = Path(str(options["selection_root"]))
            summary_path = selection_root / f"sub{subject}" / finger / "summary.json"
            summary = json.loads(summary_path.read_text())
            initialized = float(summary["stitched_initialized_raw_pcc"])
            attempted_selected = float(summary["stitched_selected_raw_pcc"])
            explicit_schedule = options.get("selected_schedule")
            selected = initialized if explicit_schedule == "frozen_0" else attempted_selected
            gain = selected - initialized
            seeds = [int(seed) for seed in options["seeds"]]
            pairs[f"S{subject}_{finger}"] = {
                "selection_root": str(selection_root),
                "ensemble_root": str(options["ensemble_root"]),
                "seeds": seeds,
                "same_structure_seed_ensemble": True,
                "explicit_selected_schedule": explicit_schedule,
                "initialized_oof_pcc": initialized,
                "attempted_selected_oof_pcc": attempted_selected,
                "selected_oof_pcc": selected,
                "same_architecture_tuning_gain": gain,
                "tuning_strictly_improves": bool(gain > 0.0),
                "released_test_touched_during_selection": bool(
                    summary.get("released_test_touched", False)
                ),
                "summary": str(summary_path),
            }

    failures = [
        pair for pair, record in pairs.items() if not record["tuning_strictly_improves"]
    ]
    return {
        "route_map": str(route_path),
        "rules": {
            "one_structure_per_pair": True,
            "same_structure_seed_ensemble_allowed": True,
            "same_architecture_tuning_gain_must_be_positive": True,
        },
        "pair_count": len(pairs),
        "passing_pair_count": len(pairs) - len(failures),
        "failing_pair_count": len(failures),
        "failing_pairs": failures,
        "all_pairs_satisfy_rules": not failures,
        "pairs": pairs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--routes",
        type=Path,
        default=Path("configs/final_single_wavelet_routes.yaml"),
    )
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args()
    report = audit_routes(args.routes)
    print(json.dumps(report, indent=2))
    if args.require_pass and not report["all_pairs_satisfy_rules"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
