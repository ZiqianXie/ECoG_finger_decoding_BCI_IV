#!/usr/bin/env python3
"""Select residual recurrent candidates using development OOF only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml


def resolve_route(
    route_map: dict[str, object], subject: int, finger: str
) -> dict[str, object]:
    route = dict(route_map["default"])
    subjects = route_map.get("subjects", {})
    subject_routes = subjects.get(subject, subjects.get(str(subject), {}))
    route.update(subject_routes.get(finger, {}))
    return route


def pooled_schedule(summary: dict[str, object]) -> tuple[str, dict[str, float]]:
    names = [str(name) for name in summary["schedule_candidates"]]
    records = [
        inner
        for outer in summary["outer_folds"]
        for inner in outer["inner_records"]
    ]
    means = {
        name: float(np.mean([row["metrics"][name]["raw_pcc"] for row in records]))
        for name in names
    }
    return max(means, key=means.get), means


def mean_outer_metrics(summary: dict[str, object]) -> dict[str, float]:
    keys = (
        "raw_pcc", "cleaned_pcc", "event_macro_nmse", "rest_false_positive_rms",
        "movement_rmse", "velocity_pcc", "amplitude_slope",
    )
    return {
        key: float(np.mean([
            outer["selected_outer_metrics"][key] for outer in summary["outer_folds"]
        ]))
        for key in keys
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path,
        default=Path("outputs/residual_recurrent_single_wavelet_oof_v1")
    )
    parser.add_argument(
        "--route-map", type=Path,
        default=Path("experiments/configs/final_lstm_lr_baseline_routes.yaml")
    )
    parser.add_argument("--minimum-oof-gain", type=float, default=0.002)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    route_map = yaml.safe_load(args.route_map.read_text())
    grouped: dict[tuple[int, str], list[dict[str, object]]] = {}
    for path in sorted(args.root.glob("residual_*/lr*/sub*/*/summary.json")):
        summary = json.loads(path.read_text())
        schedule, schedule_means = pooled_schedule(summary)
        record = {
            "family": summary["settings"]["recurrent_cell"],
            "learning_rate": float(summary["settings"]["head_learning_rate"]),
            "initialized_oof_pcc": float(summary["stitched_initialized_raw_pcc"]),
            "selected_oof_pcc": float(summary["stitched_selected_raw_pcc"]),
            "selected_schedule_by_outer_fold": [
                outer["selected_schedule"] for outer in summary["outer_folds"]
            ],
            "full_development_refit_schedule": schedule,
            "full_development_inner_mean_pcc_by_schedule": schedule_means,
            "mean_outer_metrics": mean_outer_metrics(summary),
            "summary": str(path),
        }
        grouped.setdefault((int(summary["subject"]), str(summary["finger"])), []).append(record)

    report_pairs = {}
    for (subject, finger), candidates in sorted(grouped.items()):
        route = resolve_route(route_map, subject, finger)
        incumbent_path = (
            Path(str(route["selection_root"])) / f"sub{subject}" / finger / "summary.json"
        )
        incumbent_summary = json.loads(incumbent_path.read_text())
        incumbent_oof = float(incumbent_summary["stitched_selected_raw_pcc"])
        initialized_oof = float(np.mean([row["initialized_oof_pcc"] for row in candidates]))
        deployable = [
            row for row in candidates
            if row["full_development_refit_schedule"] != "frozen_0"
        ]
        best = max(deployable, key=lambda row: row["selected_oof_pcc"], default=None)
        reference = max(incumbent_oof, initialized_oof)
        gain = float(best["selected_oof_pcc"] - reference) if best else float("-inf")
        promoted = best is not None and gain >= args.minimum_oof_gain
        report_pairs[f"S{subject}_{finger}"] = {
            "incumbent_oof_pcc": incumbent_oof,
            "initialized_residual_oof_pcc": initialized_oof,
            "reference_oof_pcc": reference,
            "candidates": sorted(
                candidates, key=lambda row: (row["family"], -row["learning_rate"])
            ),
            "selected": (
                {**best, "oof_gain_over_reference": gain, "promoted": True}
                if promoted
                else {
                    "choice": "no_residual_promotion",
                    "oof_gain_over_reference": gain if best is not None else None,
                    "promoted": False,
                    "reason": "no deployable residual candidate cleared the OOF gain threshold",
                }
            ),
        }

    report = {
        "protocol": (
            "development-only nested event-fold selection; released test labels "
            "are neither loaded nor used"
        ),
        "released_test_used": False,
        "minimum_oof_gain": args.minimum_oof_gain,
        "pairs": report_pairs,
    }
    output = args.output or args.root / "selection_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
