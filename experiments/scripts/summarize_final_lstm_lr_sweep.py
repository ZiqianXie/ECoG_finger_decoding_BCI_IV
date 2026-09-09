#!/usr/bin/env python3
"""Select LSTM learning rate from development-only OOF predictions."""

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


def full_development_schedule(
    summary: dict[str, object],
) -> tuple[str, dict[str, float]]:
    names = [str(name) for name in summary["schedule_candidates"]]
    inner_records = [
        inner
        for outer in summary["outer_folds"]
        for inner in outer["inner_records"]
    ]
    means = {
        name: float(
            np.mean(
                [float(record["metrics"][name]["raw_pcc"]) for record in inner_records]
            )
        )
        for name in names
    }
    return max(means, key=means.get), means


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/final_single_wavelet_lstm_lr_sweep_v2"),
    )
    parser.add_argument(
        "--route-map",
        type=Path,
        default=Path("experiments/configs/final_lstm_lr_baseline_routes.yaml"),
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    route_map = yaml.safe_load(args.route_map.read_text())
    grouped: dict[tuple[int, str], list[dict[str, object]]] = {}
    for path in sorted(args.root.glob("lr*/sub*/*/summary.json")):
        summary = json.loads(path.read_text())
        subject = int(summary["subject"])
        finger = str(summary["finger"])
        learning_rate = float(summary["settings"]["head_learning_rate"])
        final_schedule, schedule_means = full_development_schedule(summary)
        grouped.setdefault((subject, finger), []).append(
            {
                "learning_rate": learning_rate,
                "stitched_initialized_oof_pcc": float(
                    summary["stitched_initialized_raw_pcc"]
                ),
                "stitched_selected_oof_pcc": float(
                    summary["stitched_selected_raw_pcc"]
                ),
                "selected_schedule_by_outer_fold": [
                    outer["selected_schedule"] for outer in summary["outer_folds"]
                ],
                "full_development_refit_schedule": final_schedule,
                "full_development_inner_mean_pcc_by_schedule": schedule_means,
                "summary": str(path),
            }
        )

    pairs = {}
    for (subject, finger), candidates in sorted(grouped.items()):
        candidates.sort(key=lambda item: -float(item["learning_rate"]))
        initial_values = np.asarray(
            [item["stitched_initialized_oof_pcc"] for item in candidates],
            dtype=np.float64,
        )
        if float(np.ptp(initial_values)) > 1e-5:
            raise RuntimeError(
                f"initial OOF predictions differ for S{subject} {finger}"
            )
        baseline = float(initial_values.mean())
        route = resolve_route(route_map, subject, finger)
        previous_summary_path = (
            Path(str(route["selection_root"]))
            / f"sub{subject}"
            / finger
            / "summary.json"
        )
        previous = json.loads(previous_summary_path.read_text())
        previous_oof = float(previous["stitched_selected_raw_pcc"])
        previous_refit_summary_path = (
            Path(str(route["ensemble_root"]))
            / f"sub{subject}"
            / finger
            / "seed0"
            / "summary.json"
        )
        previous_refit = json.loads(previous_refit_summary_path.read_text())
        previous_schedule = str(previous_refit["selected_schedule"])
        tuned = max(candidates, key=lambda item: item["stitched_selected_oof_pcc"])
        deployable_tuned_oof = (
            float(tuned["stitched_selected_oof_pcc"])
            if tuned["full_development_refit_schedule"] != "frozen_0"
            else float("-inf")
        )
        best_name, _ = max(
            (
                ("previous_route", previous_oof),
                ("frozen_0", baseline),
                ("fine_tune", deployable_tuned_oof),
            ),
            key=lambda item: item[1],
        )
        if best_name == "previous_route":
            selected = {
                "choice": "previous_route",
                "learning_rate": float(previous["settings"]["head_learning_rate"]),
                "full_development_refit_schedule": previous_schedule,
                "development_oof_pcc": previous_oof,
                "summary": str(previous_summary_path),
                "reason": "incumbent route retained because no new candidate improved OOF PCC",
            }
        elif best_name == "fine_tune":
            selected = {
                "choice": "fine_tune",
                "learning_rate": tuned["learning_rate"],
                "full_development_refit_schedule": tuned[
                    "full_development_refit_schedule"
                ],
                "development_oof_pcc": tuned["stitched_selected_oof_pcc"],
                "summary": tuned["summary"],
            }
        else:
            selected = {
                "choice": "frozen_0",
                "learning_rate": None,
                "full_development_refit_schedule": "frozen_0",
                "development_oof_pcc": baseline,
                "summary": None,
                "reason": "untouched initialization had the highest OOF PCC",
            }

        pairs[f"S{subject}_{finger}"] = {
            "baseline_oof_pcc": baseline,
            "previous_selected_oof_pcc": previous_oof,
            "candidates": candidates,
            "selected": selected,
        }

    report = {
        "protocol": (
            "learning rate and update count selected from nested event-fold "
            "development OOF predictions; released test labels are untouched"
        ),
        "released_test_used": False,
        "selection_metric": "stitched development OOF raw PCC",
        "pairs": pairs,
    }
    output = args.output or args.root / "selection_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
