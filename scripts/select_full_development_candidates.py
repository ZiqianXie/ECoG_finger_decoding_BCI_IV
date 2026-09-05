#!/usr/bin/env python3
"""Select per-finger candidates using full-development OOF predictions only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from ecog_decoding.training import FINGER_NAMES


def validate_report(report: dict[str, object], required_scope: str) -> None:
    if bool(report.get("released_test_touched", False)):
        raise ValueError("candidate summary touched the released test")
    scopes = report.get("selection_scopes", [])
    if scopes != [required_scope]:
        raise ValueError(
            f"candidate selection scope {scopes!r} does not equal {[required_scope]!r}"
        )


def select_candidates(
    reports: dict[str, dict[str, object]],
    candidate_options: dict[str, dict[str, object]],
    refit_defaults: dict[str, object],
    *,
    subjects: tuple[int, ...] = (1, 2, 3),
) -> tuple[dict[str, object], dict[str, object]]:
    ensemble_map: dict[str, object] = {
        "protocol": "selected by full-development event-fold OOF PCC",
        "released_test_touched_during_selection": False,
        "default": dict(refit_defaults),
        "subjects": {},
    }
    audit: dict[str, object] = {
        "selection_metric": "ensemble_oof_pcc",
        "released_test_touched": False,
        "subjects": {},
    }
    for subject in subjects:
        subject_map: dict[str, object] = {}
        subject_audit: dict[str, object] = {"per_finger": {}}
        for finger in FINGER_NAMES:
            scores = {
                name: float(
                    report["subjects"][str(subject)]["per_finger"][finger][
                        "ensemble_oof_pcc"
                    ]
                )
                for name, report in reports.items()
            }
            selected = max(sorted(scores), key=scores.__getitem__)
            selected_result = reports[selected]["subjects"][str(subject)][
                "per_finger"
            ][finger]
            options = dict(candidate_options[selected])
            options.pop("summary", None)
            options["seeds"] = [int(seed) for seed in selected_result["included_seeds"]]
            subject_map[finger] = options
            subject_audit["per_finger"][finger] = {
                "candidate_oof_pcc": scores,
                "selected": selected,
                "selected_oof_pcc": scores[selected],
                "selected_seeds": options["seeds"],
            }
        ensemble_map["subjects"][subject] = subject_map
        audit["subjects"][str(subject)] = subject_audit
    return ensemble_map, audit


def score_matrix(
    report: dict[str, object], subjects: tuple[int, ...]
) -> np.ndarray:
    return np.asarray(
        [
            [
                report["subjects"][str(subject)]["per_finger"][finger][
                    "ensemble_oof_pcc"
                ]
                for finger in FINGER_NAMES
            ]
            for subject in subjects
        ],
        dtype=np.float64,
    )


def annotate(axis: plt.Axes, values: np.ndarray) -> None:
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            axis.text(
                column,
                row,
                f"{values[row, column]:.3f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if values[row, column] > 0.55 else "black",
            )


def render_comparison(
    path: Path,
    reports: dict[str, dict[str, object]],
    audit: dict[str, object],
    subjects: tuple[int, ...],
) -> None:
    names = list(reports)
    matrices = {name: score_matrix(report, subjects) for name, report in reports.items()}
    selected = np.asarray(
        [
            [
                audit["subjects"][str(subject)]["per_finger"][finger][
                    "selected_oof_pcc"
                ]
                for finger in FINGER_NAMES
            ]
            for subject in subjects
        ],
        dtype=np.float64,
    )
    panels = [(name, matrices[name]) for name in names] + [("selected", selected)]
    figure, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4.2))
    axes = np.atleast_1d(axes)
    for axis, (name, values) in zip(axes, panels):
        image = axis.imshow(values, cmap="viridis", vmin=0.2, vmax=0.8)
        annotate(axis, values)
        axis.set_title(name)
        axis.set_xticks(range(5), FINGER_NAMES, rotation=35, ha="right")
        axis.set_yticks(range(len(subjects)), [f"S{subject}" for subject in subjects])
        figure.colorbar(image, ax=axis, shrink=0.76, label="full-development OOF PCC")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/full_development_candidate_selection.yaml"),
    )
    parser.add_argument(
        "--ensemble-map",
        type=Path,
        default=Path("configs/full_development_event_ensemble.yaml"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/full_development_selection"),
    )
    args = parser.parse_args()

    configuration = yaml.safe_load(args.config.read_text())
    required_scope = str(configuration["required_selection_scope"])
    candidates = configuration["candidates"]
    reports: dict[str, dict[str, object]] = {}
    for name, options in candidates.items():
        report = json.loads(Path(options["summary"]).read_text())
        validate_report(report, required_scope)
        reports[name] = report
    subjects = tuple(
        sorted(
            set.intersection(
                *(set(report["subjects"]) for report in reports.values())
            ),
            key=int,
        )
    )
    subject_ids = tuple(int(subject) for subject in subjects)
    ensemble_map, audit = select_candidates(
        reports,
        candidates,
        configuration["refit_defaults"],
        subjects=subject_ids,
    )
    ensemble_map["selection_config"] = str(args.config)
    ensemble_map["fold_root"] = str(configuration["fold_root"])
    ensemble_map["target_map"] = str(configuration["target_map"])

    args.ensemble_map.parent.mkdir(parents=True, exist_ok=True)
    args.ensemble_map.write_text(yaml.safe_dump(ensemble_map, sort_keys=False))
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "selection.json").write_text(
        json.dumps(audit, indent=2) + "\n"
    )
    render_comparison(
        args.output_root / "candidate_selection.png",
        reports,
        audit,
        subject_ids,
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
