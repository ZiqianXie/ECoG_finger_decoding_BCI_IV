#!/usr/bin/env python3
"""Summarize a frozen per-finger full-refit ensemble and render its events."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ecog_decoding.training import FINGER_NAMES
from evaluate_cv_ensemble_final_validation import movement_groups
from refit_frozen_event_model import plot_events
from render_extension_report import normalize_for_display
from summarize_event_lars_lstm_cv import morphology_metrics, pearson


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, required=True, choices=(1, 2, 3))
    parser.add_argument("--finger", required=True, choices=tuple(FINGER_NAMES))
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, default=Path("outputs/preprocessed_v2"))
    parser.add_argument("--morphology-target", default="local_w2_q10")
    parser.add_argument("--oof-summary", type=Path, default=None)
    parser.add_argument(
        "--base-candidate-root",
        type=Path,
        default=None,
        help="optional five-finger candidate whose selected column is replaced",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    finger_index = list(FINGER_NAMES).index(args.finger)
    member_predictions: list[np.ndarray] = []
    member_reports: list[dict[str, object]] = []
    for seed in args.seeds:
        root = args.input_root / f"sub{args.subject}" / args.finger / f"seed{seed}"
        prediction = np.load(root / "released_test_prediction.npy")
        summary = json.loads((root / "summary.json").read_text())
        member_predictions.append(np.asarray(prediction, dtype=np.float64))
        member_reports.append(
            {
                "seed": seed,
                "selected_epoch": int(summary["selected_epoch"]),
                "raw_pcc": float(summary["released_test_metrics"]["raw_pcc"]),
                "prediction_sd": float(np.std(prediction)),
            }
        )

    members = np.stack(member_predictions)
    ensemble = np.mean(members, axis=0)
    prepared = args.prepared_root / f"sub{args.subject}"
    raw = np.load(prepared / "test_glove_25hz_raw.npy")[24:, finger_index]
    cleaned = np.load(
        prepared / f"test_glove_{args.morphology_target}.npy"
    )[24:, finger_index]
    development = np.load(
        prepared / f"train_glove_{args.morphology_target}.npy"
    )[24:, finger_index]
    if ensemble.shape != raw.shape:
        raise ValueError(f"prediction shape {ensemble.shape} does not match target {raw.shape}")
    display, normalization = normalize_for_display(ensemble, development)

    pairwise = [
        pearson(members[left], members[right])
        for left in range(members.shape[0])
        for right in range(left + 1, members.shape[0])
    ]
    report: dict[str, object] = {
        "protocol": "OOF-frozen seed ensemble; released test used once for terminal reporting",
        "subject": args.subject,
        "finger": args.finger,
        "seeds": args.seeds,
        "released_test_used_for_seed_selection": False,
        "released_test_touched": True,
        "members": member_reports,
        "ensemble_raw_pcc": pearson(ensemble, raw),
        "ensemble_gain_over_best_member": pearson(ensemble, raw)
        - max(float(item["raw_pcc"]) for item in member_reports),
        "mean_pairwise_prediction_pcc": float(np.mean(pairwise)) if pairwise else None,
        "display_normalization": normalization,
        "display_morphology": morphology_metrics(
            display, cleaned, movement_groups(cleaned, 0.08)
        ),
    }
    if args.oof_summary is not None:
        report["training_only_oof_summary"] = str(args.oof_summary)
        report["training_only_oof"] = json.loads(args.oof_summary.read_text())

    output = args.output_root / f"sub{args.subject}"
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "released_test_prediction.npy", ensemble.astype(np.float32))
    np.save(output / "released_test_display_prediction.npy", display.astype(np.float32))
    if args.base_candidate_root is not None:
        base = np.load(
            args.base_candidate_root / f"sub{args.subject}" / "test_prediction.npy"
        )
        if base.shape != (ensemble.size, len(FINGER_NAMES)):
            raise ValueError(f"base candidate has unexpected shape {base.shape}")
        candidate = np.asarray(base, dtype=np.float32).copy()
        candidate[:, finger_index] = ensemble
        np.save(output / "test_prediction.npy", candidate)
        report["base_candidate_root"] = str(args.base_candidate_root)
    plot_events(
        output / f"{args.finger}_strongest_events.png",
        raw,
        cleaned,
        display,
        f"S{args.subject} {args.finger}: OOF-frozen full-refit ensemble",
    )
    (args.output_root / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
