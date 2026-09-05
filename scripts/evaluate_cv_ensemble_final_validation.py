#!/usr/bin/env python3
"""Evaluate a frozen cross-fold ensemble on the official final validation.

Run this only after architecture, target policy, optimization schedule, and
seed inclusion rules have been frozen using the event-grouped fit partition.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from ecog_decoding.training import FINGER_NAMES
from summarize_event_lars_lstm_cv import (
    morphology_metrics,
    pearson,
    resolve_ensemble_spec,
)
from train_exact_window_end_to_end import ExactWindowFingerDecoder


def movement_groups(target: np.ndarray, threshold: float, padding: int = 25) -> list[dict[str, int]]:
    indices = np.flatnonzero(target >= threshold)
    if not indices.size:
        return [{"start": 0, "stop": int(target.size)}]
    breaks = np.flatnonzero(np.diff(indices) > 1) + 1
    runs = np.split(indices, breaks)
    padded = [
        [max(0, int(run[0]) - padding), min(target.size, int(run[-1]) + 1 + padding)]
        for run in runs
    ]
    merged: list[list[int]] = []
    for start, stop in padded:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], stop)
        else:
            merged.append([start, stop])
    return [{"start": start, "stop": stop} for start, stop in merged]


def frozen_oof_seed_inclusion(
    input_root: Path,
    fold_root: Path,
    subject: int,
    finger_name: str,
    seeds: list[int] | tuple[int, ...],
) -> tuple[list[int], dict[str, object]]:
    """Freeze seed membership using only training-partition OOF predictions."""
    definition = json.loads(
        (fold_root / f"sub{subject}" / finger_name / "folds.json").read_text()
    )
    rows = int(definition["training_rows"])
    cleaned = np.full(rows, np.nan, dtype=np.float32)
    predictions = {
        seed: np.full(rows, np.nan, dtype=np.float32) for seed in seeds
    }
    for fold in range(3):
        reference = input_root / f"sub{subject}" / finger_name / f"fold{fold}" / f"seed{seeds[0]}"
        indices = np.load(reference / "validation_indices.npy")
        fold_cleaned = np.load(reference / "validation_cleaned_target.npy")
        if np.any(np.isfinite(cleaned[indices])):
            raise RuntimeError(f"overlapping OOF rows for S{subject} {finger_name}")
        cleaned[indices] = fold_cleaned
        for seed in seeds:
            root = input_root / f"sub{subject}" / finger_name / f"fold{fold}" / f"seed{seed}"
            predictions[seed][indices] = np.load(root / "validation_prediction.npy")

    if not np.isfinite(cleaned).all() or any(
        not np.isfinite(values).all() for values in predictions.values()
    ):
        raise RuntimeError(f"incomplete OOF coverage for S{subject} {finger_name}")
    target_sd = max(float(np.std(cleaned)), 1.0e-8)
    collapse_threshold = max(1.0e-4, 0.05 * target_sd)
    seed_sd = {str(seed): float(np.std(values)) for seed, values in predictions.items()}
    collapsed = [seed for seed in seeds if seed_sd[str(seed)] < collapse_threshold]
    included = [seed for seed in seeds if seed not in collapsed]
    if not included:
        raise RuntimeError(f"all requested seeds collapsed for S{subject} {finger_name}")
    return included, {
        "selection_partition": "training-partition out-of-fold predictions only",
        "collapse_threshold": collapse_threshold,
        "seed_oof_prediction_sd": seed_sd,
        "included_seeds": included,
        "collapsed_seeds": collapsed,
    }


def restore_model(
    checkpoint_path: Path,
    summary_path: Path,
    input_channels: int,
    device: torch.device,
) -> ExactWindowFingerDecoder:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint["model_state_dict"]
    selected = np.asarray(checkpoint["feature_indices"], dtype=np.int64)
    summary = json.loads(summary_path.read_text())
    component_count = int(state["spatial.weight"].shape[0])
    hidden_size = int(state["temporal.weight"].shape[1])
    model = ExactWindowFingerDecoder(
        input_channels=input_channels,
        component_count=component_count,
        selected_indices=selected,
        feature_mean=np.asarray(state["feature_mean"], dtype=np.float32),
        feature_scale=np.asarray(state["feature_scale"], dtype=np.float32),
        hidden_size=hidden_size,
        frontend="asymmetric",
        head_initialization="lars_linear_regime",
        output_activation=summary["configuration"]["output_activation"],
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.inference_mode()
def predict_contiguous(
    model: ExactWindowFingerDecoder,
    windows: torch.Tensor,
    start: int,
    stop: int,
    chunk_steps: int,
) -> np.ndarray:
    parts: list[np.ndarray] = []
    for begin in range(start, stop, chunk_steps):
        end = min(stop, begin + chunk_steps)
        prediction = model(windows[begin:end][None])[0]
        parts.append(prediction.float().cpu().numpy())
    return np.concatenate(parts)


def plot_validation(
    path: Path,
    raw_target: np.ndarray,
    cleaned_target: np.ndarray,
    prediction: np.ndarray,
    title: str,
) -> None:
    groups = movement_groups(cleaned_target, 0.08)
    selected = sorted(
        groups,
        key=lambda group: float(
            np.max(cleaned_target[group["start"]:group["stop"]])
        ),
        reverse=True,
    )[:12]
    figure, axes = plt.subplots(4, 3, figsize=(14, 10))
    for axis, group in zip(axes.flat, selected):
        start, stop = group["start"], group["stop"]
        time = np.arange(start, stop) / 25.0
        axis.plot(
            time,
            raw_target[start:stop],
            color="#94a3b8",
            linewidth=0.7,
            label="raw glove",
        )
        axis.plot(
            time,
            cleaned_target[start:stop],
            color="black",
            linewidth=0.9,
            label="cleaned target",
        )
        axis.plot(time, prediction[start:stop], color="#2563eb", linewidth=0.9, label="CV ensemble")
        axis.set_title(f"{start / 25:.1f}-{stop / 25:.1f} s", fontsize=9)
    for axis in axes.flat[len(selected):]:
        axis.set_visible(False)
    axes.flat[0].legend(frameon=False, ncol=3, fontsize=8)
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--fold-root", type=Path, required=True)
    parser.add_argument("--target-map", type=Path, required=True)
    parser.add_argument(
        "--ensemble-map",
        type=Path,
        default=None,
        help="optional frozen per-finger input-root and seed overrides",
    )
    parser.add_argument("--prepared-root", type=Path, default=Path("outputs/preprocessed_v2"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--subjects", type=int, nargs="+", default=(1, 2, 3))
    parser.add_argument("--fingers", nargs="+", choices=tuple(FINGER_NAMES), default=tuple(FINGER_NAMES))
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1))
    parser.add_argument("--window-samples", type=int, default=1000)
    parser.add_argument("--stride-samples", type=int, default=40)
    parser.add_argument("--prediction-chunk-steps", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    target_map = yaml.safe_load(args.target_map.read_text())
    ensemble_map = yaml.safe_load(args.ensemble_map.read_text()) if args.ensemble_map else {}
    device = torch.device(args.device)
    report: dict[str, object] = {
        "protocol": "frozen event-crossfold seed ensemble on official final validation",
        "official_final_validation_touched": True,
        "released_test_touched": False,
        "subjects": {},
    }
    for subject in args.subjects:
        prepared = args.prepared_root / f"sub{subject}"
        ecog = torch.from_numpy(np.array(np.load(prepared / "train_ecog.npy", mmap_mode="r"), copy=True)).to(device)
        windows = ecog.unfold(0, args.window_samples, args.stride_samples)
        raw = np.load(prepared / "train_glove_25hz_raw.npy")[24 : 24 + windows.shape[0]]
        subject_targets = target_map.get(subject, target_map.get(str(subject)))
        subject_report: dict[str, object] = {"per_finger": {}}
        subject_output = args.output_root / f"sub{subject}"
        subject_output.mkdir(parents=True, exist_ok=True)
        for finger_name in args.fingers:
            finger = list(FINGER_NAMES).index(finger_name)
            definition = json.loads(
                (args.fold_root / f"sub{subject}" / finger_name / "folds.json").read_text()
            )
            validation_start = int(definition["training_rows"])
            validation_stop = int(windows.shape[0])
            target_name = subject_targets[finger_name]
            cleaned = np.load(prepared / f"train_glove_{target_name}.npy")[
                24 + validation_start : 24 + validation_stop, finger
            ]
            raw_target = raw[validation_start:validation_stop, finger]
            model_root, requested_seeds = resolve_ensemble_spec(
                args.input_root,
                tuple(args.seeds),
                ensemble_map,
                subject,
                finger_name,
            )
            included_seeds, inclusion_report = frozen_oof_seed_inclusion(
                model_root,
                args.fold_root,
                subject,
                finger_name,
                requested_seeds,
            )
            predictions: list[np.ndarray] = []
            member_reports: list[dict[str, object]] = []
            for seed in included_seeds:
                for fold in range(3):
                    root = model_root / f"sub{subject}" / finger_name / f"fold{fold}" / f"seed{seed}"
                    model = restore_model(
                        root / "model.pt", root / "summary.json", ecog.shape[1], device
                    )
                    prediction = predict_contiguous(
                        model, windows, validation_start, validation_stop,
                        args.prediction_chunk_steps,
                    )
                    predictions.append(prediction)
                    member_reports.append(
                        {
                            "seed": seed,
                            "fold": fold,
                            "raw_pcc": pearson(prediction, raw_target),
                            "prediction_sd": float(np.std(prediction)),
                        }
                    )
                    del model
                    torch.cuda.empty_cache()
            ensemble = np.mean(np.stack(predictions), axis=0)
            groups = movement_groups(cleaned, 0.08)
            metrics = morphology_metrics(ensemble, cleaned, groups)
            metrics["raw_pcc"] = pearson(ensemble, raw_target)
            np.save(subject_output / f"{finger_name}_validation_prediction.npy", ensemble)
            plot_validation(
                subject_output / f"{finger_name}_validation_events.png",
                raw_target, cleaned, ensemble,
                f"Subject {subject} {finger_name}: official final validation",
            )
            subject_report["per_finger"][finger_name] = {
                "target": target_name,
                "model_root": str(model_root),
                "requested_seeds": list(requested_seeds),
                "metrics": metrics,
                "seed_inclusion": inclusion_report,
                "members": member_reports,
            }
        report["subjects"][str(subject)] = subject_report
        del windows, ecog
        torch.cuda.empty_cache()
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
