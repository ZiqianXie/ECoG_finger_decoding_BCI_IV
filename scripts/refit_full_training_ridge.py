#!/usr/bin/env python3
"""Refit a validation-selected ridge decoder on all labeled training data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from benchmark_ridge_target_variants import (
    correlation_screen,
    lagged,
    ridge_fit,
    ridge_predict,
)
from ecog_decoding.training import FINGER_NAMES, trajectory_metrics


def selected_models(summary: dict[str, object], target: str) -> dict[str, object]:
    if "result" in summary:
        return summary["result"]["per_finger"]
    if "results" in summary:
        return summary["results"][target]["per_finger"]
    raise ValueError("unrecognized source summary structure")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, required=True)
    parser.add_argument("--prepared-root", type=Path, default=Path("outputs/preprocessed_v2"))
    parser.add_argument("--energy-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/full_training_ridge_v1"))
    parser.add_argument("--target", required=True)
    parser.add_argument("--history", type=int, default=25)
    parser.add_argument("--top-features", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    prepared = args.prepared_root / f"sub{args.subject}"
    energy_root = args.energy_root / f"sub{args.subject}"
    output = args.output_root / f"sub{args.subject}"
    output.mkdir(parents=True, exist_ok=True)
    train_energy_name = (
        "train_csp_energy.npy"
        if (energy_root / "train_csp_energy.npy").exists()
        else "train_initialized_energy.npy"
    )
    test_energy_name = (
        "test_csp_energy.npy"
        if (energy_root / "test_csp_energy.npy").exists()
        else "test_initialized_energy.npy"
    )
    train_energy = np.load(energy_root / train_energy_name)
    test_energy = np.load(energy_root / test_energy_name)
    target = np.load(prepared / f"train_glove_{args.target}.npy")
    raw_test = np.load(prepared / "test_glove_25hz_raw.npy")
    source_summary = json.loads((energy_root / "summary.json").read_text())
    source_models = selected_models(source_summary, args.target)
    train_x = lagged(train_energy, args.history)
    test_x = lagged(test_energy, args.history)
    train_y = target[args.history - 1 :]
    test_y = raw_test[args.history - 1 :]
    prediction = np.zeros_like(test_y, dtype=np.float32)
    report: dict[str, object] = {}
    device = torch.device(args.device)
    for finger, name in enumerate(FINGER_NAMES):
        selected = correlation_screen(
            train_x,
            train_y[:, finger],
            min(args.top_features, train_x.shape[1]),
        )
        alpha = float(source_models[name]["best_alpha"])
        fit = ridge_fit(train_x[:, selected], train_y[:, finger], alpha, device)
        prediction[:, finger] = ridge_predict(test_x[:, selected], fit)
        report[name] = {
            "alpha_selected_before_refit": alpha,
            "selected_feature_count": int(selected.size),
        }
    metrics = trajectory_metrics(prediction, test_y)
    np.save(output / "test_prediction.npy", prediction, allow_pickle=False)
    result = {
        "subject": args.subject,
        "method": "validation-selected feature count and ridge alpha refit on all 400 seconds",
        "source_energy_root": str(args.energy_root),
        "target": args.target,
        "history": args.history,
        "top_features": args.top_features,
        "per_finger": report,
        "test_raw_metrics": metrics,
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
