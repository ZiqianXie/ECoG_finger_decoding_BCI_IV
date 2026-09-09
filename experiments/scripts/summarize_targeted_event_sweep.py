#!/usr/bin/env python3
"""Compare targeted event-fold models and visualize their OOF behavior."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evaluate_cv_ensemble_final_validation import movement_groups
from summarize_event_lars_lstm_cv import pearson


def load_oof(root: Path, subject: int, finger: str, seed: int) -> dict[str, object]:
    prediction = target = initialized = None
    fold_scores = []
    epochs = []
    runtimes = []
    for fold in range(3):
        path = root / f"sub{subject}" / finger / f"fold{fold}" / f"seed{seed}"
        summary = json.loads((path / "summary.json").read_text())
        indices = np.load(path / "validation_indices.npy")
        if prediction is None:
            rows = max(
                int(np.max(np.load(
                    root / f"sub{subject}" / finger / f"fold{value}" / f"seed{seed}" / "validation_indices.npy"
                ))) for value in range(3)
            ) + 1
            prediction = np.full(rows, np.nan)
            target = np.full(rows, np.nan)
            initialized = np.full(rows, np.nan)
        prediction[indices] = np.load(path / "validation_prediction.npy")
        target[indices] = np.load(path / "validation_raw_target.npy")
        initialized[indices] = np.load(path / "validation_initialized_prediction.npy")
        fold_scores.append(float(summary["outer_validation_raw_pcc"]))
        epochs.append(int(summary["selected_epoch"]))
        runtimes.append(float(summary["runtime_seconds"]))
    valid = np.isfinite(prediction) & np.isfinite(target)
    return {
        "prediction": prediction,
        "target": target,
        "initialized": initialized,
        "valid": valid,
        "oof_pcc": pearson(prediction[valid], target[valid]),
        "initialized_oof_pcc": pearson(initialized[valid], target[valid]),
        "fold_pcc": fold_scores,
        "selected_epochs": epochs,
        "runtime_seconds": runtimes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--finger", default="thumb")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--roots", type=Path, nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", default=None)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/targeted_event_sweep_summary_v1"))
    args = parser.parse_args()
    labels = args.labels or [root.name for root in args.roots]
    if len(labels) != len(args.roots):
        parser.error("--labels and --roots must have equal length")
    loaded = {
        label: load_oof(root, args.subject, args.finger, args.seed)
        for label, root in zip(labels, args.roots)
    }
    best = max(loaded, key=lambda label: float(loaded[label]["oof_pcc"]))
    report = {
        "selection_protocol": "hyperparameters compared only by purged event-fold OOF PCC",
        "subject": args.subject,
        "finger": args.finger,
        "selected": best,
        "models": {
            label: {
                key: value for key, value in result.items()
                if key not in ("prediction", "target", "initialized", "valid")
            }
            for label, result in loaded.items()
        },
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "summary.json").write_text(json.dumps(report, indent=2) + "\n")

    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(labels))
    axes[0].bar(x - 0.18, [loaded[label]["initialized_oof_pcc"] for label in labels], 0.36, label="LARS initialization")
    axes[0].bar(x + 0.18, [loaded[label]["oof_pcc"] for label in labels], 0.36, label="fine-tuned")
    axes[0].set_xticks(x, labels, rotation=35, ha="right")
    axes[0].set_ylabel("purged event-fold OOF PCC")
    axes[0].legend(frameon=False)
    axes[0].set_title("Model-size and learning-rate comparison")

    reference = loaded[best]
    valid_indices = np.flatnonzero(reference["valid"])
    target_valid = reference["target"][reference["valid"]]
    groups = movement_groups(target_valid, 0.08)
    if groups:
        center = max(groups, key=lambda group: np.max(target_valid[group["start"]:group["stop"]]))
        start = max(0, int(center["start"]) - 150)
        stop = min(target_valid.size, int(center["stop"]) + 150)
    else:
        start, stop = 0, min(500, target_valid.size)
    time = valid_indices[start:stop] / 25.0
    axes[1].plot(time, target_valid[start:stop], color="black", linewidth=1.2, label="raw glove")
    for label in labels:
        values = loaded[label]["prediction"][loaded[label]["valid"]]
        axes[1].plot(time, values[start:stop], linewidth=0.8, alpha=0.8, label=label)
    axes[1].set_xlabel("original training time (s)")
    axes[1].set_title("Representative held-out event neighborhood")
    axes[1].legend(frameon=False, fontsize=7, ncol=2)
    figure.tight_layout()
    figure.savefig(args.output_root / "comparison.png", dpi=180)
    plt.close(figure)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
