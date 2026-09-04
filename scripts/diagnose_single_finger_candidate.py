#!/usr/bin/env python3
"""Compare one experimental finger trace with the frozen baseline visually."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ecog_decoding.training import FINGER_NAMES
try:
    from train_blocked_hurdle_gru import morphology, pearson
except ModuleNotFoundError:  # imported as scripts.diagnose_single_finger_candidate
    from scripts.train_blocked_hurdle_gru import morphology, pearson


def load_prediction(root: Path, split: str) -> np.ndarray:
    return np.load(root / f"{split}_prediction.npy")


def strongest_windows(target: np.ndarray, finger: int, width: int) -> list[int]:
    other = np.max(np.delete(target, finger, axis=1), axis=1)
    isolated = np.maximum(target[:, finger] - other, 0.0)
    score = np.convolve(isolated, np.ones(width), mode="valid")
    order = np.argsort(score)[::-1]
    selected: list[int] = []
    for start in order:
        if all(abs(int(start) - old) >= width for old in selected):
            selected.append(int(start))
        if len(selected) == 2:
            break
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, required=True)
    parser.add_argument("--finger", choices=FINGER_NAMES, required=True)
    parser.add_argument("--prepared-root", type=Path, default=Path("outputs/preprocessed_v2"))
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--history", type=int, default=25)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--window-bins", type=int, default=200)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    finger = FINGER_NAMES.index(args.finger)
    root = args.prepared_root / f"sub{args.subject}"
    metadata = json.loads((root / "metadata.json").read_text())
    split = int(metadata["target_fit_samples_25hz"])
    offset = args.history - 1
    validation_cleaned = np.load(root / f"train_glove_{args.target}.npy")[split:, finger]
    validation_raw = np.load(root / "train_glove_25hz_raw.npy")[split:, finger]
    test_cleaned_all = np.load(root / f"test_glove_{args.target}.npy")[offset:]
    test_raw = np.load(root / "test_glove_25hz_raw.npy")[offset:, finger]
    predictions = {
        "baseline": {
            "validation": load_prediction(args.baseline, "validation")[:, finger],
            "test": load_prediction(args.baseline, "test")[:, finger],
        },
        "candidate": {
            "validation": load_prediction(args.candidate, "validation")[:, finger],
            "test": load_prediction(args.candidate, "test")[:, finger],
        },
    }
    report = {
        "subject": args.subject,
        "finger": args.finger,
        "selection_policy": "candidate remains experimental unless it beats the baseline on chronological validation",
        "methods": {},
    }
    for name, values in predictions.items():
        report["methods"][name] = {
            "validation_raw_pcc": pearson(values["validation"], validation_raw),
            "validation_cleaned_morphology": morphology(
                values["validation"], validation_cleaned, args.threshold
            ),
            "test_raw_pcc": pearson(values["test"], test_raw),
            "test_cleaned_morphology": morphology(
                values["test"], test_cleaned_all[:, finger], args.threshold
            ),
        }

    starts = strongest_windows(test_cleaned_all, finger, args.window_bins)
    global_rest = np.max(test_cleaned_all, axis=1) < args.threshold
    candidate_error = np.abs(predictions["candidate"]["test"]) * global_rest
    rest_score = np.convolve(candidate_error, np.ones(args.window_bins), mode="valid")
    rest_count = np.convolve(global_rest.astype(float), np.ones(args.window_bins), mode="valid")
    valid = np.flatnonzero(rest_count == args.window_bins)
    starts.append(int(valid[np.argmax(rest_score[valid])]) if valid.size else int(np.argmax(rest_score)))

    figure, axes = plt.subplots(3, 1, figsize=(13, 8), constrained_layout=True)
    for index, (axis, start) in enumerate(zip(axes, starts, strict=True)):
        stop = start + args.window_bins
        time = np.arange(args.window_bins) / 25.0 + start / 25.0
        axis.plot(time, test_cleaned_all[start:stop, finger], color="black", lw=2.2, label="cleaned target")
        axis.plot(time, predictions["baseline"]["test"][start:stop], color="#2563eb", lw=1.5, label="baseline")
        axis.plot(time, predictions["candidate"]["test"][start:stop], color="#dc2626", lw=1.5, label="candidate")
        axis.axhline(args.threshold, color="#777", ls="--", lw=0.8)
        axis.set_title("isolated movement" if index < 2 else "worst candidate rest window")
        axis.grid(alpha=0.2)
    axes[-1].set_xlabel("held-out test time after history offset (s)")
    axes[0].legend(frameon=False, ncol=3)
    figure.suptitle(f"S{args.subject} {args.finger}: frozen baseline vs experimental candidate")
    args.output.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output / "comparison.png", dpi=160)
    plt.close(figure)
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
