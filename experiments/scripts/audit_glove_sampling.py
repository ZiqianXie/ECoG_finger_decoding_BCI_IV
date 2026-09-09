#!/usr/bin/env python3
"""Audit glove sampling phase and baseline-order conventions independently."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ecog_decoding.io import load_subject
from ecog_decoding.preprocessing import paper_baseline_correct


FINGERS = ("thumb", "index", "middle", "ring", "little")


def pearson_columns(a: np.ndarray, b: np.ndarray) -> list[float]:
    result = []
    for column in range(a.shape[1]):
        x = a[:, column] - a[:, column].mean()
        y = b[:, column] - b[:, column].mean()
        denominator = np.linalg.norm(x) * np.linalg.norm(y)
        result.append(float(x @ y / denominator) if denominator else 0.0)
    return result


def normalize(corrected: np.ndarray, fit_stop: int) -> tuple[np.ndarray, np.ndarray]:
    scale = np.quantile(corrected[:fit_stop], 0.995, axis=0)
    floor = max(float(np.max(scale)) * 0.01, np.finfo(np.float64).eps)
    scale = np.maximum(scale, floor)
    return np.clip(corrected / scale, 0.0, 2.0).astype(np.float32), scale


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/bci_competition_iv_ds4"))
    parser.add_argument("--prepared-root", type=Path, default=Path("outputs/preprocessed_v2"))
    parser.add_argument("--smoothness", type=float, default=1.0e5)
    parser.add_argument("--plot-start-seconds", type=float, default=80.0)
    parser.add_argument("--plot-duration-seconds", type=float, default=20.0)
    args = parser.parse_args()

    subject = load_subject(args.data_root, args.subject)
    root = args.prepared_root / f"sub{args.subject}"
    metadata = json.loads((root / "metadata.json").read_text())
    split = int(metadata["target_fit_samples_25hz"])
    reference = subject.train_glove[::40]
    phase_report: dict[str, object] = {}
    for phase in range(40):
        sampled = subject.train_glove[phase::40]
        phase_report[str(phase)] = {
            "pearson_vs_phase0": pearson_columns(sampled, reference),
            "rmse_vs_phase0": np.sqrt(np.mean((sampled - reference) ** 2, axis=0)).tolist(),
        }
    reshaped = subject.train_glove.reshape(-1, 40, 5)
    variants = {
        "sample_phase0": reference,
        "sample_phase20": subject.train_glove[20::40],
        "sample_phase39": subject.train_glove[39::40],
        "bin_mean": reshaped.mean(axis=1),
    }
    variant_report: dict[str, object] = {}
    corrected: dict[str, np.ndarray] = {}
    baselines: dict[str, np.ndarray] = {}
    for name, trajectory in variants.items():
        print(f"subject={args.subject} baseline={name}", flush=True)
        result = paper_baseline_correct(trajectory, smoothness=args.smoothness)
        target, scale = normalize(result.corrected, split)
        np.save(root / f"train_glove_paper_{name}.npy", target, allow_pickle=False)
        corrected[name] = target
        baselines[name] = result.baseline
        second_difference = np.diff(result.baseline, n=2, axis=0)
        variant_report[name] = {
            "scale": scale.tolist(),
            "target_pearson_vs_phase0": None,
            "baseline_second_difference_rms": np.sqrt(
                np.mean(second_difference * second_difference, axis=0)
            ).tolist(),
            "near_zero_fraction": np.mean(target < 0.02, axis=0).tolist(),
        }
    for name in variants:
        variant_report[name]["target_pearson_vs_phase0"] = pearson_columns(
            corrected[name], corrected["sample_phase0"]
        )

    start = int(round(args.plot_start_seconds * 25.0))
    stop = min(reference.shape[0], start + int(round(args.plot_duration_seconds * 25.0)))
    time = np.arange(start, stop) / 25.0
    fig, axes = plt.subplots(5, 2, figsize=(15, 12), sharex=True)
    colors = {"sample_phase0": "#2563eb", "sample_phase39": "#f97316", "bin_mean": "#16a34a"}
    for finger, label in enumerate(FINGERS):
        left, right = axes[finger]
        left.plot(time, variants["sample_phase0"][start:stop, finger], color="#111827", lw=1.1, label="raw phase 0")
        left.plot(time, baselines["sample_phase0"][start:stop, finger], color="#dc2626", lw=1.0, label="constrained baseline")
        for name in ("sample_phase0", "sample_phase39", "bin_mean"):
            right.plot(time, corrected[name][start:stop, finger], color=colors[name], lw=1.0, label=name)
        left.set_ylabel(label)
        if finger == 0:
            left.legend(loc="upper right", ncol=2, fontsize=8)
            right.legend(loc="upper right", ncol=3, fontsize=8)
    axes[0, 0].set_title("Raw glove and fitted lower envelope")
    axes[0, 1].set_title("Normalized baseline-subtracted target")
    axes[-1, 0].set_xlabel("seconds")
    axes[-1, 1].set_xlabel("seconds")
    fig.suptitle(f"Subject {args.subject}: glove sampling and baseline audit")
    fig.tight_layout()
    fig.savefig(root / "glove_sampling_baseline_audit.png", dpi=160)
    plt.close(fig)

    report = {
        "subject": args.subject,
        "fit_stop": split,
        "phase_report": phase_report,
        "variant_report": variant_report,
        "test_labels_used": False,
    }
    (root / "glove_sampling_audit.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
