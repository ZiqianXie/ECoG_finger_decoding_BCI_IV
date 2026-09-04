#!/usr/bin/env python3
"""Post-hoc fit/validation/test shift audit for one subject and finger."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ecog_decoding.training import FINGER_NAMES


def column_correlation(values: np.ndarray, target: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    centered_values = values - values.mean(axis=0, keepdims=True)
    centered_target = target - target.mean()
    numerator = centered_values.T @ centered_target
    denominator = np.linalg.norm(centered_values, axis=0) * np.linalg.norm(
        centered_target
    )
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0,
    )


def vector_cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(left @ right / denominator) if denominator > 0 else 0.0


def target_summary(values: np.ndarray, threshold: float) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "q50": float(np.quantile(values, 0.50)),
        "q95": float(np.quantile(values, 0.95)),
        "q99": float(np.quantile(values, 0.99)),
        "active_fraction": float(np.mean(values >= threshold)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, required=True)
    parser.add_argument("--finger", choices=FINGER_NAMES, required=True)
    parser.add_argument(
        "--prepared-root", type=Path, default=Path("outputs/preprocessed_v2")
    )
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--selection-root", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--history", type=int, default=25)
    parser.add_argument("--movement-threshold", type=float, default=0.1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    finger = FINGER_NAMES.index(args.finger)
    prepared = args.prepared_root / f"sub{args.subject}"
    features = args.feature_root / f"sub{args.subject}"
    selection = args.selection_root / f"sub{args.subject}"
    metadata = json.loads((prepared / "metadata.json").read_text())
    split = int(metadata["target_fit_samples_25hz"])
    offset = args.history - 1
    train_count = split - offset
    selection_summary = json.loads((selection / "summary.json").read_text())
    indices = np.asarray(
        selection_summary["per_finger"][args.finger]["selected_source_indices"],
        dtype=np.int64,
    )

    train_features = np.load(
        features / "train_initialized_window_features.npy", mmap_mode="r"
    )[:, indices]
    test_features = np.load(
        features / "test_initialized_window_features.npy", mmap_mode="r"
    )[:, indices]
    train_cleaned = np.load(prepared / f"train_glove_{args.target}.npy")[
        offset:, finger
    ]
    test_cleaned = np.load(prepared / f"test_glove_{args.target}.npy")[
        offset:, finger
    ]
    train_raw = np.load(prepared / "train_glove_25hz_raw.npy")[offset:, finger]
    test_raw = np.load(prepared / "test_glove_25hz_raw.npy")[offset:, finger]

    partitions = {
        "fit": (np.asarray(train_features[:train_count]), train_cleaned[:train_count]),
        "validation": (
            np.asarray(train_features[train_count:]),
            train_cleaned[train_count:],
        ),
        "test": (np.asarray(test_features), test_cleaned),
    }
    correlations = {
        name: column_correlation(values, target)
        for name, (values, target) in partitions.items()
    }
    fit_mean = partitions["fit"][0].mean(axis=0, dtype=np.float64)
    fit_std = partitions["fit"][0].std(axis=0, dtype=np.float64)
    fit_std[fit_std < 1.0e-8] = 1.0
    standardized_mean_shift = {
        name: (values.mean(axis=0, dtype=np.float64) - fit_mean) / fit_std
        for name, (values, _) in partitions.items()
        if name != "fit"
    }
    std_ratio = {
        name: values.std(axis=0, dtype=np.float64) / fit_std
        for name, (values, _) in partitions.items()
        if name != "fit"
    }

    cleaned_targets = {
        "fit": train_cleaned[:train_count],
        "validation": train_cleaned[train_count:],
        "test": test_cleaned,
    }
    raw_targets = {
        "fit": train_raw[:train_count],
        "validation": train_raw[train_count:],
        "test": test_raw,
    }
    report = {
        "subject": args.subject,
        "finger": args.finger,
        "target": args.target,
        "selected_feature_count": int(indices.size),
        "feature_target_correlation": {
            "cosine_fit_validation": vector_cosine(
                correlations["fit"], correlations["validation"]
            ),
            "cosine_fit_test": vector_cosine(
                correlations["fit"], correlations["test"]
            ),
            "cosine_validation_test": vector_cosine(
                correlations["validation"], correlations["test"]
            ),
            "sign_agreement_fit_validation": float(
                np.mean(np.sign(correlations["fit"]) == np.sign(correlations["validation"]))
            ),
            "sign_agreement_fit_test": float(
                np.mean(np.sign(correlations["fit"]) == np.sign(correlations["test"]))
            ),
            "median_absolute": {
                name: float(np.median(np.abs(values)))
                for name, values in correlations.items()
            },
        },
        "feature_distribution_shift": {
            name: {
                "median_absolute_standardized_mean": float(np.median(np.abs(values))),
                "q95_absolute_standardized_mean": float(np.quantile(np.abs(values), 0.95)),
                "median_std_ratio": float(np.median(std_ratio[name])),
                "q95_std_ratio": float(np.quantile(std_ratio[name], 0.95)),
            }
            for name, values in standardized_mean_shift.items()
        },
        "cleaned_target": {
            name: target_summary(values, args.movement_threshold)
            for name, values in cleaned_targets.items()
        },
        "raw_target": {
            name: target_summary(values, args.movement_threshold)
            for name, values in raw_targets.items()
        },
        "test_labels_used_for_posthoc_audit": True,
        "selection_policy": "diagnostic only; never use this report to fit or choose a submitted model",
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes[0, 0].scatter(correlations["fit"], correlations["validation"], s=12, alpha=0.45)
    axes[0, 0].axline((0, 0), slope=1, color="black", lw=1, ls="--")
    axes[0, 0].set(xlabel="fit feature-target r", ylabel="validation r", title="Fit vs validation")
    axes[0, 1].scatter(correlations["fit"], correlations["test"], s=12, alpha=0.45, color="#dc2626")
    axes[0, 1].axline((0, 0), slope=1, color="black", lw=1, ls="--")
    axes[0, 1].set(xlabel="fit feature-target r", ylabel="test r", title="Fit vs released test")
    bins = np.linspace(-3, 3, 61)
    for name, color in (("validation", "#2563eb"), ("test", "#dc2626")):
        axes[1, 0].hist(
            standardized_mean_shift[name], bins=bins, alpha=0.5, density=True,
            label=name, color=color,
        )
    axes[1, 0].set(xlabel="selected-feature mean shift (fit SD)", ylabel="density", title="Feature distribution shift")
    axes[1, 0].legend()
    target_bins = np.linspace(0, 1.5, 61)
    for name, color in (("fit", "#111827"), ("validation", "#2563eb"), ("test", "#dc2626")):
        axes[1, 1].hist(cleaned_targets[name], bins=target_bins, histtype="step", density=True, lw=1.5, label=name, color=color)
    axes[1, 1].set(xlabel="cleaned target amplitude", ylabel="density", title="Target distribution")
    axes[1, 1].legend()
    for axis in axes.ravel():
        axis.grid(alpha=0.18)
    fig.suptitle(f"Subject {args.subject} {args.finger}: partition-shift audit")
    fig.tight_layout()
    args.output.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output / "partition_shift.png", dpi=160)
    plt.close(fig)
    (args.output / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
