#!/usr/bin/env python3
"""Assemble the explicitly retrospective per-finger extension and its figures.

The routing file is a diagnostic record, not a model-selection protocol: it may
name different saved prediction sources for different fingers after the
released test set has been inspected.  The generated JSON therefore labels the
result as retrospective and keeps it separate from the leakage-controlled
full-refit table.
"""

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
from evaluate_cv_ensemble_final_validation import movement_groups
from summarize_event_lars_lstm_cv import morphology_metrics, pearson
from summarize_frozen_full_refit import PAPER_PCC


COLORS = {
    "paper": "#94a3b8",
    "extension": "#2563eb",
    "target": "#111827",
    "prediction": "#2563eb",
}


def resolve_subject_path(value: object, subject: int) -> Path:
    path = Path(str(value))
    if (path / "test_prediction.npy").exists() or path.name == f"sub{subject}":
        return path
    return path / f"sub{subject}"


def load_routed_prediction(
    routing: dict[str, object], subject: int, expected_shape: tuple[int, int]
) -> tuple[np.ndarray, dict[str, str]]:
    subject_map = routing.get(subject, routing.get(str(subject)))
    if not isinstance(subject_map, dict):
        raise KeyError(f"routing has no subject {subject}")
    prediction = np.empty(expected_shape, dtype=np.float64)
    sources: dict[str, str] = {}
    for finger, name in enumerate(FINGER_NAMES):
        root = resolve_subject_path(subject_map[name], subject)
        values = np.load(root / "test_prediction.npy")
        if values.shape != expected_shape:
            raise ValueError(f"{root} has shape {values.shape}, expected {expected_shape}")
        prediction[:, finger] = values[:, finger]
        sources[name] = str(root / "test_prediction.npy")
    return prediction, sources


def load_source_audit(prediction_path: str) -> dict[str, object] | None:
    """Expose compact protocol flags from a routed candidate's summary."""
    path = Path(prediction_path)
    for summary_path in (path.parent / "summary.json", path.parent.parent / "summary.json"):
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text())
        keys = (
            "protocol",
            "released_test_used_for_weight_selection",
            "suitable_as_confirmatory_benchmark",
            "right_weight",
            "left_pcc",
            "right_pcc",
            "blend_pcc",
        )
        audit = {key: summary[key] for key in keys if key in summary}
        if audit:
            audit["summary"] = str(summary_path)
            return audit
    return None


def smooth_nonnegative_projection(values: np.ndarray, baseline: float, beta: float = 10.0) -> np.ndarray:
    centered = np.asarray(values, dtype=np.float64) - baseline
    projected = np.logaddexp(0.0, beta * centered) / beta - np.log(2.0) / beta
    return np.maximum(projected, 0.0)


def normalize_for_display(
    prediction: np.ndarray, development_target: np.ndarray
) -> tuple[np.ndarray, dict[str, float]]:
    """Map an arbitrary decoder coordinate to nonnegative flexion without test labels."""
    baseline = float(np.quantile(prediction, 0.20))
    projected = smooth_nonnegative_projection(prediction, baseline)
    source_q995 = float(np.quantile(projected, 0.995))
    target_q995 = float(np.quantile(np.maximum(development_target, 0.0), 0.995))
    gain = target_q995 / source_q995 if source_q995 > 1.0e-8 else 1.0
    return projected * gain, {
        "prediction_baseline_quantile": baseline,
        "projected_prediction_q995": source_q995,
        "development_target_q995": target_q995,
        "gain": gain,
    }


def plot_pcc(path: Path, results: dict[int, list[float]]) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    x = np.arange(len(FINGER_NAMES))
    for axis, subject in zip(axes, (1, 2, 3)):
        paper = np.asarray(PAPER_PCC[subject])
        current = np.asarray(results[subject])
        axis.bar(x - 0.2, paper, width=0.4, color=COLORS["paper"], label="2018 paper")
        axis.bar(x + 0.2, current, width=0.4, color=COLORS["extension"], label="2026 retrospective extension")
        axis.axhline(0.0, color="#cbd5e1", linewidth=0.7)
        axis.set_xticks(x, [name.title() for name in FINGER_NAMES], rotation=35, ha="right")
        axis.set_title(f"Subject {subject}  macro {current.mean():.3f} vs {paper.mean():.3f}")
        axis.grid(axis="y", color="#e2e8f0", linewidth=0.6)
    axes[0].set_ylabel("Released-test Pearson correlation")
    axes[0].legend(frameon=False, fontsize=9)
    figure.suptitle("Per-finger reconstruction: paper and retrospective diagnostic ceiling")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_event_windows(
    path: Path,
    target: np.ndarray,
    prediction: np.ndarray,
    subject: int,
    threshold: float,
) -> None:
    figure, axes = plt.subplots(5, 3, figsize=(15, 13), sharey="row")
    for finger, name in enumerate(FINGER_NAMES):
        groups = movement_groups(target[:, finger], threshold)
        selected = sorted(
            groups,
            key=lambda group: float(np.max(target[group["start"] : group["stop"], finger])),
            reverse=True,
        )[:3]
        for column, axis in enumerate(axes[finger]):
            if column >= len(selected):
                axis.set_visible(False)
                continue
            group = selected[column]
            start, stop = int(group["start"]), int(group["stop"])
            margin = 12
            start = max(0, start - margin)
            stop = min(target.shape[0], stop + margin)
            time = (np.arange(start, stop) - start) / 25.0
            axis.plot(time, target[start:stop, finger], color=COLORS["target"], linewidth=1.0, label="cleaned target")
            axis.plot(time, prediction[start:stop, finger], color=COLORS["prediction"], linewidth=1.0, label="prediction")
            axis.axhline(0.0, color="#cbd5e1", linewidth=0.6)
            axis.set_title(f"{name.title()} event {column + 1}", fontsize=9)
            if finger == 4:
                axis.set_xlabel("Time in window (s)")
        axes[finger, 0].set_ylabel(name.title())
    axes[0, 0].legend(frameon=False, ncol=2, fontsize=8)
    figure.suptitle(f"Subject {subject}: strongest released-test movements")
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def plot_morphology(path: Path, reports: dict[int, dict[str, object]]) -> None:
    metrics = ("cleaned_pcc", "velocity_pcc", "movement_state_f1", "median_event_peak_ratio")
    labels = ("Cleaned PCC", "Velocity PCC", "Movement F1", "Median peak ratio")
    figure, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    x = np.arange(5)
    for axis, metric, label in zip(axes, metrics, labels):
        for subject, color in zip((1, 2, 3), ("#2563eb", "#db2777", "#16a34a")):
            values = [reports[subject][name][metric] for name in FINGER_NAMES]
            axis.plot(x, values, marker="o", linewidth=1.4, color=color, label=f"S{subject}")
        axis.set_xticks(x, [name.title() for name in FINGER_NAMES], rotation=35, ha="right")
        axis.set_title(label)
        axis.grid(color="#e2e8f0", linewidth=0.6)
    axes[0].legend(frameon=False)
    figure.suptitle("Trajectory morphology beyond a single PCC score")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_frequency_response(path: Path, audit_path: Path) -> None:
    audit = json.loads(audit_path.read_text())
    frequency = np.asarray(audit["frequency_hz"], dtype=np.float64)
    magnitude = np.asarray(audit["normalized_magnitude"], dtype=np.float64)
    figure, axis = plt.subplots(figsize=(11, 5.5))
    for band, values in zip(audit["band_order"], magnitude):
        axis.plot(frequency, values, linewidth=1.2, label=band)
    axis.set_xlim(0.0, float(audit["sampling_rate_hz"]) / 2.0)
    axis.set_ylim(0.0, 1.05)
    axis.set_xlabel("Frequency (Hz)")
    axis.set_ylabel("Normalized magnitude")
    axis.set_title("Initialized three-level bior6.8 dilated filter tree")
    axis.grid(color="#e2e8f0", linewidth=0.6)
    axis.legend(frameon=False, ncol=4)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--routing", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, default=Path("outputs/preprocessed_v2"))
    parser.add_argument("--frequency-audit", type=Path, default=Path("outputs/audit/wavelet_frequency_response.json"))
    parser.add_argument("--figure-root", type=Path, default=Path("docs/figures"))
    parser.add_argument("--result", type=Path, default=Path("docs/results/retrospective-extension.json"))
    parser.add_argument("--movement-threshold", type=float, default=0.08)
    args = parser.parse_args()
    routing = yaml.safe_load(args.routing.read_text())
    args.figure_root.mkdir(parents=True, exist_ok=True)
    args.result.parent.mkdir(parents=True, exist_ok=True)

    pcc: dict[int, list[float]] = {}
    morphology: dict[int, dict[str, object]] = {}
    subjects: dict[str, object] = {}
    for subject in (1, 2, 3):
        prepared = args.prepared_root / f"sub{subject}"
        raw = np.load(prepared / "test_glove_25hz_raw.npy")[24:]
        prediction, sources = load_routed_prediction(routing, subject, raw.shape)
        selected_pcc = [pearson(prediction[:, finger], raw[:, finger]) for finger in range(5)]
        pcc[subject] = selected_pcc
        per_finger: dict[str, object] = {}
        subject_morphology: dict[str, object] = {}
        cleaned_test = np.empty_like(raw, dtype=np.float64)
        display_prediction = np.empty_like(prediction, dtype=np.float64)
        subject_routing = routing[subject if subject in routing else str(subject)]
        normalization: dict[str, object] = {}
        for finger, name in enumerate(FINGER_NAMES):
            target_name = subject_routing.get("target", {}).get(name)
            if target_name is None:
                cleaned = raw[:, finger]
                development = np.load(prepared / "train_glove_25hz_raw.npy")[24:, finger]
            else:
                cleaned = np.load(prepared / f"test_glove_{target_name}.npy")[24:, finger]
                development = np.load(prepared / f"train_glove_{target_name}.npy")[24:, finger]
            display, display_audit = normalize_for_display(prediction[:, finger], development)
            cleaned_test[:, finger] = cleaned
            display_prediction[:, finger] = display
            normalization[name] = display_audit
            metrics = morphology_metrics(
                display, cleaned, movement_groups(cleaned, args.movement_threshold)
            )
            metrics["raw_pcc"] = selected_pcc[finger]
            subject_morphology[name] = metrics
            per_finger[name] = {
                "source": sources[name],
                "source_audit": load_source_audit(sources[name]),
                "raw_pcc": selected_pcc[finger],
                "paper_raw_pcc": PAPER_PCC[subject][finger],
                "delta_vs_paper": selected_pcc[finger] - PAPER_PCC[subject][finger],
                "morphology": metrics,
            }
        morphology[subject] = subject_morphology
        subjects[str(subject)] = {
            "per_finger": per_finger,
            "raw_pcc": selected_pcc,
            "macro_raw_pcc": float(np.mean(selected_pcc)),
            "paper_macro_raw_pcc": float(np.mean(PAPER_PCC[subject])),
            "display_normalization": normalization,
        }
        plot_event_windows(
            args.figure_root / f"retrospective-extension-s{subject}-events.png",
            cleaned_test,
            display_prediction,
            subject,
            args.movement_threshold,
        )
    report = {
        "protocol": "retrospective diagnostic routing after released-test inspection",
        "released_test_used_for_route_selection": True,
        "suitable_as_confirmatory_benchmark": False,
        "display_normalization": (
            "label-free test-prediction 20th-percentile baseline, smooth nonnegative "
            "projection, and gain matching to the development cleaned-target 99.5th percentile"
        ),
        "released_test_labels_used_for_display_gain": False,
        "subjects": subjects,
    }
    args.result.write_text(json.dumps(report, indent=2) + "\n")
    plot_pcc(args.figure_root / "retrospective-extension-pcc.png", pcc)
    plot_morphology(args.figure_root / "retrospective-extension-morphology.png", morphology)
    plot_frequency_response(args.figure_root / "wavelet-initialization-frequency-response.png", args.frequency_audit)
    print(json.dumps({key: value["raw_pcc"] for key, value in subjects.items()}, indent=2))


if __name__ == "__main__":
    main()
