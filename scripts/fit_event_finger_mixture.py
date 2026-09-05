#!/usr/bin/env python3
"""Soft event-level finger attribution from glove and OOF neural evidence.

Latent states comprise idle, five single-finger states, and all ten two-finger
states.  This is a diagnostic generalized-EM model: it never reads official
validation or released test targets, and ambiguous events remain soft.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

from diagnose_cross_finger_oof import contiguous_events, reconstruct_oof, robust_evidence
from ecog_decoding.training import FINGER_NAMES


TARGETS = {1: "local_w2_q10", 2: "local_w1_q10", 3: "local_w4_q20"}


def state_dictionary() -> tuple[np.ndarray, list[str]]:
    codes = [np.zeros(5, dtype=np.float64)]
    names = ["idle"]
    for finger, name in enumerate(FINGER_NAMES):
        code = np.zeros(5, dtype=np.float64)
        code[finger] = 1.0
        codes.append(code)
        names.append(name)
    for left, right in combinations(range(5), 2):
        code = np.zeros(5, dtype=np.float64)
        code[[left, right]] = 1.0 / np.sqrt(2.0)
        codes.append(code)
        names.append(f"{FINGER_NAMES[left]}+{FINGER_NAMES[right]}")
    return np.stack(codes), names


def normalize_rows(values: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norm, 1.0e-8)


def event_signatures(
    cleaned: np.ndarray,
    evidence: np.ndarray,
    threshold: float,
    merge_gap_bins: int,
) -> tuple[list[tuple[int, int]], np.ndarray, np.ndarray]:
    smoothed = ndimage.gaussian_filter1d(cleaned, sigma=1.0, axis=0, mode="nearest")
    scale = np.maximum(np.quantile(smoothed, 0.99, axis=0), 1.0e-4)
    active = (smoothed / scale >= threshold).any(axis=1)
    active = ndimage.binary_closing(active, structure=np.ones(merge_gap_bins + 1))
    events = contiguous_events(active, minimum_samples=3)
    glove_rows: list[np.ndarray] = []
    neural_rows: list[np.ndarray] = []
    for start, stop in events:
        glove_rows.append(np.quantile(np.maximum(smoothed[start:stop], 0.0), 0.90, axis=0) / scale)
        neural_rows.append(np.quantile(np.maximum(evidence[start:stop], 0.0), 0.90, axis=0))
    return events, normalize_rows(np.stack(glove_rows)), normalize_rows(np.stack(neural_rows))


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponential = np.exp(shifted)
    return exponential / exponential.sum(axis=1, keepdims=True)


def ridge_nonnegative_map(
    latent: np.ndarray,
    observed: np.ndarray,
    identity_strength: float,
) -> np.ndarray:
    augmented_x = np.vstack((latent, np.sqrt(identity_strength) * np.eye(5)))
    coefficients = np.empty((5, 5), dtype=np.float64)
    for observed_finger in range(5):
        prior = np.zeros(5, dtype=np.float64)
        prior[observed_finger] = np.sqrt(identity_strength)
        augmented_y = np.r_[observed[:, observed_finger], prior]
        coefficients[observed_finger] = np.linalg.lstsq(
            augmented_x, augmented_y, rcond=None
        )[0]
    coefficients = np.maximum(coefficients, 0.0)
    column_norm = np.linalg.norm(coefficients, axis=0, keepdims=True)
    return coefficients / np.maximum(column_norm, 1.0e-8)


def fit_mixture(
    glove: np.ndarray,
    neural: np.ndarray,
    iterations: int,
    temperature: float,
    neural_weight: float,
    identity_strength: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float], np.ndarray, list[str]]:
    codes, names = state_dictionary()
    glove_map = np.eye(5, dtype=np.float64)
    neural_map = np.eye(5, dtype=np.float64)
    prior = np.r_[0.02, np.full(5, 0.80 / 5), np.full(10, 0.18 / 10)]
    posterior = np.tile(prior, (glove.shape[0], 1))
    objective_history: list[float] = []
    for _ in range(iterations):
        glove_templates = normalize_rows(codes @ glove_map.T)
        neural_templates = normalize_rows(codes @ neural_map.T)
        glove_error = np.sum(
            np.square(glove[:, None, :] - glove_templates[None, :, :]), axis=2
        )
        neural_error = np.sum(
            np.square(neural[:, None, :] - neural_templates[None, :, :]), axis=2
        )
        logits = (
            np.log(np.maximum(prior, 1.0e-12))[None]
            - (glove_error + neural_weight * neural_error) / temperature
        )
        updated = softmax(logits)
        latent = updated @ codes
        glove_map = ridge_nonnegative_map(latent, glove, identity_strength)
        neural_map = ridge_nonnegative_map(latent, neural, identity_strength)
        objective_history.append(
            float(np.sum(updated * (glove_error + neural_weight * neural_error)))
        )
        if np.max(np.abs(updated - posterior)) < 1.0e-6:
            posterior = updated
            break
        posterior = updated
    return posterior, glove_map, neural_map, objective_history, codes, names


def render_diagnostic(
    path: Path,
    posterior: np.ndarray,
    glove_map: np.ndarray,
    neural_map: np.ndarray,
    state_names: list[str],
    subject: int,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for axis, matrix, title in (
        (axes[0], glove_map, "Learned glove mixing"),
        (axes[1], neural_map, "Learned decoder evidence"),
    ):
        image = axis.imshow(matrix, vmin=0, vmax=1, cmap="Blues")
        axis.set_xticks(range(5), FINGER_NAMES, rotation=35, ha="right")
        axis.set_yticks(range(5), FINGER_NAMES)
        axis.set_xlabel("latent finger")
        axis.set_ylabel("observed channel")
        axis.set_title(title)
        for row in range(5):
            for column in range(5):
                axis.text(column, row, f"{matrix[row, column]:.2f}", ha="center", va="center", fontsize=8)
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    assignments = np.argmax(posterior, axis=1)
    counts = np.bincount(assignments, minlength=len(state_names))
    axes[2].bar(np.arange(len(state_names)), counts, color="#2563eb")
    axes[2].set_xticks(np.arange(len(state_names)), state_names, rotation=60, ha="right", fontsize=8)
    axes[2].set_ylabel("MAP event count")
    axes[2].set_title("Soft-mixture MAP states")
    figure.suptitle(f"Subject {subject}: event-level glove/neural attribution")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", type=int, nargs="+", default=(1, 2, 3))
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1))
    parser.add_argument("--input-root", type=Path, default=Path("outputs/event_lars_lstm_wavelet_nested_v1"))
    parser.add_argument("--fold-root", type=Path, default=Path("outputs/event_stratified_folds_v1"))
    parser.add_argument("--prepared-root", type=Path, default=Path("outputs/preprocessed_v2"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/event_finger_mixture_v1"))
    parser.add_argument("--threshold", type=float, default=0.08)
    parser.add_argument("--merge-gap-bins", type=int, default=12)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.15)
    parser.add_argument("--neural-weight", type=float, default=1.0)
    parser.add_argument("--identity-strength", type=float, default=4.0)
    parser.add_argument("--ambiguity-threshold", type=float, default=0.70)
    args = parser.parse_args()

    for subject in args.subjects:
        prediction = reconstruct_oof(
            args.input_root, args.fold_root, subject, list(args.seeds)
        )
        rows = prediction.shape[0]
        prepared = args.prepared_root / f"sub{subject}"
        cleaned = np.load(prepared / f"train_glove_{TARGETS[subject]}.npy")[24 : 24 + rows]
        scale = np.maximum(0.08, 0.20 * np.quantile(cleaned, 0.99, axis=0))
        rest = ~(cleaned > scale).any(axis=1)
        evidence = robust_evidence(prediction, rest)
        events, glove, neural = event_signatures(
            cleaned, evidence, args.threshold, args.merge_gap_bins
        )
        posterior, glove_map, neural_map, objective, codes, state_names = fit_mixture(
            glove,
            neural,
            args.iterations,
            args.temperature,
            args.neural_weight,
            args.identity_strength,
        )
        maximum = posterior.max(axis=1)
        assignment = np.argmax(posterior, axis=1)
        records = []
        for event, probabilities, confidence, state in zip(
            events, posterior, maximum, assignment
        ):
            records.append(
                {
                    "start": event[0],
                    "stop": event[1],
                    "map_state": state_names[int(state)],
                    "confidence": float(confidence),
                    "ambiguous": bool(confidence < args.ambiguity_threshold),
                    "posterior": {
                        name: float(value)
                        for name, value in zip(state_names, probabilities)
                        if value >= 0.01
                    },
                }
            )
        destination = args.output_root / f"sub{subject}"
        destination.mkdir(parents=True, exist_ok=True)
        np.save(destination / "event_posterior.npy", posterior)
        np.save(destination / "event_latent_finger_weights.npy", posterior @ codes)
        render_diagnostic(
            destination / "event_finger_mixture.png",
            posterior,
            glove_map,
            neural_map,
            state_names,
            subject,
        )
        report = {
            "subject": subject,
            "protocol": "diagnostic generalized EM using cleaned training glove and OOF neural evidence only",
            "official_final_validation_touched": False,
            "released_test_touched": False,
            "event_count": len(events),
            "ambiguous_event_fraction": float(np.mean(maximum < args.ambiguity_threshold)),
            "state_names": state_names,
            "map_state_counts": {
                name: int(np.sum(assignment == index))
                for index, name in enumerate(state_names)
            },
            "glove_mixing_observed_by_latent": glove_map.tolist(),
            "decoder_evidence_observed_by_latent": neural_map.tolist(),
            "objective_history": objective,
            "events": records,
        }
        (destination / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "subject": subject,
                    "events": len(events),
                    "ambiguous_fraction": report["ambiguous_event_fraction"],
                    "map_state_counts": report["map_state_counts"],
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
