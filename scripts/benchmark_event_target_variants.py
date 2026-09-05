#!/usr/bin/env python3
"""Compare glove target variants with fixed features under outer event folds."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from ecog_decoding.training import FINGER_NAMES
from train_event_grouped_lars_lstm import (
    correlation_order,
    indices_from_intervals,
    pearson,
)


DEFAULT_METHODS = (
    "raw_25hz",
    "paper_baseline_only",
    "local_w1_q10", "local_w1_q20", "local_w1_q30",
    "local_w2_q10", "local_w2_q20", "local_w2_q30",
    "local_w3_q10", "local_w3_q20", "local_w3_q30",
    "local_w4_q10", "local_w4_q20", "local_w4_q30",
    "local_w5_q10", "local_w5_q20", "local_w5_q30",
    "local_w6_q10", "local_w6_q20", "local_w6_q30",
)

LOCAL_METHOD = re.compile(r"^local_w(?P<window>[0-9]+(?:p[0-9]+)?)_q[0-9]+$")


def target_support_bins(
    method: str,
    sampling_rate_hz: float = 25.0,
    smoothing_seconds: float = 0.16,
    gaussian_sigmas: float = 5.0,
) -> int | None:
    """Return the held-out margin needed by a precomputed target.

    Local targets use a centred percentile window followed by Gaussian
    smoothing.  Five Gaussian standard deviations make the residual influence
    of a held-out sample negligible.  The paper rope baseline is global, so no
    finite temporal margin can make a precomputed copy fold-local.
    """
    if method == "raw_25hz":
        return 0
    if method == "paper_baseline_only":
        return None
    match = LOCAL_METHOD.match(method)
    if match is None:
        raise ValueError(f"cannot determine temporal support for {method!r}")
    window_seconds = float(match.group("window").replace("p", "."))
    window_samples = max(3, int(round(window_seconds * sampling_rate_hz)))
    if window_samples % 2 == 0:
        window_samples += 1
    return window_samples // 2 + int(
        math.ceil(gaussian_sigmas * smoothing_seconds * sampling_rate_hz)
    )


def purge_near_validation(
    training: np.ndarray,
    validation_intervals: list[list[int]],
    row_count: int,
    margin: int,
) -> np.ndarray:
    """Remove training rows whose target transform can see validation rows."""
    if margin <= 0:
        return training
    safe = np.ones(row_count, dtype=bool)
    for start, stop in validation_intervals:
        safe[max(0, start - margin) : min(row_count, stop + margin)] = False
    return training[safe[training]]


def target_path(prepared: Path, method: str) -> Path:
    if method == "raw_25hz":
        return prepared / "train_glove_25hz_raw.npy"
    return prepared / f"train_glove_{method}.npy"


def render_heatmap(
    path: Path,
    methods: list[str],
    scores: np.ndarray,
    selected: np.ndarray,
    subject: int,
) -> None:
    figure, axis = plt.subplots(figsize=(9, max(7, 0.34 * len(methods))))
    image = axis.imshow(scores, aspect="auto", cmap="viridis", vmin=0.0, vmax=max(0.8, float(np.max(scores))))
    axis.set_xticks(range(5), FINGER_NAMES)
    axis.set_yticks(range(len(methods)), methods)
    axis.set_title(f"Subject {subject}: event-OOF raw PCC by glove target")
    for finger in range(5):
        axis.scatter(finger, int(selected[finger]), marker="s", s=90, facecolors="none", edgecolors="white", linewidths=1.5)
    figure.colorbar(image, ax=axis, label="OOF raw-glove PCC")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", type=int, nargs="+", default=(1, 2, 3))
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--prepared-root", type=Path, default=Path("outputs/preprocessed_v2"))
    parser.add_argument("--feature-root", type=Path, default=Path("outputs/windowed_ica_wavelet_asymmetric_v1"))
    parser.add_argument("--fold-root", type=Path, default=Path("outputs/event_stratified_folds_v1"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/event_target_variant_ridge_v1"))
    parser.add_argument("--max-features", type=int, default=512)
    parser.add_argument("--alphas", type=float, nargs="+", default=(1.0e-3, 1.0e-2, 0.1, 1.0, 10.0, 100.0))
    parser.add_argument("--sampling-rate", type=float, default=25.0)
    parser.add_argument("--target-smoothing-seconds", type=float, default=0.16)
    parser.add_argument("--gaussian-safety-sigmas", type=float, default=5.0)
    args = parser.parse_args()

    for subject in args.subjects:
        prepared = args.prepared_root / f"sub{subject}"
        methods = [method for method in args.methods if target_path(prepared, method).exists()]
        raw_full = np.load(prepared / "train_glove_25hz_raw.npy")
        target_arrays = {
            method: np.load(target_path(prepared, method)) for method in methods
        }
        finite_supports = [
            support
            for method in methods
            if (
                support := target_support_bins(
                    method,
                    sampling_rate_hz=args.sampling_rate,
                    smoothing_seconds=args.target_smoothing_seconds,
                    gaussian_sigmas=args.gaussian_safety_sigmas,
                )
            )
            is not None
        ]
        maximum_support = max(finite_supports, default=0)
        feature_all = np.load(
            args.feature_root / f"sub{subject}" / "train_initialized_window_features.npy",
            mmap_mode="r",
        )
        report: dict[str, object] = {
            "subject": subject,
            "protocol": (
                "outer per-finger event folds; common raw-target outer-training feature screen; "
                "training-only RidgeCV; uniform maximum target-support purge"
            ),
            "selection_eligibility": (
                "raw and finite-support local targets only; the globally fitted paper rope "
                "baseline is reported for context but cannot be made leakage-safe by a finite purge"
            ),
            "official_final_validation_touched": False,
            "released_test_touched": False,
            "uniform_target_support_purge_bins": int(maximum_support),
            "methods": {},
            "per_finger_selection": {},
        }
        score_matrix = np.empty((len(methods), 5), dtype=np.float64)
        best_method_indices: list[int] = []
        for finger_index, finger in enumerate(FINGER_NAMES):
            definition = json.loads(
                (args.fold_root / f"sub{subject}" / finger / "folds.json").read_text()
            )
            rows = int(definition["training_rows"])
            raw = raw_full[24 : 24 + rows, finger_index]
            predictions = {
                method: np.full(rows, np.nan, dtype=np.float32) for method in methods
            }
            fold_records = {method: [] for method in methods}
            for fold, fold_definition in enumerate(definition["folds"]):
                training = indices_from_intervals(
                    fold_definition["training_intervals_after_purge"]
                )
                validation = indices_from_intervals(
                    fold_definition["validation_intervals"]
                )
                safe_training = purge_near_validation(
                    training,
                    fold_definition["validation_intervals"],
                    rows,
                    maximum_support,
                )
                # Screen once using only the raw outer-training target.  Every
                # target candidate therefore receives the same representation
                # and no candidate gets to tune its own feature subset.
                selection = correlation_order(
                    np.asarray(feature_all[safe_training]), raw[safe_training]
                )[: args.max_features]
                scaler = StandardScaler()
                x_train = scaler.fit_transform(
                    np.asarray(feature_all[safe_training][:, selection], dtype=np.float64)
                )
                x_validation = scaler.transform(
                    np.asarray(feature_all[validation][:, selection], dtype=np.float64)
                )
                for method in methods:
                    support = target_support_bins(
                        method,
                        sampling_rate_hz=args.sampling_rate,
                        smoothing_seconds=args.target_smoothing_seconds,
                        gaussian_sigmas=args.gaussian_safety_sigmas,
                    )
                    target = target_arrays[method][24 : 24 + rows, finger_index]
                    model = RidgeCV(alphas=np.asarray(args.alphas), fit_intercept=True)
                    model.fit(x_train, target[safe_training])
                    estimate = model.predict(x_validation).astype(np.float32)
                    predictions[method][validation] = estimate
                    fold_records[method].append(
                        {
                            "fold": fold,
                            "alpha": float(model.alpha_),
                            "raw_pcc": pearson(estimate, raw[validation]),
                            "target_support_bins": support,
                            "uniform_purge_bins": int(maximum_support),
                            "training_bins": int(safe_training.size),
                            "feature_count": int(selection.size),
                            "eligible_for_selection": support is not None,
                        }
                    )
            scores = {}
            for method_index, method in enumerate(methods):
                if not np.isfinite(predictions[method]).all():
                    raise RuntimeError(f"incomplete OOF target benchmark for S{subject} {finger} {method}")
                score = pearson(predictions[method], raw)
                score_matrix[method_index, finger_index] = score
                scores[method] = {
                    "oof_raw_pcc": score,
                    "folds": fold_records[method],
                }
            eligible = np.asarray(
                [
                    target_support_bins(
                        method,
                        sampling_rate_hz=args.sampling_rate,
                        smoothing_seconds=args.target_smoothing_seconds,
                        gaussian_sigmas=args.gaussian_safety_sigmas,
                    )
                    is not None
                    for method in methods
                ],
                dtype=bool,
            )
            eligible_indices = np.flatnonzero(eligible)
            best_index = int(
                eligible_indices[
                    np.argmax(score_matrix[eligible_indices, finger_index])
                ]
            )
            report["per_finger_selection"][finger] = {
                "selected_method": methods[best_index],
                "oof_raw_pcc": float(score_matrix[best_index, finger_index]),
            }
            best_method_indices.append(best_index)
            report["methods"][finger] = scores
        selected = np.asarray(best_method_indices, dtype=np.int64)
        report["selected_macro_five"] = float(
            np.mean(score_matrix[selected, np.arange(5)])
        )
        output = args.output_root / f"sub{subject}"
        output.mkdir(parents=True, exist_ok=True)
        render_heatmap(
            output / "target_variant_oof_heatmap.png",
            methods,
            score_matrix,
            selected,
            subject,
        )
        (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "subject": subject,
                    "selected_macro_five": report["selected_macro_five"],
                    "per_finger_selection": report["per_finger_selection"],
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
