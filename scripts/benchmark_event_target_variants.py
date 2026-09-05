#!/usr/bin/env python3
"""Compare glove target variants with fixed features under outer event folds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from ecog_decoding.training import FINGER_NAMES
from train_event_grouped_lars_lstm import indices_from_intervals, pearson


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


def target_path(prepared: Path, method: str) -> Path:
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
    parser.add_argument("--selection-cache-root", type=Path, default=Path("outputs/event_lars_selection_v1"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/event_target_variant_ridge_v1"))
    parser.add_argument("--alphas", type=float, nargs="+", default=(1.0e-3, 1.0e-2, 0.1, 1.0, 10.0, 100.0))
    args = parser.parse_args()

    for subject in args.subjects:
        prepared = args.prepared_root / f"sub{subject}"
        methods = [method for method in args.methods if target_path(prepared, method).exists()]
        raw_full = np.load(prepared / "train_glove_25hz_raw.npy")
        target_arrays = {
            method: np.load(target_path(prepared, method)) for method in methods
        }
        feature_all = np.load(
            args.feature_root / f"sub{subject}" / "train_initialized_window_features.npy",
            mmap_mode="r",
        )
        report: dict[str, object] = {
            "subject": subject,
            "protocol": "outer per-finger event folds; fixed outer-training-selected wavelet features; training-only RidgeCV",
            "official_final_validation_touched": False,
            "released_test_touched": False,
            "methods": {},
            "per_finger_selection": {},
        }
        score_matrix = np.empty((len(methods), 5), dtype=np.float64)
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
                selection = np.load(
                    args.selection_cache_root / f"sub{subject}" / finger / f"fold{fold}.npz"
                )["selected_source"]
                x_train = np.asarray(feature_all[training][:, selection], dtype=np.float64)
                x_validation = np.asarray(feature_all[validation][:, selection], dtype=np.float64)
                scaler = StandardScaler()
                x_train = scaler.fit_transform(x_train)
                x_validation = scaler.transform(x_validation)
                for method in methods:
                    target = target_arrays[method][24 : 24 + rows, finger_index]
                    model = RidgeCV(alphas=np.asarray(args.alphas), fit_intercept=True)
                    model.fit(x_train, target[training])
                    estimate = model.predict(x_validation).astype(np.float32)
                    predictions[method][validation] = estimate
                    fold_records[method].append(
                        {
                            "fold": fold,
                            "alpha": float(model.alpha_),
                            "raw_pcc": pearson(estimate, raw[validation]),
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
            best_index = int(np.argmax(score_matrix[:, finger_index]))
            report["per_finger_selection"][finger] = {
                "selected_method": methods[best_index],
                "oof_raw_pcc": float(score_matrix[best_index, finger_index]),
            }
            report["methods"][finger] = scores
        selected = np.argmax(score_matrix, axis=0)
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
