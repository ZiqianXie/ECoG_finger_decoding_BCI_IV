#!/usr/bin/env python3
"""Cross-validate a fixed selected feature set, then refit on all training data.

The feature indices must come from a training-only selection run.  Ridge
regularization is ranked across purged chronological blocks spanning the full
released training recording.  Test labels are loaded only after the winning
regularization has been selected and the final model has been fit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

try:
    from benchmark_ridge_target_variants import ridge_fit, ridge_predict
except ModuleNotFoundError:  # imported as scripts.crossvalidate_selected_ridge in tests
    from scripts.benchmark_ridge_target_variants import ridge_fit, ridge_predict
from ecog_decoding.training import FINGER_NAMES, trajectory_metrics


def pearson(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left = left - left.mean()
    right = right - right.mean()
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return float(left @ right / denominator) if denominator > 0 else 0.0


def purged_block_folds(
    sample_count: int, fold_count: int, purge: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    if fold_count < 2:
        raise ValueError("fold_count must be at least 2")
    if purge < 0:
        raise ValueError("purge must be non-negative")
    indices = np.arange(sample_count)
    result: list[tuple[np.ndarray, np.ndarray]] = []
    for validation in np.array_split(indices, fold_count):
        lower = max(0, int(validation[0]) - purge)
        upper = min(sample_count, int(validation[-1]) + purge + 1)
        keep = (indices < lower) | (indices >= upper)
        training = indices[keep]
        if training.size == 0 or validation.size == 0:
            raise ValueError("fold configuration leaves an empty partition")
        result.append((training, validation))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, required=True)
    parser.add_argument(
        "--prepared-root", type=Path, default=Path("outputs/preprocessed_v2")
    )
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--selection-root", type=Path, required=True)
    parser.add_argument(
        "--output-root", type=Path, default=Path("outputs/cv_selected_ridge_v1")
    )
    parser.add_argument("--target", required=True)
    parser.add_argument("--history", type=int, default=25)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--purge", type=int, default=None,
        help="bins excluded beside each validation block; defaults to history - 1",
    )
    parser.add_argument(
        "--alphas", type=float, nargs="+",
        default=(1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0),
    )
    parser.add_argument(
        "--stability-penalty", type=float, default=0.25,
        help="selection score is mean fold PCC minus this times fold SD",
    )
    parser.add_argument("--fingers", nargs="+", choices=FINGER_NAMES)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    prepared = args.prepared_root / f"sub{args.subject}"
    feature_root = args.feature_root / f"sub{args.subject}"
    selection_root = args.selection_root / f"sub{args.subject}"
    output = args.output_root / f"sub{args.subject}"
    output.mkdir(parents=True, exist_ok=True)
    offset = args.history - 1
    purge = offset if args.purge is None else args.purge

    train_features = np.load(
        feature_root / "train_initialized_window_features.npy", mmap_mode="r"
    )
    test_features = np.load(
        feature_root / "test_initialized_window_features.npy", mmap_mode="r"
    )
    train_target = np.load(prepared / f"train_glove_{args.target}.npy")[offset:]
    test_raw = np.load(prepared / "test_glove_25hz_raw.npy")[offset:]
    if train_features.shape[0] != train_target.shape[0]:
        raise ValueError(
            f"feature/target length mismatch: {train_features.shape[0]} vs "
            f"{train_target.shape[0]}"
        )

    selection = json.loads((selection_root / "summary.json").read_text())
    names = list(args.fingers or FINGER_NAMES)
    prediction = np.full_like(test_raw, np.nan, dtype=np.float32)
    report: dict[str, object] = {}
    folds = purged_block_folds(train_features.shape[0], args.folds, purge)
    device = torch.device(args.device)

    for name in names:
        finger = FINGER_NAMES.index(name)
        selected = np.asarray(
            selection["per_finger"][name]["selected_source_indices"], dtype=np.int64
        )
        x = np.asarray(train_features[:, selected])
        y = np.asarray(train_target[:, finger])
        alpha_reports: dict[str, object] = {}
        best_alpha = float(args.alphas[0])
        best_score = -float("inf")
        for alpha in args.alphas:
            fold_scores: list[float] = []
            for training, validation in folds:
                fit = ridge_fit(x[training], y[training], float(alpha), device)
                fold_scores.append(pearson(ridge_predict(x[validation], fit), y[validation]))
            mean = float(np.mean(fold_scores))
            standard_deviation = float(np.std(fold_scores))
            robust_score = mean - args.stability_penalty * standard_deviation
            alpha_reports[str(alpha)] = {
                "fold_pcc": fold_scores,
                "mean_pcc": mean,
                "std_pcc": standard_deviation,
                "robust_score": robust_score,
            }
            if robust_score > best_score:
                best_score = robust_score
                best_alpha = float(alpha)

        final_fit = ridge_fit(x, y, best_alpha, device)
        prediction[:, finger] = ridge_predict(
            np.asarray(test_features[:, selected]), final_fit
        )
        report[name] = {
            "selected_feature_count": int(selected.size),
            "best_alpha": best_alpha,
            "best_robust_cv_score": best_score,
            "alpha_reports": alpha_reports,
        }
        print(
            f"finger={name} features={selected.size} alpha={best_alpha:g} "
            f"robust_cv={best_score:.4f}", flush=True,
        )

    chosen = [FINGER_NAMES.index(name) for name in names]
    if len(chosen) == len(FINGER_NAMES):
        test_metrics = trajectory_metrics(prediction[:, chosen], test_raw[:, chosen])
    else:
        correlations = {
            name: pearson(prediction[:, finger], test_raw[:, finger])
            for name, finger in zip(names, chosen, strict=True)
        }
        test_metrics = {
            "pearson_by_finger": correlations,
            "pearson_macro_selected": float(np.mean(list(correlations.values()))),
            "rmse_selected": float(
                np.sqrt(np.mean((prediction[:, chosen] - test_raw[:, chosen]) ** 2))
            ),
        }
    np.save(output / "test_prediction.npy", prediction, allow_pickle=False)
    summary = {
        "subject": args.subject,
        "method": "purged blocked CV of training-selected features, then full-training ridge refit",
        "target": args.target,
        "history": args.history,
        "folds": args.folds,
        "purge_bins": purge,
        "stability_penalty": args.stability_penalty,
        "feature_root": str(args.feature_root),
        "selection_root": str(args.selection_root),
        "fit_fingers": names,
        "test_labels_used_for_selection": False,
        "per_finger": report,
        "test_raw_metrics": test_metrics,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(test_metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
