#!/usr/bin/env python3
"""Fit validation-only nonnegative stacks of diverse prediction candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ecog_decoding.training import FINGER_NAMES, trajectory_metrics


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denominator = np.linalg.norm(x_centered) * np.linalg.norm(y_centered)
    return float(x_centered @ y_centered / denominator) if denominator > 0 else 0.0


def parse_candidate(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("candidate must be NAME=OUTPUT_DIRECTORY")
    return name, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, required=True)
    parser.add_argument(
        "--prepared-root", type=Path, default=Path("outputs/preprocessed_v2")
    )
    parser.add_argument("--history", type=int, default=25)
    parser.add_argument("--candidate", action="append", type=parse_candidate, required=True)
    parser.add_argument(
        "--base",
        required=True,
        help="candidate copied unchanged for fingers that are not stacked",
    )
    parser.add_argument(
        "--finger",
        action="append",
        choices=FINGER_NAMES,
        help="finger to stack; repeat as needed. Defaults to all fingers.",
    )
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=(1.0e-4, 1.0e-3, 1.0e-2, 0.1, 1.0, 10.0, 100.0, 1000.0),
    )
    parser.add_argument(
        "--clip-min",
        type=float,
        help="optional fixed lower support bound applied after stacking",
    )
    parser.add_argument(
        "--clip-max",
        type=float,
        help="optional fixed upper support bound applied after stacking",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    prepared = args.prepared_root / f"sub{args.subject}"
    metadata = json.loads((prepared / "metadata.json").read_text())
    split = int(metadata["target_fit_samples_25hz"])
    offset = args.history - 1
    validation_target = np.load(prepared / "train_glove_25hz_raw.npy")[split:]
    test_target = np.load(prepared / "test_glove_25hz_raw.npy")[offset:]

    values: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    candidate_paths: dict[str, str] = {}
    for name, path in args.candidate:
        validation = np.load(path / "validation_prediction.npy")
        test = np.load(path / "test_prediction.npy")
        if validation.shape != validation_target.shape or test.shape != test_target.shape:
            raise ValueError(
                f"{name} prediction shapes {validation.shape}, {test.shape} do not "
                f"match targets {validation_target.shape}, {test_target.shape}"
            )
        values[name] = (validation, test)
        candidate_paths[name] = str(path)
    if args.base not in values:
        raise ValueError(f"base candidate {args.base!r} was not supplied")

    requested = set(args.finger or FINGER_NAMES)
    validation_output = values[args.base][0].copy()
    test_output = values[args.base][1].copy()
    selection: dict[str, object] = {}
    splitter = TimeSeriesSplit(n_splits=args.folds)

    for finger, name in enumerate(FINGER_NAMES):
        if name not in requested:
            selection[name] = {"status": "base", "candidate": args.base}
            continue
        usable = [
            candidate
            for candidate, (validation, _) in values.items()
            if np.isfinite(validation[:, finger]).all()
            and np.std(validation[:, finger]) > 1.0e-8
        ]
        if not usable:
            raise ValueError(f"no finite, nonconstant candidate for {name}")
        validation_x = np.stack(
            [values[candidate][0][:, finger] for candidate in usable], axis=1
        )
        test_x = np.stack(
            [values[candidate][1][:, finger] for candidate in usable], axis=1
        )
        target = validation_target[:, finger]
        alpha_scores: dict[str, float] = {}
        for alpha in args.alphas:
            fold_scores = []
            for fit_indices, audit_indices in splitter.split(validation_x):
                model = make_pipeline(
                    StandardScaler(), Ridge(alpha=alpha, positive=True)
                )
                model.fit(validation_x[fit_indices], target[fit_indices])
                fold_scores.append(
                    pearson(
                        model.predict(validation_x[audit_indices]),
                        target[audit_indices],
                    )
                )
            alpha_scores[str(alpha)] = float(np.mean(fold_scores))
        best_alpha = max(args.alphas, key=lambda alpha: alpha_scores[str(alpha)])
        model = make_pipeline(
            StandardScaler(), Ridge(alpha=best_alpha, positive=True)
        )
        model.fit(validation_x, target)
        validation_estimate = model.predict(validation_x)
        test_estimate = model.predict(test_x)
        if args.clip_min is not None or args.clip_max is not None:
            lower = args.clip_min if args.clip_min is not None else -np.inf
            upper = args.clip_max if args.clip_max is not None else np.inf
            validation_estimate = np.clip(validation_estimate, lower, upper)
            test_estimate = np.clip(test_estimate, lower, upper)
        validation_output[:, finger] = validation_estimate
        test_output[:, finger] = test_estimate
        standardizer = model.named_steps["standardscaler"]
        ridge = model.named_steps["ridge"]
        selection[name] = {
            "status": "stacked",
            "candidates": usable,
            "selected_alpha": best_alpha,
            "blocked_cv_pcc": alpha_scores[str(best_alpha)],
            "alpha_scores": alpha_scores,
            "validation_fit_pcc": pearson(validation_output[:, finger], target),
            "standardized_coefficients": {
                candidate: float(weight)
                for candidate, weight in zip(usable, ridge.coef_)
            },
            "feature_mean": standardizer.mean_.tolist(),
            "feature_scale": standardizer.scale_.tolist(),
            "intercept": float(ridge.intercept_),
        }

    report = {
        "subject": args.subject,
        "method": (
            "nonnegative ridge stack fit on chronological validation predictions; "
            "alpha selected by blocked TimeSeriesSplit within validation"
        ),
        "base_candidate": args.base,
        "candidate_paths": candidate_paths,
        "output_support": {"minimum": args.clip_min, "maximum": args.clip_max},
        "selection": selection,
        "validation_raw_metrics": trajectory_metrics(
            validation_output, validation_target
        ),
        "test_raw_metrics": trajectory_metrics(test_output, test_target),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    np.save(
        args.output / "validation_prediction.npy",
        validation_output,
        allow_pickle=False,
    )
    np.save(args.output / "test_prediction.npy", test_output, allow_pickle=False)
    (args.output / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
