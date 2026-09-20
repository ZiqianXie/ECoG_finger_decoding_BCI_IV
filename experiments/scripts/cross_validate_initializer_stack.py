#!/usr/bin/env python3
"""Outer-split-safe stacking of independently fitted initializer predictions.

Each initializer must have been fitted inside the same outer-training split.
The stacker and its ridge strength are learned using only outer-training rows;
the held outer interval is evaluated exactly once. Released test labels are
never loaded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from audit_single_wavelet_residual_predictability import (
    direct_lars,
    finite_intervals,
    grouped_rows,
    pearson,
    rows_from_intervals,
)
from ecog_decoding.training import FINGER_NAMES


ALPHAS = tuple(float(value) for value in np.logspace(-3, 5, 9))
POWERS = (0.5, 1.0, 1.5, 2.0)


def parse_initializer(value: str) -> tuple[str, Path]:
    """Parse a NAME=PATH initializer specification."""
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("initializer must be NAME=PATH")
    return name, Path(path)


def prediction_features(predictions: list[np.ndarray]) -> np.ndarray:
    """Create a compact nonlinear stack from nonnegative base predictions."""
    columns = []
    for prediction in predictions:
        nonnegative = np.maximum(np.asarray(prediction, dtype=np.float32), 0.0)
        columns.extend(nonnegative**power for power in POWERS)
    return np.column_stack(columns).astype(np.float32, copy=False)


def fit_grouped_ridge(
    features: np.ndarray,
    target: np.ndarray,
    training_intervals: list[list[int]],
) -> tuple[object, float, dict[str, float]]:
    training, groups = grouped_rows(training_intervals)
    splitter = GroupKFold(n_splits=min(5, len(training_intervals)))
    splits = list(splitter.split(training, target[training], groups))
    scores: dict[float, list[float]] = {alpha: [] for alpha in ALPHAS}
    for train_local, validation_local in splits:
        train_rows = training[train_local]
        validation_rows = training[validation_local]
        for alpha in ALPHAS:
            model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
            model.fit(features[train_rows], target[train_rows])
            estimate = np.maximum(model.predict(features[validation_rows]), 0.0)
            scores[alpha].append(pearson(estimate, target[validation_rows]))
    means = {alpha: float(np.mean(values)) for alpha, values in scores.items()}
    selected = max(means, key=means.get)
    model = make_pipeline(StandardScaler(), Ridge(alpha=selected))
    model.fit(features[training], target[training])
    return model, selected, {str(alpha): score for alpha, score in means.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, required=True)
    parser.add_argument("--finger", choices=FINGER_NAMES, required=True)
    parser.add_argument(
        "--initializer",
        type=parse_initializer,
        action="append",
        required=True,
        help="repeat NAME=PATH; PATH contains cache/outerN/outer",
    )
    parser.add_argument(
        "--prepared-root", type=Path, default=Path("outputs/preprocessed_v2")
    )
    parser.add_argument("--outer-folds", type=int, nargs="+", default=(0, 1, 2))
    parser.add_argument("--softplus-beta", type=float, default=10.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    finger_index = list(FINGER_NAMES).index(args.finger)
    stitched_prediction = None
    stitched_raw = None
    stitched_cleaned = None
    outer_records = []
    used_initializers: set[str] = set()
    skipped_initializers: set[str] = set()
    for outer_fold in args.outer_folds:
        bases = []
        fold_names = []
        reference_split = None
        reference_target = None
        for name, root in args.initializer:
            directory = root / "cache" / f"outer{outer_fold}" / "outer"
            split = json.loads((directory / "split.json").read_text())
            target = np.load(directory / "target.npy")[:, finger_index]
            if reference_split is None:
                reference_split = split
                reference_target = target
            else:
                split_scope = {
                    key: split[key]
                    for key in ("training_intervals", "validation_intervals")
                }
                reference_scope = {
                    key: reference_split[key]
                    for key in ("training_intervals", "validation_intervals")
                }
                if split_scope != reference_scope:
                    raise ValueError(f"outer split mismatch for {name}")
                if not np.allclose(target, reference_target, equal_nan=True):
                    raise ValueError(f"cleaned target mismatch for {name}")
            with np.load(directory / "initialization.npz") as archive:
                initialization = {key: archive[key] for key in archive.files}
            if "candidate_features" not in initialization:
                skipped_initializers.add(name)
                continue
            base, _, _ = direct_lars(initialization, args.softplus_beta)
            bases.append(base)
            fold_names.append(name)
            used_initializers.add(name)

        if not bases:
            raise ValueError(f"no compatible initializers for outer fold {outer_fold}")

        assert reference_split is not None and reference_target is not None
        rows = reference_target.size
        raw = np.array(
            np.load(
                args.prepared_root
                / f"sub{args.subject}"
                / "train_glove_25hz_raw.npy",
                mmap_mode="r",
            )[24 : 24 + rows, finger_index],
            dtype=np.float32,
            copy=True,
        )
        training_intervals = finite_intervals(
            reference_split["training_intervals"], reference_target
        )
        validation_intervals = finite_intervals(
            reference_split["validation_intervals"], reference_target
        )
        validation = rows_from_intervals(validation_intervals)
        features = prediction_features(bases)
        model, alpha, curve = fit_grouped_ridge(
            features, raw, training_intervals
        )
        prediction = np.maximum(model.predict(features), 0.0).astype(np.float32)
        if stitched_prediction is None:
            stitched_prediction = np.full(rows, np.nan, dtype=np.float32)
            stitched_raw = raw
            stitched_cleaned = np.full(rows, np.nan, dtype=np.float32)
        stitched_prediction[validation] = prediction[validation]
        stitched_cleaned[validation] = reference_target[validation]
        outer_records.append(
            {
                "outer_fold": outer_fold,
                "initializers": fold_names,
                "selected_alpha": alpha,
                "inner_training_pcc_curve": curve,
                "validation_raw_pcc": pearson(
                    prediction[validation], raw[validation]
                ),
            }
        )

    assert stitched_prediction is not None
    assert stitched_raw is not None and stitched_cleaned is not None
    observed = np.isfinite(stitched_prediction)
    report = {
        "protocol": (
            "outer-split-safe ridge stack over split-local initializer predictions; "
            "ridge alpha selected by grouped folds within outer training; released "
            "test untouched"
        ),
        "released_test_touched": False,
        "subject": args.subject,
        "finger": args.finger,
        "requested_initializers": [name for name, _ in args.initializer],
        "used_initializers": sorted(used_initializers),
        "skipped_incompatible_initializers": sorted(skipped_initializers),
        "powers": list(POWERS),
        "outer_records": outer_records,
        "stitched_raw_pcc": pearson(
            stitched_prediction[observed], stitched_raw[observed]
        ),
        "stitched_cleaned_pcc": pearson(
            stitched_prediction[observed], stitched_cleaned[observed]
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "stack_oof.npy", stitched_prediction, allow_pickle=False)
    np.save(args.output / "target_oof.npy", stitched_cleaned, allow_pickle=False)
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
