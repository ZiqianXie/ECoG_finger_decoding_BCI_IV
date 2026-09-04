#!/usr/bin/env python3
"""Evaluate the paper's fixed-feature LassoLarsCV decoding checkpoint.

The spatial/temporal energy bank is loaded from a completed ICA or CSP run.
Only the chronological training portion is used to fit the model.  The
chronological validation portion ranks frontend/target variants; released test
labels are reported only after fitting and are never used for model selection.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LassoLarsCV
from sklearn.preprocessing import StandardScaler

from benchmark_ridge_target_variants import lagged
from ecog_decoding.training import FINGER_NAMES, trajectory_metrics


def energy_paths(root: Path, prewindowed: bool) -> tuple[Path, Path, str]:
    candidates = ()
    if prewindowed:
        candidates += (("train_initialized_window_features.npy", "test_initialized_window_features.npy", "ica_wavelet_exact_windows"),)
    candidates += (
        ("train_csp_energy.npy", "test_csp_energy.npy", "csp"),
        ("train_initialized_energy.npy", "test_initialized_energy.npy", "ica_wavelet"),
    )
    for train_name, test_name, frontend in candidates:
        if (root / train_name).exists() and (root / test_name).exists():
            return root / train_name, root / test_name, frontend
    raise FileNotFoundError(f"no supported fixed-energy arrays in {root}")


def correlation_order(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    xc = x - x.mean(axis=0, keepdims=True)
    yc = y - y.mean()
    denominator = np.sqrt(np.sum(xc * xc, axis=0) * np.sum(yc * yc))
    correlation = np.divide(
        xc.T @ yc,
        denominator,
        out=np.zeros(x.shape[1], dtype=np.float32),
        where=denominator > 0,
    )
    return np.argsort(np.abs(correlation))[::-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, required=True)
    parser.add_argument("--prepared-root", type=Path, default=Path("outputs/preprocessed_v2"))
    parser.add_argument("--energy-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/fixed_lars_v1"))
    parser.add_argument("--target", required=True)
    parser.add_argument("--history", type=int, default=25)
    parser.add_argument("--cv", type=int, default=3)
    parser.add_argument("--max-iter", type=int, default=500)
    parser.add_argument("--max-features", type=int, default=0,
                        help="0 uses the complete fixed bank; positive values are a diagnostic prescreen")
    parser.add_argument("--fingers", nargs="+", choices=tuple(FINGER_NAMES))
    parser.add_argument("--prewindowed", action="store_true")
    args = parser.parse_args()

    prepared = args.prepared_root / f"sub{args.subject}"
    energy_root = args.energy_root / f"sub{args.subject}"
    output = args.output_root / f"sub{args.subject}"
    output.mkdir(parents=True, exist_ok=True)
    train_path, test_path, frontend = energy_paths(energy_root, args.prewindowed)
    train_energy = np.load(train_path)
    test_energy = np.load(test_path)
    metadata = json.loads((prepared / "metadata.json").read_text())
    split = int(metadata["target_fit_samples_25hz"])
    offset = args.history - 1
    train_x_all = train_energy if args.prewindowed else lagged(train_energy, args.history)
    test_x = test_energy if args.prewindowed else lagged(test_energy, args.history)
    train_x = train_x_all[: split - offset]
    validation_x = train_x_all[split - offset :]
    target = np.load(prepared / f"train_glove_{args.target}.npy")
    train_y = target[offset:split]
    validation_target = target[split:]
    validation_raw = np.load(prepared / "train_glove_25hz_raw.npy")[split:]
    test_raw = np.load(prepared / "test_glove_25hz_raw.npy")[offset:]
    validation_prediction = np.full_like(validation_raw, np.nan, dtype=np.float32)
    test_prediction = np.full_like(test_raw, np.nan, dtype=np.float32)
    names = list(args.fingers or FINGER_NAMES)
    report: dict[str, object] = {}

    for name in names:
        finger = list(FINGER_NAMES).index(name)
        started = time.monotonic()
        if args.max_features > 0 and args.max_features < train_x.shape[1]:
            selected = correlation_order(train_x, train_y[:, finger])[: args.max_features]
        else:
            selected = np.arange(train_x.shape[1])
        scaler = StandardScaler(copy=True)
        x_fit = scaler.fit_transform(train_x[:, selected]).astype(np.float64, copy=False)
        model = LassoLarsCV(
            cv=args.cv,
            fit_intercept=True,
            max_iter=args.max_iter,
            n_jobs=-1,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(x_fit, train_y[:, finger])
        validation_prediction[:, finger] = model.predict(
            scaler.transform(validation_x[:, selected])
        )
        test_prediction[:, finger] = model.predict(scaler.transform(test_x[:, selected]))
        nonzero = np.flatnonzero(model.coef_)
        report[name] = {
            "alpha": float(model.alpha_),
            "input_feature_count": int(selected.size),
            "nonzero_feature_count": int(nonzero.size),
            "selected_source_indices": selected[nonzero].tolist(),
            "iterations": int(model.n_iter_),
            "elapsed_seconds": time.monotonic() - started,
            "warnings": [str(item.message) for item in caught],
        }
        print(
            f"finger={name} alpha={model.alpha_:.7g} nonzero={nonzero.size} "
            f"seconds={report[name]['elapsed_seconds']:.1f}",
            flush=True,
        )

    chosen = [list(FINGER_NAMES).index(name) for name in names]
    validation_target_metrics = trajectory_metrics(
        validation_prediction[:, chosen], validation_target[:, chosen]
    )
    validation_raw_metrics = trajectory_metrics(
        validation_prediction[:, chosen], validation_raw[:, chosen]
    )
    test_raw_metrics = trajectory_metrics(test_prediction[:, chosen], test_raw[:, chosen])
    np.save(output / "validation_prediction.npy", validation_prediction, allow_pickle=False)
    np.save(output / "test_prediction.npy", test_prediction, allow_pickle=False)
    result = {
        "subject": args.subject,
        "method": "fixed frontend plus LassoLarsCV with standardized columns",
        "frontend": frontend,
        "source_energy_root": str(args.energy_root),
        "target": args.target,
        "history": args.history,
        "cv": args.cv,
        "max_iter": args.max_iter,
        "diagnostic_max_features": args.max_features,
        "fit_fingers": names,
        "test_labels_used_for_selection": False,
        "per_finger": report,
        "validation_target_metrics": validation_target_metrics,
        "validation_raw_metrics": validation_raw_metrics,
        "test_raw_metrics": test_raw_metrics,
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"validation_raw": validation_raw_metrics, "test_raw": test_raw_metrics}), flush=True)


if __name__ == "__main__":
    main()
