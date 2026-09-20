#!/usr/bin/env python3
"""Refit a development-selected full-bank ridge canonical decoder."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from audit_single_wavelet_residual_predictability import direct_lars, softplus
from cross_validate_single_wavelet import make_model, make_subject_target
from cross_validate_single_wavelet import lagged_with_future_context
from gpu_ridge import ridge_path_eigendecomposition
from refit_canonical_recurrent import (
    load_ecog,
    load_selection,
    make_frontend,
    settings_namespace,
)
from refit_single_wavelet import fit_training_affine
from single_wavelet_support import HISTORY, OFFSET, extract_energy, pearson
from ecog_decoding.training import FINGER_NAMES


def unanimous_outer_value(
    report: dict[str, object], key: str, *, nested: tuple[str, ...] = ()
) -> float:
    values = []
    for record in report["outer_records"]:
        value = record
        for component in nested:
            value = value[component]
        values.append(float(value[key]))
    unique = {round(value, 12) for value in values}
    if len(unique) != 1:
        raise ValueError(f"outer folds do not agree on {'.'.join((*nested, key))}: {values}")
    return values[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-summary", type=Path, required=True)
    parser.add_argument("--initialization-refit", type=Path, required=True)
    parser.add_argument("--development-summary", type=Path, required=True)
    parser.add_argument(
        "--probe", choices=("ridge_all_lags", "ridge_all_lags_raw"), required=True
    )
    parser.add_argument("--raw-power-calibration", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args_cli = parser.parse_args()

    started = time.perf_counter()
    selection = load_selection(args_cli.selection_summary)
    args = settings_namespace(selection, seed=2026, device=args_cli.device, compile_model=False)
    device = torch.device(args.device)
    finger_index = list(FINGER_NAMES).index(args.finger)
    with np.load(args_cli.initialization_refit / "initialization.npz") as archive:
        initialization = {name: archive[name] for name in archive.files}
    spatial = np.load(args_cli.initialization_refit / "spatial.npy")
    model = make_model(spatial, initialization, args).to(device).eval()

    prepared = args.prepared_root / f"sub{args.subject}"
    train_raw_full = np.load(prepared / "train_glove_25hz_raw.npy", mmap_mode="r")
    test_raw_full = np.load(prepared / "test_glove_25hz_raw.npy", mmap_mode="r")
    train_rows = int(train_raw_full.shape[0] - OFFSET)
    test_rows = int(test_raw_full.shape[0] - OFFSET)
    train_raw = np.asarray(train_raw_full[OFFSET : OFFSET + train_rows], dtype=np.float32)
    test_raw = np.asarray(test_raw_full[OFFSET : OFFSET + test_rows], dtype=np.float32)
    cleaned = make_subject_target(
        train_raw,
        [[0, train_rows]],
        [],
        args.subject,
        finger_index=finger_index,
        little_event_decontamination=args.little_event_decontamination,
        little_event_ratio_low=args.little_event_ratio_low,
        little_event_ratio_high=args.little_event_ratio_high,
    )

    base_train, standardized_train, _ = direct_lars(
        initialization, args.softplus_beta
    )
    test_ecog = load_ecog(prepared / "test_ecog.npy", args.model_rate)
    test_frontend, _ = make_frontend(args, device)
    test_frontend.load_state_dict(model.wavelet.state_dict())
    test_energy = extract_energy(
        test_ecog,
        model.spatial.weight[:, :, 0].detach().cpu().numpy(),
        test_frontend,
        device,
        args.component_chunk,
    )
    test_stream = test_energy.reshape(test_energy.shape[0], -1)
    test_lagged = lagged_with_future_context(
        np.asarray(test_stream, dtype=np.float32), HISTORY, args.future_context_bins
    )
    candidate_test_np = np.asarray(
        test_lagged[:, initialization["candidate_indices"].astype(np.int64)],
        dtype=np.float32,
    )
    standardized_test = (
        candidate_test_np - initialization["candidate_feature_mean"]
    ) / initialization["candidate_feature_scale"]
    selected_positions = initialization["selected_candidate_positions"].astype(np.int64)
    base_test = softplus(
        standardized_test[:, selected_positions] @ initialization["coefficients"]
        + float(initialization["intercept"]),
        args.softplus_beta,
    )
    if base_test.size != test_rows:
        raise RuntimeError(f"test feature rows {base_test.size} != target rows {test_rows}")

    development = json.loads(args_cli.development_summary.read_text())
    alpha_key = (
        "ridge_all_alpha"
        if args_cli.probe == "ridge_all_lags"
        else "ridge_all_raw_alpha"
    )
    alpha = unanimous_outer_value(development, alpha_key)
    fit_target = (
        cleaned[:, finger_index]
        if args_cli.probe == "ridge_all_lags"
        else train_raw[:, finger_index]
    )
    coefficients, intercepts = ridge_path_eigendecomposition(
        standardized_train,
        fit_target,
        np.asarray([alpha], dtype=np.float32),
        device=device,
    )
    prediction_train = np.maximum(
        standardized_train @ coefficients[0] + intercepts[0], 0.0
    )
    prediction_test = np.maximum(
        standardized_test @ coefficients[0] + intercepts[0], 0.0
    )
    base_power = None
    selected_power = None
    if args_cli.raw_power_calibration:
        base_power = unanimous_outer_value(
            development,
            "selected_power",
            nested=("raw_power_calibration", "lars"),
        )
        selected_power = unanimous_outer_value(
            development,
            "selected_power",
            nested=("raw_power_calibration", args_cli.probe),
        )
        base_train = np.maximum(base_train, 0.0) ** base_power
        base_test = np.maximum(base_test, 0.0) ** base_power
        prediction_train = prediction_train**selected_power
        prediction_test = prediction_test**selected_power

    slope, intercept = fit_training_affine(
        prediction_train, train_raw[:, finger_index]
    )
    calibrated_test = (slope * prediction_test + intercept).astype(np.float32)
    args_cli.output.mkdir(parents=True, exist_ok=True)
    np.savez(
        args_cli.output / "ridge_model.npz",
        coefficients=coefficients[0].astype(np.float32),
        intercept=np.asarray(intercepts[0], dtype=np.float32),
        alpha=np.asarray(alpha, dtype=np.float32),
        raw_power=np.asarray(
            np.nan if selected_power is None else selected_power, dtype=np.float32
        ),
    )
    np.savez(args_cli.output / "initialization.npz", **initialization)
    np.save(args_cli.output / "spatial.npy", spatial, allow_pickle=False)
    np.save(args_cli.output / "test_prediction.npy", prediction_test, allow_pickle=False)
    np.save(
        args_cli.output / "test_prediction_raw_scale.npy",
        calibrated_test,
        allow_pickle=False,
    )
    np.save(
        args_cli.output / "test_raw_target.npy",
        test_raw[:, finger_index],
        allow_pickle=False,
    )
    report = {
        "protocol": "full-development refit of one nested-selected full-bank ridge model",
        "subject": int(args.subject),
        "finger": str(args.finger),
        "structure_id": (
            "future16_fullbank_powercal_ridge"
            if args_cli.raw_power_calibration
            else "future16_fullbank_raw_ridge"
        ),
        "selection_summary": str(args_cli.selection_summary),
        "development_summary": str(args_cli.development_summary),
        "probe": args_cli.probe,
        "alpha_frozen_from_nested_development": alpha,
        "base_power_frozen_from_nested_development": base_power,
        "selected_power_frozen_from_nested_development": selected_power,
        "development_initialization_pcc": pearson(
            base_train, train_raw[:, finger_index]
        ),
        "development_refit_pcc": pearson(
            prediction_train, train_raw[:, finger_index]
        ),
        "released_test_used_for_selection": False,
        "released_test_raw_pcc_descriptive_only": pearson(
            calibrated_test, test_raw[:, finger_index]
        ),
        "released_test_rmse_descriptive_only": float(
            np.sqrt(np.mean((calibrated_test - test_raw[:, finger_index]) ** 2))
        ),
        "output_calibration": {
            "fit_on": "full labeled development recording only",
            "slope": slope,
            "intercept": intercept,
            "released_test_labels_used": False,
        },
        "runtime_seconds": time.perf_counter() - started,
    }
    (args_cli.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
