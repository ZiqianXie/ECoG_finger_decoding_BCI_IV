#!/usr/bin/env python3
"""Refit one nested-selected recurrent decoder on all development data.

The selection summary is authoritative for architecture and optimization
settings.  Released-test labels are loaded only after every development-side
choice (candidate family and update schedule) is fixed.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from cross_validate_single_wavelet import (
    CSP_BANDS_HZ,
    CSP_MODES,
    TARGET_POLICIES,
    cached_initialization_features,
    fit_initialization,
    lagged_with_future_context,
    linear_gamma_leaf_signals,
    linear_lower_high_gamma_leaf_signals,
    make_model,
    make_subject_target,
    scoped_event_groups,
    split_local_initialization_target_blend,
    train_final_schedule,
    valid_leaf_cache,
)
from ecog_decoding.training import FINGER_NAMES
from refit_single_wavelet import fit_training_affine, select_schedule_from_report
from single_wavelet_model import OvercompleteWaveletPacketEnergy
from single_wavelet_support import (
    HISTORY,
    OFFSET,
    WaveletPacketEnergy,
    extract_energy,
    pearson,
)


PATH_SETTING_KEYS = {
    "prepared_root",
    "fold_root",
    "resampled_cache",
    "leaf_cache",
    "output",
    "initialization_cache_root",
    "reuse_inner_metrics_from",
    "csp_band_cache_root",
    "ica_cache_root",
}

BACKWARD_COMPATIBLE_DEFAULTS = {
    "wavelet_frontend": "depth3",
    "wavelet_interlevel_skip": False,
    "wavelet_interlevel_normalization": False,
    "wavelet_final_normalization": True,
    "wavelet_signed_pooling": False,
    "signed_pooling_learning_rate": 3.0e-4,
    "interlevel_learning_rate": 3.0e-4,
    "lasso_backend": "torch_fista",
    "csp_band_mode": "joint_hhl_hhh",
    "csp_contrast_mode": "common_rest",
    "csp_band_cache_root": "/dev/shm/ecog_csp_band_cache",
    "amplitude_target_lead_bins": 0,
    "future_context_bins": 0,
    "residual_dynamics": "pointwise",
    "residual_decay": 0.95,
    "movement_head_scope": "target",
    "movement_head_objective": "binary_state",
    "auxiliary_residual_readout_l2": None,
    "raw_trajectory_blend": 0.0,
    "initialization_raw_target_blend": 0.0,
    "spatial_anchor_weight": 0.0,
    "spoc_auxiliary_weight": 0.0,
    "spoc_active_threshold": 0.20,
    "spatial_orthogonality_weight": 0.0,
    "raw_movement_correlation_weight": 0.0,
    "raw_movement_derivative_correlation_weight": 0.0,
    "warmup_steps": 0,
    "split_seed": None,
    "sampler_seed": None,
}


def load_selection(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text())
    required = {"subject", "finger", "settings", "outer_folds", "schedule_candidates"}
    missing = sorted(required - set(report))
    if missing:
        raise ValueError(f"selection summary is missing {missing}")
    return report


def settings_namespace(
    report: dict[str, Any], *, seed: int, device: str, compile_model: bool
) -> SimpleNamespace:
    settings = dict(report["settings"])
    for key, value in BACKWARD_COMPATIBLE_DEFAULTS.items():
        settings.setdefault(key, value)
    for key in PATH_SETTING_KEYS:
        if key in settings and settings[key] is not None:
            settings[key] = Path(settings[key])
    settings["seed"] = seed
    settings["device"] = device
    settings["compile"] = compile_model
    settings["samples_per_bin"] = int(settings["model_rate"]) // 25
    return SimpleNamespace(**settings)


def pooled_schedule(report: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    inner_records = [
        inner
        for outer in report["outer_folds"]
        for inner in outer.get("inner_records", [])
    ]
    if not inner_records:
        raise ValueError("selection summary has no inner-fold records")
    settings = report["settings"]
    return select_schedule_from_report(
        report,
        inner_records,
        metric=str(settings["selection_metric"]),
        rule=str(settings["selection_rule"]),
    )


def make_frontend(args: SimpleNamespace, device: torch.device):
    levels = {
        "depth3": 3,
        "overcomplete_depth3_depth4": 4,
        "overcomplete_depth3_depth4_depth5": 5,
    }[args.wavelet_frontend]
    frontend_type = OvercompleteWaveletPacketEnergy if levels > 3 else WaveletPacketEnergy
    frontend = frontend_type(
        wavelet="bior6.8",
        levels=levels,
        kernel_size=17,
        trainable=False,
        padding_mode="constant",
        energy_window_samples=args.samples_per_bin,
        energy_stride_samples=args.samples_per_bin,
        tap_resample_up=args.tap_resample_up,
        tap_resample_down=args.tap_resample_down,
    ).to(device).eval()
    csp_frontend = (
        frontend
        if args.wavelet_frontend == "depth3"
        else WaveletPacketEnergy(
            wavelet="bior6.8",
            levels=3,
            kernel_size=17,
            trainable=False,
            padding_mode="constant",
            energy_window_samples=args.samples_per_bin,
            energy_stride_samples=args.samples_per_bin,
            tap_resample_up=args.tap_resample_up,
            tap_resample_down=args.tap_resample_down,
        ).to(device).eval()
    )
    return frontend, csp_frontend


def load_ecog(path: Path, model_rate: int) -> np.ndarray:
    if model_rate != 1000:
        raise ValueError("canonical recurrent refits currently require 1000 Hz inputs")
    return np.asarray(np.load(path, mmap_mode="r"))


def gamma_bins(
    ecog: np.ndarray,
    csp_frontend,
    leaf_cache: Path,
    samples_per_bin: int,
    device: torch.device,
    csp_band_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    leaf_cache.mkdir(parents=True, exist_ok=True)
    hhl_path = leaf_cache / "linear_hhl_100_125.npy"
    hhh_path = leaf_cache / "linear_hhh_125_150.npy"
    if not (
        valid_leaf_cache(hhl_path, ecog.shape[0])
        and valid_leaf_cache(hhh_path, ecog.shape[0])
    ):
        hhl_values, hhh_values = linear_gamma_leaf_signals(ecog, csp_frontend, device)
        np.save(hhl_path, hhl_values, allow_pickle=False)
        np.save(hhh_path, hhh_values, allow_pickle=False)
    usable = ecog.shape[0] // samples_per_bin * samples_per_bin
    hhl = np.load(hhl_path, mmap_mode="r")[:usable].reshape(
        -1, samples_per_bin, ecog.shape[1]
    )
    hhh = np.load(hhh_path, mmap_mode="r")[:usable].reshape(
        -1, samples_per_bin, ecog.shape[1]
    )
    lower = None
    if csp_band_mode == "separate_50_100_hhl_hhh":
        low1_path = leaf_cache / "linear_50_75.npy"
        low2_path = leaf_cache / "linear_75_100.npy"
        if not (
            valid_leaf_cache(low1_path, ecog.shape[0])
            and valid_leaf_cache(low2_path, ecog.shape[0])
        ):
            low1, low2 = linear_lower_high_gamma_leaf_signals(
                ecog, csp_frontend, device
            )
            np.save(low1_path, low1, allow_pickle=False)
            np.save(low2_path, low2, allow_pickle=False)
        low1 = np.load(low1_path, mmap_mode="r")[:usable].reshape(
            -1, samples_per_bin, ecog.shape[1]
        )
        low2 = np.load(low2_path, mmap_mode="r")[:usable].reshape(
            -1, samples_per_bin, ecog.shape[1]
        )
        lower = np.concatenate((low1, low2), axis=1)
    return hhl, hhh, lower


def designed_bins(args: SimpleNamespace, ecog: np.ndarray) -> np.ndarray | None:
    if args.csp_band_mode != "designed_seven":
        return None
    path = args.csp_band_cache_root / f"sub{args.subject}" / "train_filtered_bands.npy"
    values = np.load(path, mmap_mode="r")
    if values.ndim != 3 or values.shape[0] != len(CSP_BANDS_HZ):
        raise ValueError("designed-band cache must contain seven bands")
    usable = values.shape[1] // args.samples_per_bin * args.samples_per_bin
    return values[:, :usable].reshape(
        len(CSP_BANDS_HZ), -1, args.samples_per_bin, ecog.shape[1]
    )


def cached_test_features(
    model,
    ecog: np.ndarray,
    frontend,
    args: SimpleNamespace,
    device: torch.device,
) -> torch.Tensor:
    energy = extract_energy(ecog, model.spatial.weight[:, :, 0].detach().cpu().numpy(), frontend, device, args.component_chunk)
    stream = energy.reshape(energy.shape[0], -1)
    features = lagged_with_future_context(
        np.asarray(stream, dtype=np.float32), HISTORY, args.future_context_bins
    )
    if model.residual_input in ("causal_candidate", "selected_causal"):
        raise ValueError("causal residual inputs are not used by the canonical refits")
    indices = (
        model.candidate_indices.detach().cpu().numpy()
        if model.residual_input in ("candidate", "current_candidate")
        else model.selected_indices.detach().cpu().numpy()
    )
    return torch.from_numpy(np.asarray(features[:, indices], dtype=np.float32)).to(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--force-schedule", default=None)
    parser.add_argument(
        "--include-candidate-pool",
        action="store_true",
        help="Persist the complete lagged candidate bank for downstream ridge refits.",
    )
    parser.add_argument("--csp-band-cache-root", type=Path, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    args_cli = parser.parse_args()

    started = time.perf_counter()
    selection = load_selection(args_cli.selection_summary)
    args = settings_namespace(
        selection,
        seed=args_cli.seed,
        device=args_cli.device,
        compile_model=args_cli.compile,
    )
    if args_cli.csp_band_cache_root is not None:
        args.csp_band_cache_root = args_cli.csp_band_cache_root
    if not bool(args.frozen_only):
        raise ValueError("canonical refitter currently supports frozen-stem selections")
    schedule, schedule_audit = pooled_schedule(selection)
    if args_cli.force_schedule is not None:
        schedule = args_cli.force_schedule
        schedule_audit = {
            "forced_schedule": schedule,
            "development_basis": str(args_cli.selection_summary),
        }
    if not schedule.startswith("frozen_") or schedule == "frozen_0":
        raise ValueError(f"canonical tuning requires a positive-update frozen schedule, got {schedule}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)
    finger_index = list(FINGER_NAMES).index(args.finger)
    prepared = args.prepared_root / f"sub{args.subject}"
    train_ecog = load_ecog(prepared / "train_ecog.npy", args.model_rate)
    test_ecog = load_ecog(prepared / "test_ecog.npy", args.model_rate)
    train_raw_full = np.load(prepared / "train_glove_25hz_raw.npy", mmap_mode="r")
    test_raw_full = np.load(prepared / "test_glove_25hz_raw.npy", mmap_mode="r")
    train_rows = int(train_raw_full.shape[0] - OFFSET)
    test_rows = int(test_raw_full.shape[0] - OFFSET)
    train_raw = np.asarray(train_raw_full[OFFSET : OFFSET + train_rows], dtype=np.float64)
    test_raw = np.asarray(test_raw_full[OFFSET : OFFSET + test_rows], dtype=np.float64)
    target = make_subject_target(
        train_raw,
        [[0, train_rows]],
        [],
        args.subject,
        finger_index=finger_index,
        little_event_decontamination=args.little_event_decontamination,
        little_event_ratio_low=args.little_event_ratio_low,
        little_event_ratio_high=args.little_event_ratio_high,
    )
    groups, _, _ = scoped_event_groups(
        target,
        [[0, train_rows]],
        args.threshold,
        args.merge_gap_bins,
        args.minimum_event_bins,
        args.maximum_rest_group_bins,
        finger_index,
    )
    training_groups = [[int(start), int(stop)] for start, stop in groups]

    frontend, csp_frontend = make_frontend(args, device)
    leaf_cache = Path(args.leaf_cache)
    hhl, hhh, lower = gamma_bins(
        train_ecog,
        csp_frontend,
        leaf_cache,
        args.samples_per_bin,
        device,
        args.csp_band_mode,
    )
    joint = np.concatenate((hhl, hhh), axis=1)
    initialization_target, target_blend_audit = split_local_initialization_target_blend(
        target,
        train_raw,
        np.arange(train_rows, dtype=np.int64),
        args.initialization_raw_target_blend,
    )
    initialization, spatial, initialization_audit = fit_initialization(
        ecog=train_ecog,
        joint_bins=joint,
        target=initialization_target,
        training_intervals=[[0, train_rows]],
        training_groups=training_groups,
        frontend=frontend,
        device=device,
        ica_prescreen=args.ica_prescreen,
        component_chunk=args.component_chunk,
        finger_index=finger_index,
        csp_mode=args.csp_mode,
        csp_band_mode=args.csp_band_mode,
        csp_contrast_mode=args.csp_contrast_mode,
        amplitude_target_lead_bins=args.amplitude_target_lead_bins,
        hhl_bins=hhl,
        hhh_bins=hhh,
        lower_high_gamma_bins=lower,
        designed_band_bins=designed_bins(args, train_ecog),
        ica_weights=None,
        samples_per_bin=args.samples_per_bin,
        future_context_bins=args.future_context_bins,
        lasso_backend=args.lasso_backend,
        include_candidate_pool=(
            args_cli.include_candidate_pool
            or args.residual_input
            in ("candidate", "current_candidate", "causal_candidate", "selected_causal")
        ),
    )
    initialization_audit["initialization_raw_target_blend"] = float(
        args.initialization_raw_target_blend
    )
    initialization_audit["initialization_raw_target_affine"] = target_blend_audit

    torch.manual_seed(args.seed)
    model = make_model(spatial, initialization, args).to(device)
    cached_train = torch.from_numpy(
        cached_initialization_features(initialization, args.residual_input)
    ).to(device)
    target_tensor = torch.from_numpy(target.astype(np.float32)).to(device)
    with torch.inference_mode():
        initialized_train = model.direct_features(cached_train).float().cpu().numpy()
    train_final_schedule(
        model=model,
        cached=cached_train,
        padded_ecog=torch.empty(0, device=device),
        target=target_tensor,
        target_np=target,
        raw=train_raw[:, finger_index].astype(np.float32),
        training_groups=training_groups,
        schedule=schedule,
        args=args,
        seed=int(args.sampler_seed if args.sampler_seed is not None else args.seed),
        finger_index=finger_index,
        spoc_filtered_bins=joint,
    )
    with torch.inference_mode():
        selected_train = model.decode_features(cached_train[None])[0].float().cpu().numpy()
    test_frontend, _ = make_frontend(args, device)
    test_frontend.load_state_dict(model.wavelet.state_dict())
    cached_test = cached_test_features(model, test_ecog, test_frontend, args, device)
    with torch.inference_mode():
        test_prediction = model.decode_features(cached_test[None])[0].float().cpu().numpy()
    if test_prediction.size != test_rows:
        raise RuntimeError(
            f"test prediction has {test_prediction.size} rows, expected {test_rows}"
        )
    slope, intercept = fit_training_affine(
        selected_train, train_raw[:, finger_index]
    )
    calibrated_test = (slope * test_prediction + intercept).astype(np.float32)

    args_cli.output.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args_cli.output / "model.pt")
    np.savez(args_cli.output / "initialization.npz", **initialization)
    np.save(args_cli.output / "spatial.npy", spatial, allow_pickle=False)
    np.save(args_cli.output / "initialized_train_prediction.npy", initialized_train, allow_pickle=False)
    np.save(args_cli.output / "selected_train_prediction.npy", selected_train, allow_pickle=False)
    np.save(args_cli.output / "test_prediction.npy", test_prediction, allow_pickle=False)
    np.save(args_cli.output / "test_prediction_raw_scale.npy", calibrated_test, allow_pickle=False)
    np.save(args_cli.output / "test_raw_target.npy", test_raw[:, finger_index].astype(np.float32), allow_pickle=False)
    report = {
        "protocol": "full-development refit of one frozen-stem recurrent family selected by nested development OOF",
        "subject": int(args.subject),
        "finger": str(args.finger),
        "selection_summary": str(args_cli.selection_summary),
        "seed": int(args.seed),
        "structure_settings": {
            key: value
            for key, value in selection["settings"].items()
            if key
            not in {
                "output",
                "initialization_cache_root",
                "reuse_inner_metrics_from",
                "outer_folds",
                "seed",
                "device",
            }
        },
        "selected_schedule": schedule,
        "schedule_selection": schedule_audit,
        "development_initialization_pcc": pearson(
            initialized_train, train_raw[:, finger_index]
        ),
        "development_refit_pcc": pearson(
            selected_train, train_raw[:, finger_index]
        ),
        "released_test_used_for_selection": False,
        "released_test_raw_pcc_descriptive_only": pearson(
            calibrated_test, test_raw[:, finger_index]
        ),
        "released_test_rmse_descriptive_only": float(
            np.sqrt(
                np.mean(
                    (calibrated_test - test_raw[:, finger_index].astype(np.float32))
                    ** 2
                )
            )
        ),
        "output_calibration": {
            "fit_on": "full labeled development recording only",
            "slope": slope,
            "intercept": intercept,
            "released_test_labels_used": False,
        },
        "initialization_audit": initialization_audit,
        "runtime_seconds": time.perf_counter() - started,
    }
    (args_cli.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
