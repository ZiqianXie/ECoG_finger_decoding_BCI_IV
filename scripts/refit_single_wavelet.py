#!/usr/bin/env python3
"""Refit one selected single-wavelet model on all labeled development data."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F

from single_wavelet_support import (
    HISTORY,
    OFFSET,
    SAMPLES_PER_BIN,
    SOURCE_RATE,
    linear_gamma_leaf_signals,
    pearson,
    resample_ecog,
)
from cross_validate_single_wavelet import (
    CSP_MODES,
    fit_initialization,
    make_model,
    make_subject_target,
    scoped_event_groups,
    train_final_schedule,
)
from ecog_decoding.models import WaveletPacketEnergy
from ecog_decoding.training import FINGER_NAMES
from single_wavelet_model import extract_all


def atomic_save_npy(path: Path, values: np.ndarray) -> None:
    """Publish a shared cache only after the complete array reaches disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npy")
    np.save(temporary, values, allow_pickle=False)
    os.replace(temporary, path)


def valid_leaf_cache(path: Path, rows: int) -> bool:
    if not path.exists():
        return False
    try:
        values = np.load(path, mmap_mode="r")
        return values.ndim == 2 and values.shape[0] == rows
    except (OSError, ValueError):
        return False


def select_schedule_from_report(
    selection: dict[str, object],
    inner_records: list[dict[str, object]],
    *,
    metric: str,
    rule: str,
) -> tuple[str, dict[str, object]]:
    """Apply the schedule grid recorded by CV instead of module defaults."""
    schedule_names = [str(name) for name in selection["schedule_candidates"]]
    require_update = bool(selection.get("require_lstm_update", False))
    candidates = [
        name for name in schedule_names if not (require_update and name == "frozen_0")
    ]
    if metric not in ("event_macro_nmse", "raw_pcc"):
        raise ValueError(f"unsupported selection metric {metric!r}")
    if rule not in ("one_se", "best"):
        raise ValueError(f"unsupported selection rule {rule!r}")
    summary: dict[str, dict[str, float]] = {}
    for name in schedule_names:
        values = np.asarray(
            [float(record["metrics"][name][metric]) for record in inner_records],
            dtype=np.float64,
        )
        summary[name] = {
            "mean": float(values.mean()),
            "sem": float(values.std(ddof=1) / np.sqrt(values.size)),
        }
    if metric == "raw_pcc":
        best = max(candidates, key=lambda name: summary[name]["mean"])
        threshold = float(summary[best]["mean"] - summary[best]["sem"])
        selected = best if rule == "best" else next(
            name for name in candidates if summary[name]["mean"] >= threshold
        )
    else:
        best = min(candidates, key=lambda name: summary[name]["mean"])
        threshold = float(summary[best]["mean"] + summary[best]["sem"])
        selected = best if rule == "best" else next(
            name for name in candidates if summary[name]["mean"] <= threshold
        )
    return selected, {
        "metric": metric,
        "rule": rule,
        "require_lstm_update": require_update,
        "best": best,
        "selected": selected,
        "threshold": threshold,
        "candidates": summary,
    }


def fit_training_affine(
    prediction: np.ndarray, raw_target: np.ndarray
) -> tuple[float, float]:
    """Map the cleaned-target output back to glove units using development data."""
    keep = np.isfinite(prediction) & np.isfinite(raw_target)
    x = np.asarray(prediction[keep], dtype=np.float64)
    y = np.asarray(raw_target[keep], dtype=np.float64)
    centered = x - x.mean()
    denominator = float(centered @ centered)
    slope = float(centered @ (y - y.mean()) / denominator) if denominator else 1.0
    intercept = float(y.mean() - slope * x.mean())
    return slope, intercept


def plot_prediction(
    path: Path,
    subject: int,
    finger: str,
    target: np.ndarray,
    prediction: np.ndarray,
) -> None:
    width = min(1250, target.size)
    rolling = np.convolve(np.abs(target), np.ones(width), mode="valid")
    start = int(np.argmax(rolling))
    stop = start + width
    time_axis = np.arange(start, stop) / 25.0
    figure, axis = plt.subplots(figsize=(14, 4.8), constrained_layout=True)
    axis.plot(time_axis, target[start:stop], color="black", linewidth=0.8, label="raw glove")
    axis.plot(time_axis, prediction[start:stop], linewidth=0.9, label="prediction")
    axis.set_title(f"S{subject} {finger}: full-development refit")
    axis.set_xlabel("released-test time (s)")
    axis.set_ylabel("flexion")
    axis.legend(frameon=False)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--finger", choices=tuple(FINGER_NAMES), required=True)
    parser.add_argument("--prepared-root", type=Path, default=Path("outputs/preprocessed_v2"))
    parser.add_argument(
        "--selection-root",
        type=Path,
        default=Path("outputs/single_wavelet_1000hz_tap5over2_nested_v2"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/single_wavelet_1000hz_tap5over2_full_refit_v1"),
    )
    parser.add_argument(
        "--initialization-root",
        type=Path,
        default=None,
        help="reuse deterministic full-development ICA/CSP/LARS initialization",
    )
    parser.add_argument(
        "--seed-subdirectory",
        action="store_true",
        help="write under sub<subject>/<finger>/seed<seed>",
    )
    parser.add_argument(
        "--force-schedule",
        default=None,
        help="use a training-only validated schedule instead of recomputing selection",
    )
    parser.add_argument("--resampled-root", type=Path, default=None)
    parser.add_argument("--model-rate", type=int, choices=(400, 1000), default=1000)
    parser.add_argument("--tap-resample-up", type=int, default=5)
    parser.add_argument("--tap-resample-down", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.08)
    parser.add_argument("--movement-threshold", type=float, default=0.10)
    parser.add_argument("--merge-gap-bins", type=int, default=12)
    parser.add_argument("--minimum-event-bins", type=int, default=3)
    parser.add_argument("--maximum-rest-group-bins", type=int, default=250)
    parser.add_argument("--ica-prescreen", type=int, default=512)
    parser.add_argument("--component-chunk", type=int, default=16)
    parser.add_argument("--csp-mode", choices=tuple(CSP_MODES), default="movement_1")
    parser.add_argument(
        "--ica-initialization-root",
        type=Path,
        default=None,
        help="reuse only full-development ICA rows, then refit CSP and LARS",
    )
    parser.add_argument("--hidden-size", type=int, default=10)
    parser.add_argument(
        "--head-initialization",
        choices=("lars_linear_regime",),
        default="lars_linear_regime",
    )
    parser.add_argument("--lars-candidate-scale", type=float, default=1.0)
    parser.add_argument("--lars-near-zero-std", type=float, default=1.0e-3)
    parser.add_argument("--lars-open-gate-bias", type=float, default=5.0)
    parser.add_argument("--lars-forget-gate-bias", type=float, default=-5.0)
    parser.add_argument(
        "--selection-metric",
        choices=("event_macro_nmse", "raw_pcc"),
        default="raw_pcc",
    )
    parser.add_argument(
        "--selection-rule", choices=("one_se", "best"), default="best"
    )
    parser.add_argument(
        "--output-activation", choices=("linear", "softplus"), default="softplus"
    )
    parser.add_argument("--softplus-beta", type=float, default=10.0)
    parser.add_argument("--sequence-steps", type=int, default=100)
    parser.add_argument("--sequence-stride", type=int, default=25)
    parser.add_argument(
        "--sampler-mode",
        choices=("uniform_group", "dense_sequences"),
        default="dense_sequences",
    )
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--head-learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--spatial-learning-rate", type=float, default=3.0e-6)
    parser.add_argument("--wavelet-learning-rate", type=float, default=3.0e-6)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--feature-chunk", type=int, default=256)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--prepare-shared-only",
        action="store_true",
        help="prepare the subject-level resampling and fixed-leaf caches, then exit",
    )
    args = parser.parse_args()
    if args.model_rate % 25:
        raise ValueError("model rate must be divisible by the 25 Hz target rate")
    args.samples_per_bin = args.model_rate // 25
    if args.resampled_root is None:
        args.resampled_root = Path(
            f"/dev/shm/ecog_wavelet_{args.model_rate}hz_"
            f"taps{args.tap_resample_up}over{args.tap_resample_down}"
        )

    started = time.perf_counter()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)
    finger_index = list(FINGER_NAMES).index(args.finger)
    prepared = args.prepared_root / f"sub{args.subject}"
    selection_directory = args.selection_root / f"sub{args.subject}" / args.finger
    selection = json.loads((selection_directory / "summary.json").read_text())
    inner_records = [
        inner
        for outer in selection["outer_folds"]
        for inner in outer["inner_records"]
    ]
    if args.force_schedule is None:
        schedule, schedule_summary = select_schedule_from_report(
            selection,
            inner_records,
            metric=args.selection_metric,
            rule=args.selection_rule,
        )
    else:
        schedule = args.force_schedule
        schedule_summary = {
            "forced_schedule": schedule,
            "selection_basis": str(selection_directory / "summary.json"),
        }

    train_raw_full = np.load(prepared / "train_glove_25hz_raw.npy", mmap_mode="r")
    train_rows = int(train_raw_full.shape[0] - OFFSET)
    train_raw = np.asarray(
        train_raw_full[OFFSET : OFFSET + train_rows], dtype=np.float64
    )
    train_target = make_subject_target(
        train_raw, [[0, train_rows]], [], args.subject
    )
    groups, _, _ = scoped_event_groups(
        train_target,
        [[0, train_rows]],
        args.threshold,
        args.merge_gap_bins,
        args.minimum_event_bins,
        args.maximum_rest_group_bins,
        finger_index,
    )
    training_groups = [[int(start), int(stop)] for start, stop in groups]

    subject_cache = args.resampled_root / f"sub{args.subject}"
    if args.model_rate == SOURCE_RATE:
        train_ecog = np.asarray(np.load(prepared / "train_ecog.npy", mmap_mode="r"))
        test_ecog = np.asarray(np.load(prepared / "test_ecog.npy", mmap_mode="r"))
    else:
        train_ecog = np.asarray(
            resample_ecog(prepared / "train_ecog.npy", subject_cache / "train_ecog.npy")
        )
        test_ecog = np.asarray(
            resample_ecog(prepared / "test_ecog.npy", subject_cache / "test_ecog.npy")
        )
    frontend = WaveletPacketEnergy(
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
    hhl_path = subject_cache / "linear_hhl_100_125.npy"
    hhh_path = subject_cache / "linear_hhh_125_150.npy"
    leaves_valid = valid_leaf_cache(hhl_path, train_ecog.shape[0]) and valid_leaf_cache(
        hhh_path, train_ecog.shape[0]
    )
    if args.prepare_shared_only or args.initialization_root is None:
        if not leaves_valid:
            hhl_values, hhh_values = linear_gamma_leaf_signals(
                train_ecog, frontend, device
            )
            atomic_save_npy(hhl_path, hhl_values)
            atomic_save_npy(hhh_path, hhh_values)
        hhl = np.load(hhl_path, mmap_mode="r")
        hhh = np.load(hhh_path, mmap_mode="r")
    if args.prepare_shared_only:
        print(
            json.dumps(
                {
                    "subject": args.subject,
                    "resampled_cache": str(subject_cache),
                    "leaf_rows": int(hhl.shape[0]),
                    "status": "prepared",
                },
                indent=2,
            )
        )
        return
    if args.initialization_root is not None:
        initialization_directory = (
            args.initialization_root / f"sub{args.subject}" / args.finger
        )
        saved = np.load(initialization_directory / "initialization.npz")
        initialization = {name: saved[name] for name in saved.files}
        spatial = np.load(initialization_directory / "spatial.npy").astype(np.float32)
        source_report = json.loads(
            (initialization_directory / "summary.json").read_text()
        )
        audit = source_report["initialization_audit"]
    else:
        bins = train_ecog.shape[0] // args.samples_per_bin
        joint_bins = np.concatenate(
            (
                hhl[: bins * args.samples_per_bin].reshape(
                    bins, args.samples_per_bin, -1
                ),
                hhh[: bins * args.samples_per_bin].reshape(
                    bins, args.samples_per_bin, -1
                ),
            ),
            axis=1,
        )
        initialization, spatial, audit = fit_initialization(
            ecog=train_ecog,
            joint_bins=joint_bins,
            target=train_target,
            training_intervals=[[0, train_rows]],
            training_groups=training_groups,
            frontend=frontend,
            device=device,
            ica_prescreen=args.ica_prescreen,
            component_chunk=args.component_chunk,
            finger_index=finger_index,
            csp_mode=args.csp_mode,
            samples_per_bin=args.samples_per_bin,
            ica_weights=(
                np.load(
                    args.ica_initialization_root
                    / f"sub{args.subject}"
                    / args.finger
                    / "spatial.npy"
                )[: train_ecog.shape[1]]
                if args.ica_initialization_root is not None
                else None
            ),
        )
    model = make_model(spatial, initialization, args).to(device)
    cached = torch.from_numpy(initialization["selected_features"]).to(device)
    target_tensor = torch.from_numpy(train_target.astype(np.float32)).to(device)
    context = frontend.effective_kernel_size // 2
    train_tensor = torch.from_numpy(train_ecog.copy()).to(device)
    padded_train = F.pad(train_tensor.T[None], (context, context)).squeeze(0).T
    train_final_schedule(
        model=model,
        cached=cached,
        padded_ecog=padded_train,
        target=target_tensor,
        training_groups=training_groups,
        schedule=schedule,
        args=args,
        seed=args.seed,
        finger_index=finger_index,
    )

    test_rows = test_ecog.shape[0] // args.samples_per_bin - HISTORY + 1
    test_tensor = torch.from_numpy(test_ecog.copy()).to(device)
    padded_test = F.pad(test_tensor.T[None], (context, context)).squeeze(0).T
    test_features = extract_all(model, padded_test, test_rows, args.feature_chunk)
    train_features = (
        extract_all(model, padded_train, train_rows, args.feature_chunk)
        if schedule.startswith("unfrozen")
        else cached
    )
    with torch.inference_mode():
        test_prediction = model.decode_features(test_features[None])[0].float().cpu().numpy()
        train_prediction = model.decode_features(train_features[None])[0].float().cpu().numpy()
    test_raw_full = np.load(prepared / "test_glove_25hz_raw.npy", mmap_mode="r")
    test_raw = np.asarray(
        test_raw_full[OFFSET : OFFSET + test_rows, finger_index], dtype=np.float32
    )
    calibration_slope, calibration_intercept = fit_training_affine(
        train_prediction, train_raw[:, finger_index]
    )
    test_prediction_raw_scale = (
        calibration_slope * test_prediction + calibration_intercept
    ).astype(np.float32)

    output = args.output_root / f"sub{args.subject}" / args.finger
    if args.seed_subdirectory:
        output = output / f"seed{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output / "model.pt")
    np.save(output / "test_prediction.npy", test_prediction, allow_pickle=False)
    np.save(
        output / "test_prediction_raw_scale.npy",
        test_prediction_raw_scale,
        allow_pickle=False,
    )
    np.save(output / "test_raw_target.npy", test_raw, allow_pickle=False)
    np.savez(output / "initialization.npz", **initialization)
    np.save(output / "spatial.npy", spatial, allow_pickle=False)
    report = {
        "protocol": "single-path full-development refit after nested event-CV selection",
        "frontend": {
            "name": "single_wavelet_frequency_compressed_depth3",
            "source_rate_hz": SOURCE_RATE,
            "model_rate_hz": args.model_rate,
            "input_resampling": (
                "none" if args.model_rate == SOURCE_RATE else "polyphase_2_over_5_kaiser_8.6"
            ),
            "tap_interpolation": {
                "up": args.tap_resample_up,
                "down": args.tap_resample_down,
                "method": "polyphase_kaiser_8.6",
            },
            "wavelet": "bior6.8",
            "levels": 3,
            "leaf_count": 8,
            "nominal_frequency_edges_hz": list(range(0, 201, 25)),
            "auxiliary_temporal_branches": [],
            "lmp_branch": False,
            "energy_pool_samples": args.samples_per_bin,
        },
        "subject": args.subject,
        "finger": args.finger,
        "decoder": "LARS-initialized standard nonlinear LSTM",
        "selected_schedule": schedule,
        "selection_metric": args.selection_metric,
        "selection_rule": args.selection_rule,
        "sampler_mode": args.sampler_mode,
        "sequence_steps": args.sequence_steps,
        "output_activation": args.output_activation,
        "softplus_beta": args.softplus_beta,
        "csp_mode": args.csp_mode,
        "released_test_used_for_selection": False,
        "released_test_pcc": pearson(test_prediction, test_raw),
        "output_calibration": {
            "fit_on": "full labeled development recording only",
            "slope": calibration_slope,
            "intercept": calibration_intercept,
            "released_test_labels_used": False,
        },
        "full_development_raw_pcc": pearson(
            train_prediction, train_raw[:, finger_index]
        ),
        "full_development_prediction_sd": float(np.std(train_prediction)),
        "full_development_prediction_finite": bool(
            np.isfinite(train_prediction).all()
        ),
        "full_development_cleaned_pcc": pearson(
            train_prediction, train_target[:, finger_index]
        ),
        "initialization_audit": audit,
        "schedule_selection": schedule_summary,
        "runtime_seconds": time.perf_counter() - started,
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    plot_prediction(
        output / "test_trace.png",
        args.subject,
        args.finger,
        test_raw,
        test_prediction_raw_scale,
    )
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
