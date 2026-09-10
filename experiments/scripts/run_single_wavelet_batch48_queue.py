#!/usr/bin/env python3
"""Run resumable outer-fold queues for the fixed batch-48 residual screen."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from ecog_decoding.training import FINGER_NAMES


def parse_spec(value: str) -> tuple[int, str, int]:
    pieces = value.split(":")
    if len(pieces) != 3:
        raise argparse.ArgumentTypeError("spec must be SUBJECT:FINGER:OUTER_FOLD")
    try:
        subject = int(pieces[0])
        outer_fold = int(pieces[2])
    except ValueError as error:
        raise argparse.ArgumentTypeError("subject and outer fold must be integers") from error
    finger = pieces[1]
    if subject not in (1, 2, 3):
        raise argparse.ArgumentTypeError("subject must be 1, 2, or 3")
    if finger not in FINGER_NAMES:
        raise argparse.ArgumentTypeError(f"unknown finger: {finger}")
    if outer_fold not in (0, 1, 2):
        raise argparse.ArgumentTypeError("outer fold must be 0, 1, or 2")
    return subject, finger, outer_fold


def completed_summary(path: Path, outer_fold: int) -> bool:
    if not path.is_file():
        return False
    try:
        report = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    observed = [int(item["outer_fold"]) for item in report.get("outer_folds", [])]
    return observed == [outer_fold]


def build_command(
    python: str,
    script: Path,
    subject: int,
    finger: str,
    outer_fold: int,
    output_root: Path,
    initialization_root: Path,
    ica_cache_root: Path,
    csp_band_mode: str = "joint_hhl_hhh",
    residual_input_width: int = 64,
    correlation_loss_weight: float = 0.0,
    derivative_correlation_weight: float = 0.0,
    raw_movement_correlation_weight: float = 0.0,
    raw_movement_derivative_correlation_weight: float = 0.0,
    lasso_backend: str = "torch_fista",
    csp_mode: str = "movement_1",
    frozen_update_grid: tuple[int, ...] = (10, 25, 50, 100, 200),
    wavelet_frontend: str = "depth3",
) -> list[str]:
    output = output_root / f"sub{subject}" / finger / f"outer{outer_fold}"
    initialization = initialization_root / f"sub{subject}" / finger
    return [
        python,
        str(script),
        "--subject",
        str(subject),
        "--finger",
        finger,
        "--outer-folds",
        str(outer_fold),
        "--prepared-root",
        "outputs/preprocessed_v2",
        "--fold-root",
        "outputs/event_stratified_folds_fulldev_targetsafe_conservative_v1",
        "--resampled-cache",
        f"/dev/shm/ecog_wavelet_1000hz/sub{subject}/train_ecog.npy",
        "--leaf-cache",
        f"/dev/shm/ecog_wavelet_1000hz_taps5over2/sub{subject}",
        "--model-rate",
        "1000",
        "--tap-resample-up",
        "5",
        "--tap-resample-down",
        "2",
        "--wavelet-frontend",
        wavelet_frontend,
        "--output",
        str(output),
        "--initialization-cache-root",
        str(initialization),
        "--ica-cache-root",
        str(ica_cache_root),
        "--require-lstm-update",
        "--selection-metric",
        "raw_pcc",
        "--selection-rule",
        "best",
        "--threshold",
        "0.08",
        "--movement-threshold",
        "0.1",
        "--rest-threshold",
        "0.05",
        "--merge-gap-bins",
        "12",
        "--minimum-event-bins",
        "3",
        "--maximum-rest-group-bins",
        "250",
        "--purge-bins",
        "95",
        "--ica-prescreen",
        "512",
        "--component-chunk",
        "16",
        "--lasso-backend",
        lasso_backend,
        "--csp-mode",
        csp_mode,
        "--csp-band-mode",
        csp_band_mode,
        "--hidden-size",
        "64",
        "--head-initialization",
        "lars_linear_regime",
        "--lars-candidate-scale",
        "1.0",
        "--lars-near-zero-std",
        "0.001",
        "--lars-open-gate-bias",
        "5.0",
        "--lars-forget-gate-bias",
        "-5.0",
        "--recurrent-cell",
        "residual_lstm",
        "--residual-input",
        "current_candidate",
        "--output-activation",
        "softplus",
        "--residual-history-bins",
        "1",
        "--residual-input-width",
        str(residual_input_width),
        "--residual-include-direct",
        "--softplus-beta",
        "10.0",
        "--sequence-steps",
        "100",
        "--sequence-stride",
        "25",
        "--sampler-mode",
        "dense_sequences",
        "--batch-size",
        "48",
        "--head-learning-rate",
        "0.0003",
        "--frozen-update-grid",
        *(str(update) for update in frozen_update_grid),
        "--frozen-only",
        "--weight-decay",
        "0.0001",
        "--movement-loss-weight",
        "0.5",
        "--movement-trajectory-weight",
        "1.0",
        "--velocity-loss-weight",
        "0.0",
        "--residual-output-init-std",
        "0.0",
        "--correlation-loss-weight",
        str(correlation_loss_weight),
        "--derivative-correlation-weight",
        str(derivative_correlation_weight),
        "--raw-movement-correlation-weight",
        str(raw_movement_correlation_weight),
        "--raw-movement-derivative-correlation-weight",
        str(raw_movement_derivative_correlation_weight),
        "--seed",
        "2026",
        "--feature-chunk",
        "256",
        "--compile",
        "--device",
        "cuda",
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", action="append", type=parse_spec, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/candidate_residual_current_bin_batch48_oof_all_v1"),
    )
    parser.add_argument(
        "--initialization-root",
        type=Path,
        default=Path("/dev/shm/ecog_candidate_pool_batch48_all_v1"),
    )
    parser.add_argument(
        "--ica-cache-root",
        type=Path,
        default=Path("outputs/single_wavelet_1000hz_tap5over2_nested_v2"),
    )
    parser.add_argument(
        "--csp-band-mode",
        choices=(
            "joint_hhl_hhh",
            "separate_hhl_hhh",
            "separate_50_100_hhl_hhh",
            "designed_seven",
        ),
        default="joint_hhl_hhh",
    )
    parser.add_argument(
        "--csp-mode",
        choices=("ica_only", "movement_1", "movement_2", "movement_4", "tails_2x2", "tails_4x4"),
        default="movement_1",
    )
    parser.add_argument("--residual-input-width", type=int, default=64)
    parser.add_argument(
        "--wavelet-frontend",
        choices=("depth3", "overcomplete_depth3_depth4"),
        default="depth3",
    )
    parser.add_argument(
        "--frozen-update-grid",
        type=int,
        nargs="+",
        default=(10, 25, 50, 100, 200),
    )
    parser.add_argument("--correlation-loss-weight", type=float, default=0.0)
    parser.add_argument("--derivative-correlation-weight", type=float, default=0.0)
    parser.add_argument("--raw-movement-correlation-weight", type=float, default=0.0)
    parser.add_argument(
        "--raw-movement-derivative-correlation-weight", type=float, default=0.0
    )
    parser.add_argument(
        "--lasso-backend",
        choices=("sklearn_lars", "torch_fista"),
        default="torch_fista",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--script", type=Path, default=Path("scripts/cross_validate_single_wavelet.py")
    )
    args = parser.parse_args()

    for subject, finger, outer_fold in args.spec:
        output = args.output_root / f"sub{subject}" / finger / f"outer{outer_fold}"
        summary = output / "summary.json"
        label = f"S{subject}-{finger}-outer{outer_fold}"
        if completed_summary(summary, outer_fold):
            print(f"skip completed {label}", flush=True)
            continue
        command = build_command(
            args.python,
            args.script,
            subject,
            finger,
            outer_fold,
            args.output_root,
            args.initialization_root,
            args.ica_cache_root,
            args.csp_band_mode,
            args.residual_input_width,
            args.correlation_loss_weight,
            args.derivative_correlation_weight,
            args.raw_movement_correlation_weight,
            args.raw_movement_derivative_correlation_weight,
            args.lasso_backend,
            args.csp_mode,
            tuple(args.frozen_update_grid),
            args.wavelet_frontend,
        )
        print(f"start {label}", flush=True)
        subprocess.run(command, check=True)
        if not completed_summary(summary, outer_fold):
            raise RuntimeError(f"{label} exited without a complete summary")
        print(f"finish {label}", flush=True)


if __name__ == "__main__":
    main()
