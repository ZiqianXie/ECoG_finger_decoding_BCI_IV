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


def missing_designed_band_caches(
    specs: list[tuple[int, str, int]], csp_band_mode: str, root: Path
) -> list[Path]:
    """Return missing node-local designed-band caches for requested subjects."""
    if csp_band_mode != "designed_seven":
        return []
    return [
        root / f"sub{subject}" / "train_filtered_bands.npy"
        for subject in sorted({subject for subject, _, _ in specs})
        if not (root / f"sub{subject}" / "train_filtered_bands.npy").is_file()
    ]


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
    csp_band_cache_root: Path = Path("/dev/shm/ecog_csp_band_cache"),
    residual_input_width: int = 64,
    correlation_loss_weight: float = 0.0,
    derivative_correlation_weight: float = 0.0,
    raw_movement_correlation_weight: float = 0.0,
    raw_movement_derivative_correlation_weight: float = 0.0,
    lasso_backend: str = "torch_fista",
    csp_mode: str = "movement_1",
    frozen_update_grid: tuple[int, ...] = (10, 25, 50, 100, 200),
    wavelet_frontend: str = "depth3",
    residual_output_init_std: float = 0.0,
    head_learning_rate: float = 3.0e-4,
    residual_dynamics: str = "pointwise",
    residual_decay: float = 0.95,
    raw_trajectory_blend: float = 0.0,
    warmup_steps: int = 0,
    movement_head_scope: str = "target",
    csp_contrast_mode: str = "common_rest",
    initialization_only: bool = False,
) -> list[str]:
    output = output_root / f"sub{subject}" / finger / f"outer{outer_fold}"
    initialization = initialization_root / f"sub{subject}" / finger
    command = [
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
        *(
            ("--no-require-lstm-update",)
            if initialization_only
            else ("--require-lstm-update",)
        ),
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
        "--csp-contrast-mode",
        csp_contrast_mode,
        "--csp-band-mode",
        csp_band_mode,
        "--csp-band-cache-root",
        str(csp_band_cache_root),
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
        "--warmup-steps",
        str(warmup_steps),
        "--sequence-stride",
        "25",
        "--sampler-mode",
        "dense_sequences",
        "--batch-size",
        "48",
        "--head-learning-rate",
        str(head_learning_rate),
        "--frozen-update-grid",
        *(str(update) for update in frozen_update_grid),
        "--frozen-only",
        "--weight-decay",
        "0.0001",
        "--movement-loss-weight",
        "0.5",
        "--movement-head-scope",
        movement_head_scope,
        "--movement-trajectory-weight",
        "1.0",
        "--velocity-loss-weight",
        "0.0",
        "--residual-output-init-std",
        str(residual_output_init_std),
        "--residual-dynamics",
        residual_dynamics,
        "--residual-decay",
        str(residual_decay),
        "--raw-trajectory-blend",
        str(raw_trajectory_blend),
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
    if initialization_only:
        command.append("--initialization-only")
    return command


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
        "--csp-band-cache-root",
        type=Path,
        default=Path("/dev/shm/ecog_csp_band_cache"),
    )
    parser.add_argument(
        "--csp-mode",
        choices=("ica_only", "movement_1", "movement_2", "movement_4", "tails_2x2", "tails_4x4"),
        default="movement_1",
    )
    parser.add_argument(
        "--csp-contrast-mode",
        choices=("common_rest", "other_movement", "dual_rest_other"),
        default="common_rest",
    )
    parser.add_argument("--residual-input-width", type=int, default=64)
    parser.add_argument("--residual-output-init-std", type=float, default=0.0)
    parser.add_argument("--head-learning-rate", type=float, default=3.0e-4)
    parser.add_argument(
        "--residual-dynamics",
        choices=("pointwise", "leaky_velocity"),
        default="pointwise",
    )
    parser.add_argument("--residual-decay", type=float, default=0.95)
    parser.add_argument("--raw-trajectory-blend", type=float, default=0.0)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument(
        "--movement-head-scope",
        choices=("target", "all_fingers"),
        default="target",
    )
    parser.add_argument(
        "--wavelet-frontend",
        choices=(
            "depth3",
            "overcomplete_depth3_depth4",
            "overcomplete_depth3_depth4_depth5",
        ),
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
    parser.add_argument(
        "--initialization-only",
        action="store_true",
        help="fit and score only outer sparse initializers; skip all LSTM work",
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--script", type=Path, default=Path("scripts/cross_validate_single_wavelet.py")
    )
    args = parser.parse_args()

    missing = missing_designed_band_caches(
        args.spec, args.csp_band_mode, args.csp_band_cache_root
    )
    if missing:
        joined = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            "designed-seven CSP cache is missing: "
            f"{joined}. Prepare it once with scripts/cache_csp_band_signals.py."
        )

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
            args.csp_band_cache_root,
            args.residual_input_width,
            args.correlation_loss_weight,
            args.derivative_correlation_weight,
            args.raw_movement_correlation_weight,
            args.raw_movement_derivative_correlation_weight,
            args.lasso_backend,
            args.csp_mode,
            tuple(args.frozen_update_grid),
            args.wavelet_frontend,
            args.residual_output_init_std,
            args.head_learning_rate,
            args.residual_dynamics,
            args.residual_decay,
            args.raw_trajectory_blend,
            args.warmup_steps,
            args.movement_head_scope,
            args.csp_contrast_mode,
            args.initialization_only,
        )
        print(f"start {label}", flush=True)
        subprocess.run(command, check=True)
        if not completed_summary(summary, outer_fold):
            raise RuntimeError(f"{label} exited without a complete summary")
        print(f"finish {label}", flush=True)


if __name__ == "__main__":
    main()
