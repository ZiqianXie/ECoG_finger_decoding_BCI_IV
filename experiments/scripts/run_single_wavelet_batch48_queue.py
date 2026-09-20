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
    recurrent_cell: str = "residual_lstm",
    residual_input: str = "current_candidate",
    residual_input_width: int = 64,
    hidden_size: int = 64,
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
    initialization_raw_target_blend: float = 0.0,
    warmup_steps: int = 0,
    movement_loss_weight: float = 0.5,
    movement_head_scope: str = "target",
    movement_head_objective: str = "binary_state",
    csp_contrast_mode: str = "common_rest",
    amplitude_target_lead_bins: int = 0,
    future_context_bins: int = 0,
    initialization_only: bool = False,
    train_stem: bool = False,
    spatial_learning_rate: float = 3.0e-6,
    spatial_anchor_weight: float = 0.0,
    wavelet_learning_rate: float = 0.0,
    unfreeze_after: int = 100,
    unfrozen_update_grid: tuple[int, ...] = (20, 40, 80, 120, 200),
    spoc_auxiliary_weight: float = 0.0,
    spoc_active_threshold: float = 0.20,
    spatial_orthogonality_weight: float = 0.0,
    tap_resample_up: int = 5,
    tap_resample_down: int = 2,
    auxiliary_residual_readout_l2: float | None = None,
    model_seed: int = 2026,
    split_seed: int | None = None,
    sampler_seed: int | None = None,
    reuse_inner_metrics_root: Path | None = None,
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
        (
            f"/dev/shm/ecog_wavelet_1000hz_taps"
            f"{tap_resample_up}over{tap_resample_down}/sub{subject}"
        ),
        "--model-rate",
        "1000",
        "--tap-resample-up",
        str(tap_resample_up),
        "--tap-resample-down",
        str(tap_resample_down),
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
        "--amplitude-target-lead-bins",
        str(amplitude_target_lead_bins),
        "--future-context-bins",
        str(future_context_bins),
        "--csp-band-mode",
        csp_band_mode,
        "--csp-band-cache-root",
        str(csp_band_cache_root),
        "--hidden-size",
        str(hidden_size),
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
        recurrent_cell,
        "--residual-input",
        residual_input,
        "--output-activation",
        "softplus",
        "--residual-history-bins",
        "1",
        *(
            ("--residual-input-width", str(residual_input_width))
            if residual_input in ("current_candidate", "causal_candidate", "selected_causal")
            else ()
        ),
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
        "--spatial-learning-rate",
        str(spatial_learning_rate),
        "--spatial-anchor-weight",
        str(spatial_anchor_weight),
        "--spoc-auxiliary-weight",
        str(spoc_auxiliary_weight),
        "--spoc-active-threshold",
        str(spoc_active_threshold),
        "--spatial-orthogonality-weight",
        str(spatial_orthogonality_weight),
        "--wavelet-learning-rate",
        str(wavelet_learning_rate),
        "--frozen-update-grid",
        *(str(update) for update in frozen_update_grid),
        "--weight-decay",
        "0.0001",
        "--movement-loss-weight",
        str(movement_loss_weight),
        "--movement-head-scope",
        movement_head_scope,
        "--movement-head-objective",
        movement_head_objective,
        *(
            (
                "--auxiliary-residual-readout-l2",
                str(auxiliary_residual_readout_l2),
            )
            if auxiliary_residual_readout_l2 is not None
            else ()
        ),
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
        "--initialization-raw-target-blend",
        str(initialization_raw_target_blend),
        "--correlation-loss-weight",
        str(correlation_loss_weight),
        "--derivative-correlation-weight",
        str(derivative_correlation_weight),
        "--raw-movement-correlation-weight",
        str(raw_movement_correlation_weight),
        "--raw-movement-derivative-correlation-weight",
        str(raw_movement_derivative_correlation_weight),
        "--seed",
        str(model_seed),
        "--feature-chunk",
        "256",
        "--compile",
        "--device",
        "cuda",
    ]
    if split_seed is not None:
        command.extend(("--split-seed", str(split_seed)))
    if sampler_seed is not None:
        command.extend(("--sampler-seed", str(sampler_seed)))
    if reuse_inner_metrics_root is not None:
        command.extend(
            (
                "--reuse-inner-metrics-from",
                str(
                    reuse_inner_metrics_root
                    / f"sub{subject}"
                    / finger
                    / f"outer{outer_fold}"
                ),
            )
        )
    if initialization_only:
        command.append("--initialization-only")
    if train_stem:
        command.extend(
            [
                "--unfreeze-after",
                str(unfreeze_after),
                "--unfrozen-update-grid",
                *(str(update) for update in unfrozen_update_grid),
            ]
        )
    else:
        command.append("--frozen-only")
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
        choices=("ica_only", "movement_1", "movement_2", "movement_4", "tails_1x1", "tails_2x2", "tails_4x4"),
        default="movement_1",
    )
    parser.add_argument(
        "--csp-contrast-mode",
        choices=(
            "common_rest",
            "other_movement",
            "dual_rest_other",
            "dual_rest_amplitude",
            "triple_rest_other_continuous_amplitude",
            "continuous_amplitude",
            "continuous_velocity",
            "dual_rest_velocity",
            "all_finger_amplitude",
            "target_rest_all_finger_amplitude",
            "target_rest_all_finger_amplitude_synergy",
            "target_rest_all_finger_amplitude_synergy_conditional",
            "all_finger_rest_amplitude",
        ),
        default="common_rest",
    )
    parser.add_argument("--amplitude-target-lead-bins", type=int, default=0)
    parser.add_argument("--future-context-bins", type=int, default=0)
    parser.add_argument(
        "--recurrent-cell",
        choices=("residual_lstm", "residual_gru", "residual_bilstm"),
        default="residual_lstm",
    )
    parser.add_argument(
        "--residual-input",
        choices=("selected", "candidate", "current_candidate", "causal_candidate", "selected_causal"),
        default="current_candidate",
    )
    parser.add_argument("--residual-input-width", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--residual-output-init-std", type=float, default=0.0)
    parser.add_argument("--head-learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--train-stem", action="store_true")
    parser.add_argument("--spatial-learning-rate", type=float, default=3.0e-6)
    parser.add_argument("--spatial-anchor-weight", type=float, default=0.0)
    parser.add_argument("--spoc-auxiliary-weight", type=float, default=0.0)
    parser.add_argument("--spoc-active-threshold", type=float, default=0.20)
    parser.add_argument("--spatial-orthogonality-weight", type=float, default=0.0)
    parser.add_argument("--tap-resample-up", type=int, default=5)
    parser.add_argument("--tap-resample-down", type=int, default=2)
    parser.add_argument("--wavelet-learning-rate", type=float, default=0.0)
    parser.add_argument("--unfreeze-after", type=int, default=100)
    parser.add_argument(
        "--residual-dynamics",
        choices=("pointwise", "leaky_velocity"),
        default="pointwise",
    )
    parser.add_argument("--residual-decay", type=float, default=0.95)
    parser.add_argument("--raw-trajectory-blend", type=float, default=0.0)
    parser.add_argument("--initialization-raw-target-blend", type=float, default=0.0)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--movement-loss-weight", type=float, default=0.5)
    parser.add_argument(
        "--movement-head-scope",
        choices=("target", "all_fingers"),
        default="target",
    )
    parser.add_argument(
        "--movement-head-objective",
        choices=("binary_state", "continuous_trajectory"),
        default="binary_state",
    )
    parser.add_argument("--auxiliary-residual-readout-l2", type=float, default=None)
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
    parser.add_argument(
        "--unfrozen-update-grid",
        type=int,
        nargs="+",
        default=(20, 40, 80, 120, 200),
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
    parser.add_argument("--model-seed", type=int, default=2026)
    parser.add_argument("--split-seed", type=int, default=None)
    parser.add_argument("--sampler-seed", type=int, default=None)
    parser.add_argument("--reuse-inner-metrics-root", type=Path, default=None)
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
            args.recurrent_cell,
            args.residual_input,
            args.residual_input_width,
            args.hidden_size,
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
            args.initialization_raw_target_blend,
            args.warmup_steps,
            args.movement_loss_weight,
            args.movement_head_scope,
            args.movement_head_objective,
            args.csp_contrast_mode,
            args.amplitude_target_lead_bins,
            args.future_context_bins,
            args.initialization_only,
            args.train_stem,
            args.spatial_learning_rate,
            args.spatial_anchor_weight,
            args.wavelet_learning_rate,
            args.unfreeze_after,
            tuple(args.unfrozen_update_grid),
            args.spoc_auxiliary_weight,
            args.spoc_active_threshold,
            args.spatial_orthogonality_weight,
            args.tap_resample_up,
            args.tap_resample_down,
            args.auxiliary_residual_readout_l2,
            args.model_seed,
            args.split_seed,
            args.sampler_seed,
            args.reuse_inner_metrics_root,
        )
        print(f"start {label}", flush=True)
        subprocess.run(command, check=True)
        if not completed_summary(summary, outer_fold):
            raise RuntimeError(f"{label} exited without a complete summary")
        print(f"finish {label}", flush=True)


if __name__ == "__main__":
    main()
