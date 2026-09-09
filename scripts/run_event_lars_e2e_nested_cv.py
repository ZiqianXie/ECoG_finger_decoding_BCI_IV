#!/usr/bin/env python3
"""Coordinate end-to-end nested event-fold jobs on a shared filesystem."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

from ecog_decoding.training import FINGER_NAMES


def acquire_claim(destination: Path, timeout_hours: float) -> Path | None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    claim = destination.with_suffix(".claim")
    if claim.exists() and time.time() - claim.stat().st_mtime > timeout_hours * 3600:
        claim.unlink(missing_ok=True)
    try:
        descriptor = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return None
    with os.fdopen(descriptor, "w") as stream:
        stream.write(
            f"host={os.uname().nodename} coordinator_pid={os.getpid()} "
            f"claimed_at={time.time()}\n"
        )
    return claim


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", type=int, nargs="+", default=(1, 2, 3))
    parser.add_argument("--fingers", nargs="+", choices=tuple(FINGER_NAMES), default=tuple(FINGER_NAMES))
    parser.add_argument(
        "--pairs",
        nargs="+",
        default=None,
        metavar="SUBJECT:FINGER",
        help="Optional explicit subject:finger pairs; overrides --subjects/--fingers.",
    )
    parser.add_argument("--folds", type=int, nargs="+", default=(0, 1, 2))
    parser.add_argument("--seeds", type=int, nargs="+", default=(0,))
    parser.add_argument("--gpus", nargs="+", default=("0", "1", "2", "3"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/event_lars_e2e_nested_strict_v1"))
    parser.add_argument("--fold-root", type=Path, default=Path("outputs/event_stratified_folds_v1"))
    parser.add_argument("--feature-root", type=Path, default=Path("outputs/windowed_ica_wavelet_asymmetric_v1"))
    parser.add_argument("--ica-root", type=Path, default=Path("outputs/paper_ica_lars_v1"))
    parser.add_argument(
        "--frontend",
        choices=("asymmetric", "overcomplete"),
        default="asymmetric",
    )
    parser.add_argument("--selection-cache-root", type=Path, default=Path("outputs/event_lars_selection_v1"))
    parser.add_argument("--inner-selection-cache-root", type=Path, default=Path("outputs/event_lars_inner_selection_v1"))
    parser.add_argument("--target", default=None)
    parser.add_argument(
        "--target-map",
        type=Path,
        default=None,
        help="YAML mapping from subject and finger to target file stem",
    )
    parser.add_argument("--warmup-epochs", type=int, default=6)
    parser.add_argument("--max-epochs", type=int, default=24)
    parser.add_argument("--max-features", type=int, default=512)
    parser.add_argument("--hidden-size", type=int, default=10)
    parser.add_argument("--near-zero-std", type=float, default=1.0e-3)
    parser.add_argument("--candidate-scale", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=3.0e-5)
    parser.add_argument("--gate-learning-rate", type=float, default=1.0e-3)
    parser.add_argument(
        "--spatial-cache-root",
        "--shared-beta-csp-cache-root",
        dest="spatial_cache_root",
        type=Path,
        default=None,
        help="optional per-finger spatial/filter cache",
    )
    parser.add_argument("--spatial-learning-rate", type=float, default=1.0e-6)
    parser.add_argument("--wavelet-learning-rate", type=float, default=1.0e-6)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--unfrozen-batch-size", type=int, default=6)
    parser.add_argument("--sequence-steps", type=int, default=50)
    parser.add_argument("--sequence-stride", type=int, default=25)
    parser.add_argument("--prediction-chunk-steps", type=int, default=512)
    parser.add_argument(
        "--loss", choices=("mse", "joint", "velocity_huber"), default="mse"
    )
    parser.add_argument("--movement-weight", type=float, default=4.0)
    parser.add_argument("--velocity-weight", type=float, default=0.2)
    parser.add_argument("--velocity-huber-beta", type=float, default=1.0)
    parser.add_argument("--correlation-weight", type=float, default=0.1)
    parser.add_argument(
        "--output-activation",
        choices=("linear", "softplus", "hurdle"),
        default="linear",
    )
    parser.add_argument("--claim-timeout-hours", type=float, default=6.0)
    parser.add_argument("--cpu-threads-per-job", type=int, default=4)
    parser.add_argument(
        "--cached-only", action=argparse.BooleanOptionalAction, default=False
    )
    args = parser.parse_args()
    if args.cpu_threads_per_job < 1:
        parser.error("--cpu-threads-per-job must be positive")
    if args.target is not None and args.target_map is not None:
        parser.error("--target and --target-map are mutually exclusive")
    target_map = yaml.safe_load(args.target_map.read_text()) if args.target_map else {}

    if args.pairs is None:
        task_pairs = [
            (subject, finger)
            for subject in args.subjects
            for finger in args.fingers
        ]
    else:
        task_pairs = []
        for pair in args.pairs:
            try:
                subject_text, finger = pair.split(":", maxsplit=1)
                subject = int(subject_text)
            except ValueError:
                parser.error(f"invalid --pairs entry {pair!r}; expected SUBJECT:FINGER")
            if subject not in (1, 2, 3) or finger not in FINGER_NAMES:
                parser.error(f"invalid --pairs entry {pair!r}")
            task_pairs.append((subject, finger))
        if len(task_pairs) != len(set(task_pairs)):
            parser.error("--pairs contains a duplicate subject:finger entry")

    tasks: list[tuple[str, list[str], Path]] = []
    # Seed-major ordering makes each worker build a different selection cache;
    # later seeds then reuse it instead of occupying GPUs while waiting on locks.
    for seed in args.seeds:
        for subject, finger in task_pairs:
            for fold in args.folds:
                task_target = args.target
                if args.target_map is not None:
                    subject_targets = target_map.get(
                        subject, target_map.get(str(subject), {})
                    )
                    task_target = subject_targets.get(finger)
                    if task_target is None:
                        raise KeyError(
                            f"target map has no entry for S{subject} {finger}"
                        )
                destination = (
                    args.output_root / f"sub{subject}" / finger
                    / f"fold{fold}" / f"seed{seed}" / "summary.json"
                )
                if destination.exists():
                    continue
                name = f"s{subject}_{finger}_f{fold}_seed{seed}"
                command = [
                    sys.executable,
                    "scripts/train_event_grouped_lars_e2e_nested.py",
                    "--subject", str(subject),
                    "--finger", finger,
                    "--fold", str(fold),
                    "--seed", str(seed),
                    "--warmup-epochs", str(args.warmup_epochs),
                    "--max-epochs", str(args.max_epochs),
                    "--max-features", str(args.max_features),
                    "--hidden-size", str(args.hidden_size),
                    "--near-zero-std", str(args.near_zero_std),
                    "--candidate-scale", str(args.candidate_scale),
                    "--learning-rate", str(args.learning_rate),
                    "--gate-learning-rate", str(args.gate_learning_rate),
                    "--spatial-learning-rate", str(args.spatial_learning_rate),
                    "--wavelet-learning-rate", str(args.wavelet_learning_rate),
                    "--weight-decay", str(args.weight_decay),
                    "--batch-size", str(args.batch_size),
                    "--unfrozen-batch-size", str(args.unfrozen_batch_size),
                    "--sequence-steps", str(args.sequence_steps),
                    "--sequence-stride", str(args.sequence_stride),
                    "--prediction-chunk-steps", str(args.prediction_chunk_steps),
                    "--loss", args.loss,
                    "--movement-weight", str(args.movement_weight),
                    "--velocity-weight", str(args.velocity_weight),
                    "--velocity-huber-beta", str(args.velocity_huber_beta),
                    "--correlation-weight", str(args.correlation_weight),
                    "--output-activation", args.output_activation,
                    "--output-root", str(args.output_root),
                    "--fold-root", str(args.fold_root),
                    "--feature-root", str(args.feature_root),
                    "--ica-root", str(args.ica_root),
                    "--frontend", args.frontend,
                    "--selection-cache-root", str(args.selection_cache_root),
                    "--inner-selection-cache-root", str(args.inner_selection_cache_root),
                ]
                if args.spatial_cache_root is not None:
                    command.extend(
                        (
                            "--spatial-cache-root",
                            str(args.spatial_cache_root),
                        )
                    )
                if args.cached_only:
                    command.append("--cached-only")
                if task_target is not None:
                    command.extend(("--target", task_target))
                tasks.append((name, command, destination))

    log_root = args.output_root / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    pending = list(tasks)
    active: list[tuple[subprocess.Popen[bytes], object, str, str, Path]] = []
    failures: list[str] = []
    while pending or active:
        busy = {gpu for _, _, _, gpu, _ in active}
        for gpu in args.gpus:
            if not pending or gpu in busy:
                continue
            name, command, destination = pending.pop(0)
            if destination.exists():
                continue
            claim = acquire_claim(destination, args.claim_timeout_hours)
            if claim is None:
                continue
            log = open(log_root / f"{name}.log", "wb")
            environment = os.environ.copy()
            environment["PYTHONPATH"] = "scripts:src"
            environment["CUDA_VISIBLE_DEVICES"] = gpu
            for env_name in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
            ):
                environment[env_name] = str(args.cpu_threads_per_job)
            process = subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT, env=environment
            )
            active.append((process, log, name, gpu, claim))
            print(f"started {name} pid={process.pid} gpu={gpu}", flush=True)
        time.sleep(1.0)
        remaining: list[tuple[subprocess.Popen[bytes], object, str, str, Path]] = []
        for process, log, name, gpu, claim in active:
            code = process.poll()
            if code is None:
                remaining.append((process, log, name, gpu, claim))
                continue
            log.close()
            claim.unlink(missing_ok=True)
            print(f"finished {name} exit={code}", flush=True)
            if code:
                failures.append(name)
        active = remaining
    if failures:
        raise RuntimeError("failed tasks: " + ", ".join(failures))


if __name__ == "__main__":
    main()
