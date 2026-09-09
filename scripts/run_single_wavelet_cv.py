#!/usr/bin/env python3
"""Run the 15 subject/finger single-path models efficiently across GPUs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from queue import Empty, Queue

from cross_validate_single_wavelet import CSP_MODES
from ecog_decoding.training import FINGER_NAMES


def completed_summary(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        report = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return {int(item["outer_fold"]) for item in report.get("outer_folds", [])} == {
        0,
        1,
        2,
    }


def command(args: argparse.Namespace, subject: int, finger: str) -> list[str]:
    destination = args.output_root / f"sub{subject}" / finger
    values = [
        sys.executable,
            "scripts/cross_validate_single_wavelet.py",
        "--subject",
        str(subject),
        "--finger",
        finger,
        "--model-rate",
        str(args.model_rate),
        "--tap-resample-up",
        str(args.tap_resample_up),
        "--tap-resample-down",
        str(args.tap_resample_down),
        "--fold-root",
        str(args.fold_root),
        "--output",
        str(destination),
        "--head-learning-rate",
        str(args.head_learning_rate),
        "--spatial-learning-rate",
        str(args.spatial_learning_rate),
        "--wavelet-learning-rate",
        str(args.wavelet_learning_rate),
        "--head-initialization",
        args.head_initialization,
        "--recurrent-cell",
        args.recurrent_cell,
        "--csp-mode",
        args.csp_mode,
        "--lars-candidate-scale",
        str(args.lars_candidate_scale),
        "--lars-near-zero-std",
        str(args.lars_near_zero_std),
        "--lars-open-gate-bias",
        str(args.lars_open_gate_bias),
        "--lars-forget-gate-bias",
        str(args.lars_forget_gate_bias),
        "--sequence-steps",
        str(args.sequence_steps),
        "--sequence-stride",
        str(args.sequence_stride),
        "--sampler-mode",
        args.sampler_mode,
        "--batch-size",
        str(args.batch_size),
        "--output-activation",
        args.output_activation,
        "--softplus-beta",
        str(args.softplus_beta),
        "--little-event-ratio-low",
        str(getattr(args, "little_event_ratio_low", 0.8)),
        "--little-event-ratio-high",
        str(getattr(args, "little_event_ratio_high", 1.2)),
        "--selection-metric",
        args.selection_metric,
        "--selection-rule",
        args.selection_rule,
        "--purge-bins",
        str(args.purge_bins),
        "--frozen-update-grid",
        *[str(value) for value in args.frozen_update_grid],
        "--unfreeze-after",
        str(args.unfreeze_after),
        "--unfrozen-update-grid",
        *[str(value) for value in args.unfrozen_update_grid],
        "--seed",
        str(args.seed),
    ]
    if args.require_lstm_update:
        values.append("--require-lstm-update")
    else:
        values.append("--no-require-lstm-update")
    if getattr(args, "little_event_decontamination", False):
        values.append("--little-event-decontamination")
    if args.reuse_from_root is not None:
        source = args.reuse_from_root / f"sub{subject}" / finger
        values.extend(
            [
                "--initialization-cache-root",
                str(source),
                "--reuse-inner-metrics-from",
                str(source),
            ]
        )
    if args.initialization_from_root is not None:
        source = args.initialization_from_root / f"sub{subject}" / finger
        values.extend(["--initialization-cache-root", str(source)])
    if args.reuse_inner_metrics_from_root is not None:
        source = args.reuse_inner_metrics_from_root / f"sub{subject}" / finger
        values.extend(["--reuse-inner-metrics-from", str(source)])
    if args.ica_cache_root is not None:
        values.extend(["--ica-cache-root", str(args.ica_cache_root)])
    if not args.compile:
        values.append("--no-compile")
    return values


def prepare_shared(args: argparse.Namespace, subject: int, gpu: str) -> None:
    output = args.output_root / "shared_cache_preparation" / f"sub{subject}"
    output.mkdir(parents=True, exist_ok=True)
    values = command(args, subject, "thumb")
    destination_position = values.index("--output") + 1
    values[destination_position] = str(output)
    values.append("--prepare-shared-only")
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = gpu
    subprocess.run(values, env=environment, check=True)


def worker(
    gpu: str,
    tasks: Queue[tuple[int, str]],
    args: argparse.Namespace,
    records: list[dict[str, object]],
    lock: threading.Lock,
) -> None:
    while True:
        try:
            subject, finger = tasks.get_nowait()
        except Empty:
            return
        destination = args.output_root / f"sub{subject}" / finger
        summary = destination / "summary.json"
        log_path = args.output_root / "logs" / f"s{subject}_{finger}.log"
        started = time.perf_counter()
        if completed_summary(summary) and not args.rerun:
            record = {
                "subject": subject,
                "finger": finger,
                "gpu": gpu,
                "status": "skipped_completed",
                "runtime_seconds": 0.0,
            }
        else:
            destination.mkdir(parents=True, exist_ok=True)
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = gpu
            with log_path.open("w") as log:
                process = subprocess.run(
                    command(args, subject, finger),
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            record = {
                "subject": subject,
                "finger": finger,
                "gpu": gpu,
                "status": "finished" if process.returncode == 0 else "failed",
                "returncode": process.returncode,
                "runtime_seconds": time.perf_counter() - started,
                "log": str(log_path),
            }
        with lock:
            records.append(record)
            print(json.dumps(record), flush=True)
        tasks.task_done()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", type=int, nargs="+", default=(1, 2, 3))
    parser.add_argument(
        "--fingers", nargs="+", choices=tuple(FINGER_NAMES), default=tuple(FINGER_NAMES)
    )
    parser.add_argument("--gpus", nargs="+", default=tuple(str(i) for i in range(8)))
    parser.add_argument(
        "--fold-root",
        type=Path,
        default=Path("outputs/event_stratified_folds_fulldev_targetsafe_conservative_v1"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/single_wavelet_1000hz_tap5over2_nested_v2"),
    )
    parser.add_argument(
        "--reuse-from-root",
        type=Path,
        default=None,
        help="reuse completed split-local caches and inner curves from another run",
    )
    parser.add_argument(
        "--initialization-from-root",
        type=Path,
        default=None,
        help="reuse only deterministic split-local target/ICA/CSP/LARS caches",
    )
    parser.add_argument(
        "--reuse-inner-metrics-from-root",
        type=Path,
        default=None,
        help="reuse checkpoint curves only when the recurrent model and schedule are identical",
    )
    parser.add_argument("--head-learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--model-rate", type=int, choices=(400, 1000), default=1000)
    parser.add_argument("--tap-resample-up", type=int, default=5)
    parser.add_argument("--tap-resample-down", type=int, default=2)
    parser.add_argument("--spatial-learning-rate", type=float, default=3.0e-6)
    parser.add_argument("--wavelet-learning-rate", type=float, default=3.0e-6)
    parser.add_argument(
        "--head-initialization",
        choices=("lars_linear_regime",),
        default="lars_linear_regime",
    )
    parser.add_argument("--csp-mode", choices=tuple(CSP_MODES), default="movement_1")
    parser.add_argument("--lars-candidate-scale", type=float, default=1.0)
    parser.add_argument("--lars-near-zero-std", type=float, default=1.0e-3)
    parser.add_argument("--lars-open-gate-bias", type=float, default=5.0)
    parser.add_argument("--lars-forget-gate-bias", type=float, default=-5.0)
    parser.add_argument(
        "--recurrent-cell",
        choices=("standard", "paper_equations"),
        default="standard",
    )
    parser.add_argument("--sequence-steps", type=int, default=100)
    parser.add_argument("--sequence-stride", type=int, default=25)
    parser.add_argument(
        "--sampler-mode",
        choices=("uniform_group", "dense_sequences"),
        default="dense_sequences",
    )
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument(
        "--output-activation", choices=("linear", "softplus"), default="softplus"
    )
    parser.add_argument("--softplus-beta", type=float, default=10.0)
    parser.add_argument(
        "--little-event-decontamination",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--little-event-ratio-low", type=float, default=0.8)
    parser.add_argument("--little-event-ratio-high", type=float, default=1.2)
    parser.add_argument(
        "--selection-metric", choices=("event_macro_nmse", "raw_pcc"), default="raw_pcc"
    )
    parser.add_argument("--selection-rule", choices=("one_se", "best"), default="best")
    parser.add_argument("--purge-bins", type=int, default=95)
    parser.add_argument(
        "--frozen-update-grid", type=int, nargs="+", default=(20, 40, 80, 100, 120, 200)
    )
    parser.add_argument("--unfreeze-after", type=int, default=100)
    parser.add_argument(
        "--unfrozen-update-grid", type=int, nargs="+", default=(20, 40, 80, 120, 200, 400)
    )
    parser.add_argument("--ica-cache-root", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--require-lstm-update", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--queue-summary-name", default="queue_summary.json")
    args = parser.parse_args()

    if args.reuse_from_root is not None and (
        args.initialization_from_root is not None
        or args.reuse_inner_metrics_from_root is not None
    ):
        parser.error(
            "--reuse-from-root cannot be combined with the split cache-reuse options"
        )

    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "logs").mkdir(parents=True, exist_ok=True)
    for index, subject in enumerate(args.subjects):
        prepare_shared(args, subject, args.gpus[index % len(args.gpus)])

    queue: Queue[tuple[int, str]] = Queue()
    for subject in args.subjects:
        for finger in args.fingers:
            queue.put((subject, finger))
    records: list[dict[str, object]] = []
    lock = threading.Lock()
    threads = [
        threading.Thread(target=worker, args=(gpu, queue, args, records, lock))
        for gpu in args.gpus
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    records.sort(key=lambda item: (int(item["subject"]), str(item["finger"])))
    report = {
        "protocol": "one independent single-path ICA+CSP-wavelet-LSTM model per subject/finger",
        "models_requested": len(args.subjects) * len(args.fingers),
        "models_completed": sum(item["status"] in ("finished", "skipped_completed") for item in records),
        "models_failed": sum(item["status"] == "failed" for item in records),
        "settings": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "runs": records,
    }
    (args.output_root / args.queue_summary_name).write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2), flush=True)
    if report["models_failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
