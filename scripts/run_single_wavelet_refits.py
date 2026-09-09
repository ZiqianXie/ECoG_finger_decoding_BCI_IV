#!/usr/bin/env python3
"""Queue the 15 full-development single-path refits across GPUs."""

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", type=int, nargs="+", default=(1, 2, 3))
    parser.add_argument(
        "--fingers", nargs="+", choices=tuple(FINGER_NAMES), default=tuple(FINGER_NAMES)
    )
    parser.add_argument("--gpus", nargs="+", default=tuple(str(i) for i in range(8)))
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
        "--head-initialization", choices=("lars_linear_regime",), default="lars_linear_regime"
    )
    parser.add_argument("--head-learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--model-rate", type=int, choices=(400, 1000), default=1000)
    parser.add_argument("--tap-resample-up", type=int, default=5)
    parser.add_argument("--tap-resample-down", type=int, default=2)
    parser.add_argument("--spatial-learning-rate", type=float, default=3.0e-6)
    parser.add_argument("--wavelet-learning-rate", type=float, default=3.0e-6)
    parser.add_argument("--lars-candidate-scale", type=float, default=1.0)
    parser.add_argument("--sequence-steps", type=int, default=100)
    parser.add_argument("--sequence-stride", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--csp-mode", choices=tuple(CSP_MODES), default="movement_1")
    parser.add_argument(
        "--output-activation", choices=("linear", "softplus"), default="softplus"
    )
    parser.add_argument("--softplus-beta", type=float, default=10.0)
    parser.add_argument(
        "--sampler-mode", choices=("uniform_group", "dense_sequences"), default="dense_sequences"
    )
    parser.add_argument(
        "--selection-metric", choices=("event_macro_nmse", "raw_pcc"), default="raw_pcc"
    )
    parser.add_argument("--selection-rule", choices=("one_se", "best"), default="best")
    parser.add_argument("--ica-initialization-root", type=Path, default=None)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--queue-summary-name", default="queue_summary.json")
    args = parser.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)
    log_root = args.output_root / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    # The 5 finger models of a subject share resampled ECoG and fixed wavelet
    # leaves. Build these once so workers never race while publishing a cache.
    for subject in args.subjects:
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(args.gpus[0])
        subprocess.run(
            [
                sys.executable,
                "scripts/refit_single_wavelet.py",
                "--subject",
                str(subject),
                "--finger",
                str(args.fingers[0]),
                "--selection-root",
                str(args.selection_root),
                "--output-root",
                str(args.output_root),
                "--model-rate",
                str(args.model_rate),
                "--tap-resample-up",
                str(args.tap_resample_up),
                "--tap-resample-down",
                str(args.tap_resample_down),
                "--prepare-shared-only",
            ],
            env=environment,
            check=True,
        )
    tasks: Queue[tuple[int, str]] = Queue()
    for subject in args.subjects:
        for finger in args.fingers:
            tasks.put((subject, finger))
    records: list[dict[str, object]] = []
    lock = threading.Lock()

    def worker(gpu: str) -> None:
        while True:
            try:
                subject, finger = tasks.get_nowait()
            except Empty:
                return
            destination = args.output_root / f"sub{subject}" / finger
            started = time.perf_counter()
            if (destination / "summary.json").exists() and not args.rerun:
                status = "skipped_completed"
                returncode = 0
            else:
                environment = os.environ.copy()
                environment["CUDA_VISIBLE_DEVICES"] = gpu
                command = [
                    sys.executable,
                    "scripts/refit_single_wavelet.py",
                    "--subject",
                    str(subject),
                    "--finger",
                    finger,
                    "--selection-root",
                    str(args.selection_root),
                    "--output-root",
                    str(args.output_root),
                    "--model-rate",
                    str(args.model_rate),
                    "--tap-resample-up",
                    str(args.tap_resample_up),
                    "--tap-resample-down",
                    str(args.tap_resample_down),
                    "--head-initialization",
                    args.head_initialization,
                    "--head-learning-rate",
                    str(args.head_learning_rate),
                    "--spatial-learning-rate",
                    str(args.spatial_learning_rate),
                    "--wavelet-learning-rate",
                    str(args.wavelet_learning_rate),
                    "--lars-candidate-scale",
                    str(args.lars_candidate_scale),
                    "--sequence-steps",
                    str(args.sequence_steps),
                    "--sequence-stride",
                    str(args.sequence_stride),
                    "--batch-size",
                    str(args.batch_size),
                    "--csp-mode",
                    args.csp_mode,
                    "--output-activation",
                    args.output_activation,
                    "--softplus-beta",
                    str(args.softplus_beta),
                    "--sampler-mode",
                    args.sampler_mode,
                    "--selection-metric",
                    args.selection_metric,
                    "--selection-rule",
                    args.selection_rule,
                ]
                if args.ica_initialization_root is not None:
                    command.extend(
                        ["--ica-initialization-root", str(args.ica_initialization_root)]
                    )
                if not args.compile:
                    command.append("--no-compile")
                with (log_root / f"s{subject}_{finger}.log").open("w") as log:
                    process = subprocess.run(
                        command,
                        env=environment,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                returncode = process.returncode
                status = "finished" if returncode == 0 else "failed"
            record = {
                "subject": subject,
                "finger": finger,
                "gpu": gpu,
                "status": status,
                "returncode": returncode,
                "runtime_seconds": time.perf_counter() - started,
            }
            with lock:
                records.append(record)
                print(json.dumps(record), flush=True)
            tasks.task_done()

    threads = [threading.Thread(target=worker, args=(gpu,)) for gpu in args.gpus]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    records.sort(key=lambda item: (int(item["subject"]), str(item["finger"])))
    report = {
        "models_requested": len(args.subjects) * len(args.fingers),
        "models_completed": sum(item["returncode"] == 0 for item in records),
        "models_failed": sum(item["returncode"] != 0 for item in records),
        "runs": records,
    }
    (args.output_root / args.queue_summary_name).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if report["models_failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
