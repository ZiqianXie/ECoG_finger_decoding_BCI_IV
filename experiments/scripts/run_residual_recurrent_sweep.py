#!/usr/bin/env python3
"""Nested-OOF screen of residual LSTM/GRU heads on the single wavelet stem."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from queue import Empty, Queue
import subprocess
import sys
import threading
import time

import yaml


DEFAULT_PAIRS = ("1:middle", "2:middle", "3:index", "3:little")


def parse_pair(value: str) -> tuple[int, str]:
    subject, separator, finger = value.partition(":")
    if not separator or not subject.isdigit() or not finger:
        raise argparse.ArgumentTypeError("pairs must have the form SUBJECT:FINGER")
    return int(subject), finger


def rate_label(value: float) -> str:
    return f"lr{value:.0e}".replace("e-0", "e-").replace("e+0", "e+")


def resolve_route(
    route_map: dict[str, object], subject: int, finger: str
) -> dict[str, object]:
    route = dict(route_map["default"])
    subjects = route_map.get("subjects", {})
    subject_routes = subjects.get(subject, subjects.get(str(subject), {}))
    route.update(subject_routes.get(finger, {}))
    return route


def scaled_grid(
    base: tuple[int, ...], reference_rate: float, learning_rate: float
) -> tuple[int, ...]:
    scale = reference_rate / learning_rate
    return tuple(sorted({max(1, int(round(value * scale))) for value in base}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--route-map",
        type=Path,
        default=Path("experiments/configs/final_lstm_lr_baseline_routes.yaml"),
    )
    parser.add_argument("--pairs", type=parse_pair, nargs="+", default=DEFAULT_PAIRS)
    parser.add_argument(
        "--families", nargs="+", choices=("residual_lstm", "residual_gru"),
        default=("residual_lstm", "residual_gru")
    )
    parser.add_argument(
        "--learning-rates", type=float, nargs="+", default=(3e-4, 1e-4)
    )
    parser.add_argument(
        "--base-update-grid", type=int, nargs="+", default=(20, 50, 100, 200, 400)
    )
    parser.add_argument("--reference-learning-rate", type=float, default=3e-4)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--gpus", nargs="+", default=tuple(str(i) for i in range(8)))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/residual_recurrent_single_wavelet_oof_v1"),
    )
    parser.add_argument(
        "--fold-root",
        type=Path,
        default=Path(
            "outputs/event_stratified_folds_fulldev_targetsafe_conservative_v1"
        ),
    )
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument(
        "--compile", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()

    pairs = [
        parse_pair(pair) if isinstance(pair, str) else pair for pair in args.pairs
    ]
    rates = tuple(sorted({float(rate) for rate in args.learning_rates}, reverse=True))
    families = tuple(dict.fromkeys(args.families))
    base_grid = tuple(sorted({int(value) for value in args.base_update_grid}))
    if not rates or any(rate <= 0 for rate in rates):
        raise ValueError("learning rates must be positive")
    if not base_grid or any(value <= 0 for value in base_grid):
        raise ValueError("update checkpoints must be positive")
    if not args.gpus:
        raise ValueError("at least one GPU is required")

    route_map = yaml.safe_load(args.route_map.read_text())
    args.output_root.mkdir(parents=True, exist_ok=True)
    log_root = args.output_root / "logs"
    log_root.mkdir(parents=True, exist_ok=True)

    # Prepare shared linear wavelet leaves once per host and subject.
    for subject in sorted({subject for subject, _ in pairs}):
        finger = next(finger for candidate, finger in pairs if candidate == subject)
        route = resolve_route(route_map, subject, finger)
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(args.gpus[0])
        subprocess.run(
            [
                sys.executable,
                "scripts/cross_validate_single_wavelet.py",
                "--subject", str(subject),
                "--finger", finger,
                "--model-rate", str(route.get("model_rate_hz", 1000)),
                "--tap-resample-up", str(route.get("tap_resample_up", 5)),
                "--tap-resample-down", str(route.get("tap_resample_down", 2)),
                "--output", str(args.output_root / "prewarm" / f"sub{subject}"),
                "--prepare-shared-only",
            ],
            env=environment,
            check=True,
        )

    primary_family = families[0]
    primary_rate = rates[0]
    records: list[dict[str, object]] = []
    record_lock = threading.Lock()

    def destination(subject: int, finger: str, family: str, rate: float) -> Path:
        return args.output_root / family / rate_label(rate) / f"sub{subject}" / finger

    def run_phase(tasks: list[tuple[int, str, str, float, Path | None]]) -> None:
        queue: Queue[tuple[int, str, str, float, Path | None]] = Queue()
        for task in tasks:
            queue.put(task)

        def worker(gpu: str) -> None:
            while True:
                try:
                    subject, finger, family, rate, cache_root = queue.get_nowait()
                except Empty:
                    return
                output = destination(subject, finger, family, rate)
                started = time.perf_counter()
                if (output / "summary.json").exists() and not args.rerun:
                    code = 0
                    status = "skipped_completed"
                else:
                    route = resolve_route(route_map, subject, finger)
                    update_grid = scaled_grid(
                        base_grid, args.reference_learning_rate, rate
                    )
                    command = [
                        sys.executable,
                        "scripts/cross_validate_single_wavelet.py",
                        "--subject", str(subject),
                        "--finger", finger,
                        "--fold-root", str(args.fold_root),
                        "--output", str(output),
                        "--model-rate", str(route.get("model_rate_hz", 1000)),
                        "--tap-resample-up", str(route.get("tap_resample_up", 5)),
                        "--tap-resample-down", str(route.get("tap_resample_down", 2)),
                        "--recurrent-cell", family,
                        "--csp-mode", str(route.get("csp_mode", "movement_1")),
                        "--hidden-size", str(args.hidden_size),
                        "--head-learning-rate", str(rate),
                        "--spatial-learning-rate", "0",
                        "--wavelet-learning-rate", "0",
                        "--lars-candidate-scale", "1",
                        "--sequence-steps", "100",
                        "--sequence-stride", "25",
                        "--batch-size", "24",
                        "--sampler-mode", "dense_sequences",
                        "--selection-metric", "raw_pcc",
                        "--selection-rule", "best",
                        "--no-require-lstm-update",
                        "--output-activation", "softplus",
                        "--softplus-beta", "10",
                        "--frozen-update-grid", *[str(value) for value in update_grid],
                        "--frozen-only",
                    ]
                    if cache_root is not None:
                        command.extend(["--initialization-cache-root", str(cache_root)])
                    if route.get("little_event_decontamination", False):
                        command.append("--little-event-decontamination")
                    if not args.compile:
                        command.append("--no-compile")
                    environment = os.environ.copy()
                    environment["CUDA_VISIBLE_DEVICES"] = gpu
                    log_path = log_root / f"{family}_{rate_label(rate)}_s{subject}_{finger}.log"
                    with log_path.open("w") as log:
                        code = subprocess.run(
                            command,
                            env=environment,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                        ).returncode
                    status = "finished" if code == 0 else "failed"
                record = {
                    "subject": subject,
                    "finger": finger,
                    "family": family,
                    "learning_rate": rate,
                    "gpu": gpu,
                    "status": status,
                    "returncode": code,
                    "runtime_seconds": time.perf_counter() - started,
                    "output": str(output),
                }
                with record_lock:
                    records.append(record)
                    print(json.dumps(record), flush=True)
                queue.task_done()

        workers = [threading.Thread(target=worker, args=(gpu,)) for gpu in args.gpus]
        for thread in workers:
            thread.start()
        for thread in workers:
            thread.join()

    primary_tasks = [
        (subject, finger, primary_family, primary_rate, None)
        for subject, finger in pairs
    ]
    run_phase(primary_tasks)
    failed_primary = [
        record
        for record in records
        if record["family"] == primary_family
        and record["learning_rate"] == primary_rate
        and record["returncode"] != 0
    ]
    if failed_primary:
        raise SystemExit("primary cache-building phase failed")

    remaining = []
    for subject, finger in pairs:
        cache = destination(subject, finger, primary_family, primary_rate)
        for family in families:
            for rate in rates:
                if family == primary_family and rate == primary_rate:
                    continue
                remaining.append((subject, finger, family, rate, cache))
    run_phase(remaining)

    records.sort(
        key=lambda item: (
            int(item["subject"]), str(item["finger"]),
            str(item["family"]), -float(item["learning_rate"]),
        )
    )
    report = {
        "protocol": (
            "nested event-fold OOF residual-head screen on one shared "
            "ICA+CSP-wavelet convolutional stem; released test untouched"
        ),
        "pairs": [f"{subject}:{finger}" for subject, finger in pairs],
        "families": list(families),
        "learning_rates": list(rates),
        "hidden_size": args.hidden_size,
        "models_requested": len(pairs) * len(families) * len(rates),
        "models_completed": sum(record["returncode"] == 0 for record in records),
        "models_failed": sum(record["returncode"] != 0 for record in records),
        "runs": records,
    }
    (args.output_root / "queue_summary.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    if report["models_failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
