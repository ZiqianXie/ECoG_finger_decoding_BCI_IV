#!/usr/bin/env python3
"""Cross-validate lower LSTM learning rates on selected final-model routes."""

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


DEFAULT_PAIRS = ("1:index", "1:middle", "3:index")


def parse_pair(value: str) -> tuple[int, str]:
    subject, separator, finger = value.partition(":")
    if not separator or not subject.isdigit() or not finger:
        raise argparse.ArgumentTypeError("pairs must have the form SUBJECT:FINGER")
    return int(subject), finger


def learning_rate_label(value: float) -> str:
    return f"lr{value:.0e}".replace("e-0", "e-").replace("e+0", "e+")


def scaled_update_grid(
    base_grid: tuple[int, ...], reference_rate: float, learning_rate: float
) -> tuple[int, ...]:
    scale = reference_rate / learning_rate
    return tuple(sorted({max(1, int(round(value * scale))) for value in base_grid}))


def resolve_route(
    route_map: dict[str, object], subject: int, finger: str
) -> dict[str, object]:
    route = dict(route_map["default"])
    subjects = route_map.get("subjects", {})
    subject_routes = subjects.get(subject, subjects.get(str(subject), {}))
    route.update(subject_routes.get(finger, {}))
    return route


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--route-map",
        type=Path,
        default=Path("experiments/configs/final_lstm_lr_baseline_routes.yaml"),
    )
    parser.add_argument("--pairs", type=parse_pair, nargs="+", default=DEFAULT_PAIRS)
    parser.add_argument(
        "--learning-rates", type=float, nargs="+", default=(1e-4, 3e-5, 1e-5)
    )
    parser.add_argument(
        "--base-update-grid", type=int, nargs="+", default=(5, 10, 20, 40, 80, 120, 200)
    )
    parser.add_argument("--reference-learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--gpus", nargs="+", default=tuple(str(index) for index in range(8))
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/final_single_wavelet_lstm_lr_sweep_v2"),
    )
    parser.add_argument(
        "--reuse-sweep-root",
        type=Path,
        default=None,
        help="reuse completed inner-fold metrics from a matched earlier sweep",
    )
    parser.add_argument(
        "--fold-root",
        type=Path,
        default=Path(
            "outputs/event_stratified_folds_fulldev_targetsafe_conservative_v1"
        ),
    )
    parser.add_argument("--summary-name", default="queue_summary.json")
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument(
        "--compile", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()

    route_map = yaml.safe_load(args.route_map.read_text())
    pairs = [
        parse_pair(value) if isinstance(value, str) else value for value in args.pairs
    ]
    base_grid = tuple(sorted({int(value) for value in args.base_update_grid}))
    if any(value <= 0 for value in base_grid):
        raise ValueError("base update checkpoints must be positive")
    if any(value <= 0 for value in args.learning_rates):
        raise ValueError("learning rates must be positive")

    args.output_root.mkdir(parents=True, exist_ok=True)
    log_root = args.output_root / "logs"
    log_root.mkdir(parents=True, exist_ok=True)

    # Each host builds its own /dev/shm leaf cache once before workers start.
    for subject in sorted({subject for subject, _ in pairs}):
        finger = next(finger for candidate, finger in pairs if candidate == subject)
        route = resolve_route(route_map, subject, finger)
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(args.gpus[0])
        command = [
            sys.executable,
            "scripts/cross_validate_single_wavelet.py",
            "--subject",
            str(subject),
            "--finger",
            finger,
            "--model-rate",
            str(route.get("model_rate_hz", 1000)),
            "--tap-resample-up",
            str(route.get("tap_resample_up", 5)),
            "--tap-resample-down",
            str(route.get("tap_resample_down", 2)),
            "--output",
            str(args.output_root / "prewarm" / f"sub{subject}"),
            "--prepare-shared-only",
        ]
        subprocess.run(command, env=environment, check=True)

    tasks: Queue[tuple[int, str, float]] = Queue()
    for subject, finger in pairs:
        for learning_rate in args.learning_rates:
            tasks.put((subject, finger, learning_rate))

    records: list[dict[str, object]] = []
    lock = threading.Lock()

    def worker(gpu: str) -> None:
        while True:
            try:
                subject, finger, learning_rate = tasks.get_nowait()
            except Empty:
                return
            route = resolve_route(route_map, subject, finger)
            label = learning_rate_label(learning_rate)
            output = args.output_root / label / f"sub{subject}" / finger
            started = time.perf_counter()
            if (output / "summary.json").exists() and not args.rerun:
                returncode = 0
                status = "skipped_completed"
            else:
                update_grid = scaled_update_grid(
                    base_grid, args.reference_learning_rate, learning_rate
                )
                selection_pair = (
                    Path(str(route["selection_root"])) / f"sub{subject}" / finger
                )
                command = [
                    sys.executable,
                    "scripts/cross_validate_single_wavelet.py",
                    "--subject",
                    str(subject),
                    "--finger",
                    finger,
                    "--fold-root",
                    str(args.fold_root),
                    "--output",
                    str(output),
                    "--initialization-cache-root",
                    str(selection_pair),
                    "--model-rate",
                    str(route.get("model_rate_hz", 1000)),
                    "--tap-resample-up",
                    str(route.get("tap_resample_up", 5)),
                    "--tap-resample-down",
                    str(route.get("tap_resample_down", 2)),
                    "--recurrent-cell",
                    str(route.get("recurrent_cell", "standard")),
                    "--csp-mode",
                    str(route.get("csp_mode", "movement_1")),
                    "--head-learning-rate",
                    str(learning_rate),
                    "--spatial-learning-rate",
                    "3e-6",
                    "--wavelet-learning-rate",
                    "3e-6",
                    "--lars-candidate-scale",
                    "1",
                    "--sequence-steps",
                    "100",
                    "--sequence-stride",
                    "25",
                    "--batch-size",
                    "24",
                    "--sampler-mode",
                    "dense_sequences",
                    "--selection-metric",
                    "raw_pcc",
                    "--selection-rule",
                    "best",
                    "--no-require-lstm-update",
                    "--output-activation",
                    "softplus",
                    "--softplus-beta",
                    "10",
                    "--frozen-update-grid",
                    *[str(value) for value in update_grid],
                    "--frozen-only",
                ]
                if args.reuse_sweep_root is not None:
                    reused = (
                        args.reuse_sweep_root
                        / label
                        / f"sub{subject}"
                        / finger
                    )
                    command.extend(["--reuse-inner-metrics-from", str(reused)])
                if route.get("little_event_decontamination", False):
                    command.append("--little-event-decontamination")
                if not args.compile:
                    command.append("--no-compile")
                environment = os.environ.copy()
                environment["CUDA_VISIBLE_DEVICES"] = gpu
                log_path = log_root / f"{label}_s{subject}_{finger}.log"
                with log_path.open("w") as log:
                    process = subprocess.run(
                        command, env=environment, stdout=log, stderr=subprocess.STDOUT
                    )
                returncode = process.returncode
                status = "finished" if returncode == 0 else "failed"
            record = {
                "subject": subject,
                "finger": finger,
                "learning_rate": learning_rate,
                "update_grid": list(
                    scaled_update_grid(
                        base_grid, args.reference_learning_rate, learning_rate
                    )
                ),
                "gpu": gpu,
                "status": status,
                "returncode": returncode,
                "runtime_seconds": time.perf_counter() - started,
                "output": str(output),
            }
            with lock:
                records.append(record)
                print(json.dumps(record), flush=True)
            tasks.task_done()

    workers = [threading.Thread(target=worker, args=(gpu,)) for gpu in args.gpus]
    for worker_thread in workers:
        worker_thread.start()
    for worker_thread in workers:
        worker_thread.join()

    records.sort(
        key=lambda item: (
            int(item["subject"]),
            str(item["finger"]),
            -float(item["learning_rate"]),
        )
    )
    report = {
        "protocol": (
            "development-only nested event-fold comparison of LARS initialization "
            "versus frozen-stem LSTM tuning; released test is untouched"
        ),
        "pairs": [f"{subject}:{finger}" for subject, finger in pairs],
        "reference_learning_rate": args.reference_learning_rate,
        "base_update_grid": list(base_grid),
        "reuse_sweep_root": (
            str(args.reuse_sweep_root) if args.reuse_sweep_root is not None else None
        ),
        "models_requested": len(pairs) * len(args.learning_rates),
        "models_completed": sum(record["returncode"] == 0 for record in records),
        "models_failed": sum(record["returncode"] != 0 for record in records),
        "runs": records,
    }
    (args.output_root / args.summary_name).write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2), flush=True)
    if report["models_failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
