#!/usr/bin/env python3
"""Run development-only spatial-bank gates concurrently across GPUs."""

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

from ecog_decoding.training import FINGER_NAMES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--fingers", nargs="+", choices=tuple(FINGER_NAMES), default=tuple(FINGER_NAMES))
    parser.add_argument("--folds", type=int, nargs="+", default=(0, 1, 2))
    parser.add_argument("--gpus", nargs="+", default=tuple(str(index) for index in range(8)))
    parser.add_argument("--initialization-root", type=Path, required=True)
    parser.add_argument("--little-initialization-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-rate", type=int, choices=(400, 1000), default=1000)
    parser.add_argument("--tap-resample-up", type=int, default=5)
    parser.add_argument("--tap-resample-down", type=int, default=2)
    parser.add_argument("--cpu-threads-per-job", type=int, default=4)
    args = parser.parse_args()

    tasks: Queue[tuple[str, int]] = Queue()
    for finger in args.fingers:
        for fold in args.folds:
            summary = args.output_root / finger / f"fold{fold}" / "summary.json"
            if not summary.exists():
                tasks.put((finger, fold))

    records: list[dict[str, object]] = []
    lock = threading.Lock()
    log_root = args.output_root / "logs"
    log_root.mkdir(parents=True, exist_ok=True)

    def worker(gpu: str) -> None:
        while True:
            try:
                finger, fold = tasks.get_nowait()
            except Empty:
                return
            started = time.perf_counter()
            initialization = (
                args.little_initialization_root
                if finger == "little" and args.little_initialization_root is not None
                else args.initialization_root
            )
            output = args.output_root / finger / f"fold{fold}"
            output.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable,
                "scripts/ablate_spatial_csp_count.py",
                "--subject", str(args.subject),
                "--finger", finger,
                "--outer-folds", str(fold),
                "--model-rate", str(args.model_rate),
                "--tap-resample-up", str(args.tap_resample_up),
                "--tap-resample-down", str(args.tap_resample_down),
                "--initialization-root", str(initialization),
                "--output", str(output),
                "--device", "cuda",
            ]
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = gpu
            environment["PYTHONPATH"] = "experiments/scripts:scripts:src"
            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
                environment[name] = str(args.cpu_threads_per_job)
            log_path = log_root / f"{finger}_fold{fold}.log"
            with log_path.open("w") as handle:
                process = subprocess.run(command, env=environment, stdout=handle, stderr=subprocess.STDOUT)
            record = {
                "finger": finger,
                "fold": fold,
                "gpu": gpu,
                "returncode": process.returncode,
                "runtime_seconds": time.perf_counter() - started,
                "log": str(log_path),
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
    records.sort(key=lambda item: (str(item["finger"]), int(item["fold"])))
    report = {"subject": args.subject, "runs": records}
    (args.output_root / "queue_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    if any(int(record["returncode"]) != 0 for record in records):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
