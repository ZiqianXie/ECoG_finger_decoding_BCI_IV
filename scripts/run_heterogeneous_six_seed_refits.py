#!/usr/bin/env python3
"""Run the selected heterogeneous six-seed refits efficiently across GPUs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/heterogeneous_six_seed_refit.yaml"))
    parser.add_argument("--gpus", nargs="+", default=tuple(str(index) for index in range(8)))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/heterogeneous_six_seed_refit_v1"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    seeds = [int(seed) for seed in config["seeds"]]
    tasks: list[tuple[str, list[str]]] = []
    for subject, fingers in config["targets"].items():
        for finger in fingers:
            for seed in seeds:
                destination = args.output_root / f"sub{subject}" / finger / f"seed{seed}" / "summary.json"
                if destination.exists():
                    continue
                name = f"s{subject}_{finger}_seed{seed}"
                tasks.append((name, [
                    sys.executable, "scripts/refit_heterogeneous_fixed_lstm.py",
                    "--subject", str(subject), "--finger", finger, "--seed", str(seed),
                    "--output-root", str(args.output_root),
                ]))
    logs = args.output_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    active: list[tuple[subprocess.Popen[bytes], object, str, str]] = []
    failures: list[str] = []
    while tasks or active:
        busy = {gpu for _, _, _, gpu in active}
        for gpu in args.gpus:
            if not tasks or gpu in busy:
                continue
            name, command = tasks.pop(0)
            log = open(logs / f"{name}.log", "wb")
            environment = os.environ.copy()
            environment["PYTHONPATH"] = "scripts:src"
            environment["CUDA_VISIBLE_DEVICES"] = gpu
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment)
            active.append((process, log, name, gpu))
            print(f"started {name} pid={process.pid} gpu={gpu}", flush=True)
        time.sleep(1.0)
        remaining = []
        for process, log, name, gpu in active:
            code = process.poll()
            if code is None:
                remaining.append((process, log, name, gpu))
                continue
            log.close()
            print(f"finished {name} exit={code}", flush=True)
            if code:
                failures.append(name)
        active = remaining
    if failures:
        raise RuntimeError("failed tasks: " + ", ".join(failures))


if __name__ == "__main__":
    main()
