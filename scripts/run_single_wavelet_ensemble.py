#!/usr/bin/env python3
"""Run six fixed-configuration single-path refits for all 15 pairs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

from ecog_decoding.training import FINGER_NAMES
from cross_validate_single_wavelet import CSP_MODES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", type=int, nargs="+", default=(1, 2, 3))
    parser.add_argument(
        "--fingers", nargs="+", choices=tuple(FINGER_NAMES), default=tuple(FINGER_NAMES)
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 2, 3, 4, 5))
    parser.add_argument("--gpus", nargs="+", default=tuple(str(i) for i in range(8)))
    parser.add_argument(
        "--selection-root",
        type=Path,
        default=Path("outputs/single_wavelet_1000hz_tap5over2_nested_v2"),
    )
    parser.add_argument(
        "--initialization-root",
        type=Path,
        default=Path("outputs/single_wavelet_1000hz_tap5over2_full_refit_v1"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/single_wavelet_1000hz_tap5over2_six_seed_refit_v1"),
    )
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--force-schedule", default=None)
    parser.add_argument(
        "--head-initialization",
        choices=("lars_linear_regime",),
        default="lars_linear_regime",
    )
    parser.add_argument("--head-learning-rate", type=float, default=None)
    parser.add_argument("--model-rate", type=int, choices=(400, 1000), default=1000)
    parser.add_argument("--tap-resample-up", type=int, default=5)
    parser.add_argument("--tap-resample-down", type=int, default=2)
    parser.add_argument("--spatial-learning-rate", type=float, default=None)
    parser.add_argument("--wavelet-learning-rate", type=float, default=None)
    parser.add_argument("--lars-candidate-scale", type=float, default=1.0)
    parser.add_argument("--lars-near-zero-std", type=float, default=1.0e-3)
    parser.add_argument("--lars-open-gate-bias", type=float, default=5.0)
    parser.add_argument("--lars-forget-gate-bias", type=float, default=-5.0)
    parser.add_argument(
        "--recurrent-cell",
        choices=("standard", "paper_equations", "residual_lstm", "residual_gru"),
        default="standard",
    )
    parser.add_argument("--sequence-steps", type=int, default=100)
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
    parser.add_argument("--csp-mode", choices=tuple(CSP_MODES), default="movement_1")
    parser.add_argument("--ica-initialization-root", type=Path, default=None)
    parser.add_argument(
        "--selection-metric",
        choices=("event_macro_nmse", "raw_pcc"),
        default="raw_pcc",
    )
    parser.add_argument(
        "--selection-rule", choices=("one_se", "best"), default="best"
    )
    parser.add_argument(
        "--sampler-mode",
        choices=("uniform_group", "dense_sequences"),
        default="dense_sequences",
    )
    parser.add_argument("--sequence-stride", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--queue-summary-name", default="queue_summary.json")
    args = parser.parse_args()

    tasks: list[tuple[str, list[str]]] = []
    for subject in args.subjects:
        for finger in args.fingers:
            for seed in args.seeds:
                destination = (
                    args.output_root
                    / f"sub{subject}"
                    / finger
                    / f"seed{seed}"
                    / "summary.json"
                )
                if destination.exists() and not args.rerun:
                    continue
                name = f"s{subject}_{finger}_seed{seed}"
                tasks.append(
                    (
                        name,
                        [
                            sys.executable,
                            "scripts/refit_single_wavelet.py",
                            "--subject",
                            str(subject),
                            "--finger",
                            finger,
                            "--seed",
                            str(seed),
                            "--selection-root",
                            str(args.selection_root),
                            "--initialization-root",
                            str(args.initialization_root),
                            "--output-root",
                            str(args.output_root),
                            "--model-rate",
                            str(args.model_rate),
                            "--tap-resample-up",
                            str(args.tap_resample_up),
                            "--tap-resample-down",
                            str(args.tap_resample_down),
                            "--seed-subdirectory",
                            "--head-initialization",
                            args.head_initialization,
                            "--recurrent-cell",
                            args.recurrent_cell,
                            "--sequence-steps",
                            str(args.sequence_steps),
                            "--output-activation",
                            args.output_activation,
                            "--softplus-beta",
                            str(args.softplus_beta),
                            "--little-event-ratio-low",
                            str(args.little_event_ratio_low),
                            "--little-event-ratio-high",
                            str(args.little_event_ratio_high),
                            "--csp-mode",
                            args.csp_mode,
                            "--selection-metric",
                            args.selection_metric,
                            "--selection-rule",
                            args.selection_rule,
                            "--sampler-mode",
                            args.sampler_mode,
                            "--sequence-stride",
                            str(args.sequence_stride),
                            "--batch-size",
                            str(args.batch_size),
                            "--lars-candidate-scale",
                            str(args.lars_candidate_scale),
                            "--lars-near-zero-std",
                            str(args.lars_near_zero_std),
                            "--lars-open-gate-bias",
                            str(args.lars_open_gate_bias),
                            "--lars-forget-gate-bias",
                            str(args.lars_forget_gate_bias),
                        ],
                    )
                )
                if args.force_schedule is not None:
                    tasks[-1][1].extend(["--force-schedule", args.force_schedule])
                if args.ica_initialization_root is not None:
                    tasks[-1][1].extend(
                        ["--ica-initialization-root", str(args.ica_initialization_root)]
                    )
                if args.little_event_decontamination:
                    tasks[-1][1].append("--little-event-decontamination")
                if not args.compile:
                    tasks[-1][1].append("--no-compile")
                for option, value in (
                    ("--head-learning-rate", args.head_learning_rate),
                    ("--spatial-learning-rate", args.spatial_learning_rate),
                    ("--wavelet-learning-rate", args.wavelet_learning_rate),
                ):
                    if value is not None:
                        tasks[-1][1].extend([option, str(value)])

    logs = args.output_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    active: list[tuple[subprocess.Popen[bytes], object, str, str, float]] = []
    records: list[dict[str, object]] = []
    failures: list[str] = []
    while tasks or active:
        busy = {gpu for _, _, _, gpu, _ in active}
        for gpu in args.gpus:
            if not tasks or gpu in busy:
                continue
            name, command = tasks.pop(0)
            log = open(logs / f"{name}.log", "wb")
            environment = os.environ.copy()
            environment["PYTHONPATH"] = "scripts:src"
            environment["CUDA_VISIBLE_DEVICES"] = gpu
            started = time.perf_counter()
            process = subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT, env=environment
            )
            active.append((process, log, name, gpu, started))
            print(f"started {name} pid={process.pid} gpu={gpu}", flush=True)
        time.sleep(0.25)
        remaining = []
        for process, log, name, gpu, started in active:
            code = process.poll()
            if code is None:
                remaining.append((process, log, name, gpu, started))
                continue
            log.close()
            runtime = time.perf_counter() - started
            records.append(
                {"name": name, "gpu": gpu, "returncode": code, "runtime_seconds": runtime}
            )
            print(f"finished {name} exit={code} seconds={runtime:.2f}", flush=True)
            if code:
                failures.append(name)
        active = remaining

    import json

    report = {
        "models_requested": len(args.subjects) * len(args.fingers) * len(args.seeds),
        "models_completed": sum(item["returncode"] == 0 for item in records),
        "models_failed": len(failures),
        "runs": sorted(records, key=lambda item: str(item["name"])),
    }
    (args.output_root / args.queue_summary_name).write_text(
        json.dumps(report, indent=2) + "\n"
    )
    if failures:
        raise RuntimeError("failed tasks: " + ", ".join(failures))


if __name__ == "__main__":
    main()
