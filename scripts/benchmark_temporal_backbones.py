#!/usr/bin/env python3
"""Benchmark eager and torch-compiled GRU/diagonal-SSM decoders."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from ecog_decoding.models import EcogTrajectoryDecoder


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--channels", type=int, default=61)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--output-steps", type=int, default=100)
    parser.add_argument("--history-bins", type=int, default=25)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--inference-steps", type=int, default=20)
    parser.add_argument(
        "--compile-mode",
        choices=("default", "reduce-overhead", "max-autotune"),
        default="reduce-overhead",
    )
    parser.add_argument(
        "--matmul-precision",
        choices=("highest", "high", "medium"),
        default="high",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/benchmarks/torch_compile_a100.json"),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    torch.set_float32_matmul_precision(args.matmul_precision)
    device = torch.device(args.device)
    energy_bins = args.output_steps + args.history_bins - 1
    input_samples = 40 * energy_bins
    results: list[dict[str, object]] = []

    if device.type == "cuda":
        # Pay CUDA context initialization once, outside every measured variant.
        torch.randn(256, 256, device=device, requires_grad=True).square().sum().backward()
        synchronize(device)

    for backbone in ("gru", "diagonal_ssm"):
        for compiled in (False, True):
            torch.manual_seed(0)
            model = EcogTrajectoryDecoder(
                input_channels=args.channels,
                spatial_components=args.channels,
                feature_width=64,
                temporal_backbone=backbone,
                temporal_layers=3,
                state_size=16,
                dropout=0.1,
                output_fingers=5,
                cnn_history_bins=args.history_bins,
            ).to(device)
            executable = (
                torch.compile(model, mode=args.compile_mode) if compiled else model
            )
            optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
            ecog = torch.randn(
                args.batch_size, args.channels, input_samples, device=device
            )
            target = torch.rand(
                args.batch_size, args.output_steps, 5, device=device
            )

            def step() -> float:
                optimizer.zero_grad(set_to_none=True)
                prediction = executable(ecog)
                loss = F.mse_loss(prediction, target)
                loss.backward()
                optimizer.step()
                return float(loss.detach())

            synchronize(device)
            cold_started = time.perf_counter()
            loss = step()
            synchronize(device)
            cold_training_ms = 1000.0 * (time.perf_counter() - cold_started)
            for _ in range(args.warmup):
                loss = step()
            synchronize(device)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            for _ in range(args.steps):
                loss = step()
            synchronize(device)
            training_elapsed = time.perf_counter() - started
            peak_memory_mb = (
                torch.cuda.max_memory_allocated(device) / 2**20
                if device.type == "cuda"
                else None
            )

            executable.eval()
            with torch.inference_mode():
                synchronize(device)
                cold_started = time.perf_counter()
                prediction = executable(ecog)
                synchronize(device)
                cold_inference_ms = 1000.0 * (time.perf_counter() - cold_started)
                for _ in range(args.warmup):
                    prediction = executable(ecog)
                synchronize(device)
                started = time.perf_counter()
                for _ in range(args.inference_steps):
                    prediction = executable(ecog)
                synchronize(device)
                inference_elapsed = time.perf_counter() - started

            results.append(
                {
                    "backbone": backbone,
                    "execution": "compiled" if compiled else "eager",
                    "compile_mode": args.compile_mode if compiled else None,
                    "parameters": sum(
                        parameter.numel() for parameter in model.parameters()
                    ),
                    "batch_size": args.batch_size,
                    "input_samples_1khz": input_samples,
                    "output_steps_25hz": args.output_steps,
                    "cold_training_step_ms": cold_training_ms,
                    "milliseconds_per_training_step": (
                        1000.0 * training_elapsed / args.steps
                    ),
                    "training_output_sequences_per_second": (
                        args.batch_size * args.steps / training_elapsed
                    ),
                    "cold_inference_call_ms": cold_inference_ms,
                    "milliseconds_per_inference_call": (
                        1000.0 * inference_elapsed / args.inference_steps
                    ),
                    "inference_output_sequences_per_second": (
                        args.batch_size * args.inference_steps / inference_elapsed
                    ),
                    "peak_cuda_memory_mb": peak_memory_mb,
                    "final_synthetic_loss": loss,
                    "prediction_shape": list(prediction.shape),
                }
            )

    for backbone in ("gru", "diagonal_ssm"):
        eager = next(
            row
            for row in results
            if row["backbone"] == backbone and row["execution"] == "eager"
        )
        compiled = next(
            row
            for row in results
            if row["backbone"] == backbone and row["execution"] == "compiled"
        )
        compiled["training_speedup_vs_eager"] = (
            eager["milliseconds_per_training_step"]
            / compiled["milliseconds_per_training_step"]
        )
        compiled["inference_speedup_vs_eager"] = (
            eager["milliseconds_per_inference_call"]
            / compiled["milliseconds_per_inference_call"]
        )
    payload = {
        "device": str(device),
        "pytorch_version": torch.__version__,
        "matmul_precision": args.matmul_precision,
        "results": results,
    }
    rendered = json.dumps(payload, indent=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
