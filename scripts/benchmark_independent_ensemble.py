#!/usr/bin/env python3
"""Benchmark independent full-model ensemble execution on one accelerator.

The members share only the input tensor.  Every spatial projection, wavelet
stem, recurrent decoder, and output head has independent parameters.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.func import functional_call, stack_module_state

from scripts.train_exact_window_end_to_end import ExactWindowFingerDecoder
from ecog_decoding.training import FINGER_NAMES


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed_steps(fn, *, warmup: int, steps: int, device: torch.device) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for _ in range(steps):
        fn()
    synchronize(device)
    elapsed = time.perf_counter() - started
    peak_gib = (
        torch.cuda.max_memory_allocated(device) / 2**30
        if device.type == "cuda"
        else float("nan")
    )
    return {
        "seconds_per_step": elapsed / steps,
        "peak_allocated_gib": peak_gib,
    }


def clear_gradients(parameters) -> None:
    for parameter in parameters:
        parameter.grad = None


class IndependentModelEnsemble(nn.Module):
    """One module containing independent full decoders and no shared weights."""

    def __init__(self, members: list[ExactWindowFingerDecoder]) -> None:
        super().__init__()
        self.members = nn.ModuleList(members)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return torch.stack([member(values) for member in self.members], dim=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, default=1)
    parser.add_argument("--finger", choices=tuple(FINGER_NAMES), default="index")
    parser.add_argument("--members", type=int, default=6)
    parser.add_argument("--sequence-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--frontend", choices=("wavelet", "asymmetric"), default="wavelet")
    parser.add_argument("--wavelet-levels", type=int, choices=(3, 4), default=3)
    parser.add_argument("--prepared-root", type=Path, default=Path("outputs/preprocessed_v2"))
    parser.add_argument("--ica-root", type=Path, default=Path("outputs/ica_sklearn_constant_v1"))
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path("outputs/exact_e2e_s1_index_h40_v1"),
        help="output root containing subN/FINGER.pt from the exact-window trainer",
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.members < 1 or args.sequence_steps < 1 or args.batch_size < 1:
        parser.error("members, sequence steps, and batch size must be positive")

    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")
    prepared = args.prepared_root / f"sub{args.subject}"
    ica_dir = args.ica_root / f"sub{args.subject}"
    finger = list(FINGER_NAMES).index(args.finger)
    checkpoint = torch.load(
        args.checkpoint_root / f"sub{args.subject}" / f"{args.finger}.pt",
        map_location="cpu",
        weights_only=False,
    )
    state = checkpoint["model_state_dict"]
    indices = np.asarray(checkpoint["feature_indices"], dtype=np.int64)
    mean = state["feature_mean"].cpu().numpy()
    scale = state["feature_scale"].cpu().numpy()
    hidden_size = int(state["lstm.weight_hh_l0"].shape[1])
    ica = np.load(ica_dir / "fastica_unmixing.npy")

    recording = torch.from_numpy(
        np.asarray(np.load(prepared / "train_ecog.npy", mmap_mode="r")).copy()
    ).to(device)
    windows = recording.unfold(0, 1000, 40)
    needed = args.batch_size * args.sequence_steps
    x = windows[:needed].reshape(
        args.batch_size, args.sequence_steps, recording.shape[1], 1000
    )
    raw_target = np.load(prepared / "train_glove_25hz_raw.npy")
    y = torch.from_numpy(
        np.asarray(raw_target[24 : 24 + needed, finger], dtype=np.float32)
    ).to(device).reshape(args.batch_size, args.sequence_steps)

    prototype = ExactWindowFingerDecoder(
        recording.shape[1],
        ica.shape[0],
        indices,
        mean,
        scale,
        hidden_size,
        wavelet_levels=args.wavelet_levels,
        frontend=args.frontend,
        output_activation="softplus",
        softplus_beta=10.0,
    )
    with torch.no_grad():
        prototype.spatial.weight[:, :, 0].copy_(torch.from_numpy(ica))
    prototype.load_state_dict(state)

    sequential_models = []
    for member in range(args.members):
        torch.manual_seed(member)
        model = copy.deepcopy(prototype).to(device)
        model.lstm.reset_parameters()
        sequential_models.append(model)

    def sequential_step() -> None:
        for model in sequential_models:
            prediction = model(x)
            (prediction.sub(y).square().mean()).backward()
            clear_gradients(model.parameters())

    sequential = timed_steps(
        sequential_step, warmup=args.warmup, steps=args.steps, device=device
    )
    del sequential_models
    if device.type == "cuda":
        torch.cuda.empty_cache()

    packed_members = []
    for member in range(args.members):
        torch.manual_seed(member)
        model = copy.deepcopy(prototype).to(device)
        model.lstm.reset_parameters()
        packed_members.append(model)
    packed = IndependentModelEnsemble(packed_members).to(device)

    def packed_step() -> None:
        prediction = packed(x)
        loss = prediction.sub(y.unsqueeze(0)).square().mean(dim=(1, 2)).sum()
        loss.backward()
        clear_gradients(packed.parameters())

    packed_eager = timed_steps(
        packed_step, warmup=args.warmup, steps=args.steps, device=device
    )
    packed_eager["speedup_over_sequential"] = (
        sequential["seconds_per_step"] / packed_eager["seconds_per_step"]
    )

    compiled_packed = torch.compile(packed, mode="reduce-overhead")

    def packed_compiled_step() -> None:
        prediction = compiled_packed(x)
        loss = prediction.sub(y.unsqueeze(0)).square().mean(dim=(1, 2)).sum()
        loss.backward()
        clear_gradients(packed.parameters())

    try:
        packed_compiled = timed_steps(
            packed_compiled_step,
            warmup=args.warmup,
            steps=args.steps,
            device=device,
        )
        packed_compiled["speedup_over_sequential"] = (
            sequential["seconds_per_step"] / packed_compiled["seconds_per_step"]
        )
        packed_compiled["supported"] = True
    except Exception as error:
        packed_compiled = {
            "supported": False,
            "error_type": type(error).__name__,
            "error": str(error),
        }
    del compiled_packed, packed
    if device.type == "cuda":
        torch.cuda.empty_cache()

    models = []
    for member in range(args.members):
        torch.manual_seed(member)
        model = copy.deepcopy(prototype).to(device)
        model.lstm.reset_parameters()
        models.append(model)
    parameters, buffers = stack_module_state(models)
    # Keep a storage-backed stateless template.  nn.LSTM consults its flattened
    # weight pointers before dispatch, which makes a meta-device template fail
    # even though functional_call supplies real batched parameters.
    base = models[0]
    del models

    def member_forward(member_parameters, member_buffers, values):
        return functional_call(base, (member_parameters, member_buffers), (values,))

    vectorized_forward = torch.vmap(
        member_forward, in_dims=(0, 0, None), randomness="different"
    )

    def vectorized_step() -> None:
        prediction = vectorized_forward(parameters, buffers, x)
        # Sum member-wise mean losses so each model receives the same gradient
        # magnitude as an independently trained model.
        loss = prediction.sub(y.unsqueeze(0)).square().mean(dim=(1, 2)).sum()
        loss.backward()
        clear_gradients(parameters.values())

    try:
        vectorized = timed_steps(
            vectorized_step, warmup=args.warmup, steps=args.steps, device=device
        )
        vectorized["speedup_over_sequential"] = (
            sequential["seconds_per_step"] / vectorized["seconds_per_step"]
        )
        vectorized["supported"] = True
    except Exception as error:  # Report unsupported operators rather than hiding them.
        vectorized = {
            "supported": False,
            "error_type": type(error).__name__,
            "error": str(error),
        }

    result = {
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
        ),
        "members": args.members,
        "batch_size": args.batch_size,
        "sequence_steps": args.sequence_steps,
        "input_shape": list(x.shape),
        "feature_count": int(indices.size),
        "sequential": sequential,
        "packed_modulelist_eager": packed_eager,
        "packed_modulelist_compiled": packed_compiled,
        "vectorized_independent_full_models": vectorized,
    }
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
