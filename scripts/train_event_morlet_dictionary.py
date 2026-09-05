#!/usr/bin/env python3
"""Event-fold diagnostic for a fixed and learnable overcomplete Morlet bank."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from ecog_decoding.models import fit_fastica_spatial_weights
from ecog_decoding.spectrotemporal import ContinuousMorletDecoder
from ecog_decoding.training import FINGER_NAMES
from train_event_grouped_lars_lstm import indices_from_intervals, starts_from_intervals
from train_morlet_dictionary import (
    batch_slices,
    cpu_state,
    pearson,
    plot_filter_responses,
    plot_trajectories,
)


TARGETS = {1: "local_w2_q10", 2: "local_w1_q10", 3: "local_w4_q20"}


def sample_training_ecog(
    ecog: np.ndarray,
    intervals: list[list[int]],
    history_offset: int,
    stride_samples: int,
    maximum_samples: int,
    seed: int,
) -> np.ndarray:
    ranges = [
        (start * stride_samples, (stop + history_offset) * stride_samples)
        for start, stop in intervals
    ]
    lengths = np.asarray([stop - start for start, stop in ranges], dtype=np.int64)
    total = int(lengths.sum())
    generator = np.random.default_rng(seed)
    virtual = np.sort(
        generator.choice(total, size=min(total, maximum_samples), replace=False)
    )
    boundaries = np.cumsum(lengths)
    segment = np.searchsorted(boundaries, virtual, side="right")
    previous = np.r_[0, boundaries[:-1]]
    raw_indices = np.asarray(
        [ranges[int(part)][0] + int(value - previous[int(part)]) for value, part in zip(virtual, segment)],
        dtype=np.int64,
    )
    return np.asarray(ecog[raw_indices], dtype=np.float32)


def gather_sequences(
    values: torch.Tensor, starts: np.ndarray, sequence_steps: int
) -> torch.Tensor:
    offsets = torch.arange(sequence_steps, device=values.device)
    indices = torch.as_tensor(starts, device=values.device)[:, None] + offsets[None]
    return values[:, indices].squeeze(0)


@torch.inference_mode()
def predict_cached_intervals(
    model: ContinuousMorletDecoder,
    representation: torch.Tensor,
    broadband: torch.Tensor,
    intervals: list[list[int]],
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    indices: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    for start, stop in intervals:
        indices.append(np.arange(start, stop, dtype=np.int64))
        predictions.append(
            model.forward_from_representation(
                representation[..., start:stop], broadband[..., start:stop]
            )[0]
            .float()
            .cpu()
            .numpy()
        )
    return np.concatenate(indices), np.concatenate(predictions)


@torch.inference_mode()
def predict_live_intervals(
    model: ContinuousMorletDecoder,
    recording: torch.Tensor,
    offset: int,
    rows: int,
    intervals: list[list[int]],
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    features = model.extract(recording[None])[:, offset : offset + rows]
    indices: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    for start, stop in intervals:
        indices.append(np.arange(start, stop, dtype=np.int64))
        predictions.append(model.decode(features[:, start:stop])[0].float().cpu().numpy())
    return np.concatenate(indices), np.concatenate(predictions)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, required=True, choices=(1, 2, 3))
    parser.add_argument("--finger", required=True, choices=tuple(FINGER_NAMES))
    parser.add_argument("--fold", type=int, required=True, choices=(0, 1, 2))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prepared-root", type=Path, default=Path("outputs/preprocessed_v2"))
    parser.add_argument("--fold-root", type=Path, default=Path("outputs/event_stratified_folds_v1"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/event_morlet_dictionary_v1"))
    parser.add_argument("--history", type=int, default=25)
    parser.add_argument("--stride-samples", type=int, default=40)
    parser.add_argument("--spatial-components", type=int, default=32)
    parser.add_argument("--atoms", type=int, default=64)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--kernel-size", type=int, default=1001)
    parser.add_argument("--convolution-units", type=int, default=4)
    parser.add_argument("--feature-width", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--sequence-steps", type=int, default=128)
    parser.add_argument("--sequence-stride", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--fixed-epochs", type=int, default=20)
    parser.add_argument("--learnable-epochs", type=int, default=30)
    parser.add_argument("--dense-epochs", type=int, default=8)
    parser.add_argument("--anneal-epochs", type=int, default=16)
    parser.add_argument("--head-learning-rate", type=float, default=1e-3)
    parser.add_argument("--finetune-learning-rate", type=float, default=2e-4)
    parser.add_argument("--atom-learning-rate", type=float, default=5e-4)
    parser.add_argument("--gate-learning-rate", type=float, default=2e-3)
    parser.add_argument("--spatial-learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--movement-threshold", type=float, default=0.08)
    parser.add_argument("--movement-weight", type=float, default=4.0)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.atoms <= args.top_k:
        parser.error("atoms must exceed top-k")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)

    definition = json.loads(
        (args.fold_root / f"sub{args.subject}" / args.finger / "folds.json").read_text()
    )
    fold = definition["folds"][args.fold]
    training_intervals = fold["training_intervals_after_purge"]
    validation_intervals = fold["validation_intervals"]
    rows = int(definition["training_rows"])
    offset = args.history - 1
    raw_stop = (rows + offset) * args.stride_samples
    prepared = args.prepared_root / f"sub{args.subject}"
    ecog = np.load(prepared / "train_ecog.npy", mmap_mode="r")
    sample = sample_training_ecog(
        ecog,
        training_intervals,
        offset,
        args.stride_samples,
        50_000,
        args.seed,
    )
    ecog_mean = sample.mean(axis=0, dtype=np.float64).astype(np.float32)
    weights = fit_fastica_spatial_weights(
        sample,
        n_components=args.spatial_components,
        max_samples=sample.shape[0],
        random_state=args.seed,
        backend="torch",
        device=device,
    )
    centered_recording = torch.from_numpy(
        np.asarray(ecog[:raw_stop], dtype=np.float32).copy()
    ).to(device)
    centered_recording = (centered_recording - torch.from_numpy(ecog_mean).to(device)).T.contiguous()
    spatial = torch.matmul(torch.from_numpy(weights).to(device), centered_recording)
    spatial_scale = spatial.std(dim=1, correction=0).clamp_min(1e-6)
    spatial = spatial / spatial_scale[:, None]

    finger_index = list(FINGER_NAMES).index(args.finger)
    cleaned_all = np.load(prepared / f"train_glove_{TARGETS[args.subject]}.npy")
    raw_all = np.load(prepared / "train_glove_25hz_raw.npy")
    cleaned = torch.from_numpy(
        np.asarray(cleaned_all[offset : offset + rows, finger_index], dtype=np.float32)
    ).to(device)
    raw = np.asarray(raw_all[offset : offset + rows, finger_index], dtype=np.float32)

    model = ContinuousMorletDecoder(
        input_components=args.spatial_components,
        input_channels=ecog.shape[1],
        atom_count=args.atoms,
        top_k=args.top_k,
        trainable_atoms=True,
        kernel_size=args.kernel_size,
        convolution_units=args.convolution_units,
        feature_width=args.feature_width,
        hidden_size=args.hidden_size,
        output_activation="softplus",
    ).to(device)
    with torch.no_grad():
        normalized_weights = torch.from_numpy(weights).to(device) / spatial_scale[:, None]
        model.spatial_projection.weight[:, :, 0].copy_(normalized_weights)
        representation_all, broadband_all = model.morlet(spatial[None])
    representation = representation_all[..., offset : offset + rows]
    broadband = broadband_all[..., offset : offset + rows]
    del representation_all, broadband_all, spatial

    atom_parameters = [model.morlet.center_logits, model.morlet.width_logits]
    gate_parameters = [] if model.atom_gate is None else [model.atom_gate.logits]
    spatial_parameters = list(model.spatial_projection.parameters())
    for parameter in atom_parameters + gate_parameters + spatial_parameters:
        parameter.requires_grad_(False)
    head_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(head_parameters, lr=args.head_learning_rate, weight_decay=args.weight_decay)
    fixed_forward = model.forward_from_representation
    if args.compile:
        fixed_forward = torch.compile(fixed_forward, mode="reduce-overhead")
    starts = starts_from_intervals(training_intervals, args.sequence_steps, args.sequence_stride)
    rng = np.random.default_rng(args.seed)
    best_fixed_score = -np.inf
    best_fixed_state = cpu_state(model)
    best_fixed_prediction: np.ndarray | None = None
    fixed_history: list[dict[str, float]] = []
    started = time.perf_counter()
    offsets = np.arange(args.sequence_steps, dtype=np.int64)
    for epoch in range(1, args.fixed_epochs + 1):
        rng.shuffle(starts)
        model.train()
        losses: list[float] = []
        for begin in range(0, starts.size, args.batch_size):
            selected = starts[begin : begin + args.batch_size]
            indices = torch.as_tensor(selected[:, None] + offsets[None], device=device)
            maps, residual = batch_slices(
                representation, broadband, selected, args.sequence_steps
            )
            prediction = fixed_forward(maps, residual)
            observed = cleaned[indices]
            weight = 1.0 + args.movement_weight * (observed >= args.movement_threshold)
            loss = (weight * (prediction - observed).square()).sum() / weight.sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head_parameters, 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        order, prediction = predict_cached_intervals(
            model, representation, broadband, validation_intervals
        )
        score = pearson(prediction, raw[order])
        record = {"epoch": epoch, "loss": float(np.mean(losses)), "validation_raw_pcc": score}
        fixed_history.append(record)
        print(json.dumps({"stage": "fixed", **record}), flush=True)
        if score > best_fixed_score:
            best_fixed_score = score
            best_fixed_state = cpu_state(model)
            best_fixed_prediction = prediction.copy()
    fixed_seconds = time.perf_counter() - started

    model.load_state_dict(best_fixed_state)
    for parameter in atom_parameters + gate_parameters + spatial_parameters:
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        [
            {"params": head_parameters, "lr": args.finetune_learning_rate},
            {"params": atom_parameters, "lr": args.atom_learning_rate},
            {"params": gate_parameters, "lr": args.gate_learning_rate},
            {"params": spatial_parameters, "lr": args.spatial_learning_rate},
        ],
        weight_decay=args.weight_decay,
    )
    live_extract = model.extract
    if args.compile:
        live_extract = torch.compile(live_extract, mode="reduce-overhead")
    best_learned_score = best_fixed_score
    best_learned_state = cpu_state(model)
    best_learned_prediction = best_fixed_prediction.copy() if best_fixed_prediction is not None else None
    learned_history: list[dict[str, float]] = []
    learned_started = time.perf_counter()
    for epoch in range(1, args.learnable_epochs + 1):
        if epoch <= args.dense_epochs:
            hard_fraction, temperature = 0.0, 1.0
        else:
            progress = min(1.0, (epoch - args.dense_epochs) / max(1, args.anneal_epochs))
            hard_fraction, temperature = progress, 1.0 - 0.75 * progress
        model.set_gate_schedule(temperature, hard_fraction)
        model.train()
        features = live_extract(centered_recording[None])[:, offset : offset + rows]
        sequences = gather_sequences(features, starts, args.sequence_steps)
        prediction = model.decode(sequences)
        target_indices = torch.as_tensor(starts, device=device)[:, None] + torch.arange(args.sequence_steps, device=device)[None]
        observed = cleaned[target_indices]
        weight = 1.0 + args.movement_weight * (observed >= args.movement_threshold)
        loss = (weight * (prediction - observed).square()).sum() / weight.sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        order, validation_prediction = predict_live_intervals(
            model, centered_recording, offset, rows, validation_intervals
        )
        score = pearson(validation_prediction, raw[order])
        record = {
            "epoch": epoch,
            "loss": float(loss.detach()),
            "temperature": temperature,
            "hard_fraction": hard_fraction,
            "validation_raw_pcc": score,
        }
        learned_history.append(record)
        print(json.dumps({"stage": "learned", **record}), flush=True)
        if score > best_learned_score:
            best_learned_score = score
            best_learned_state = cpu_state(model)
            best_learned_prediction = validation_prediction.copy()
    learned_seconds = time.perf_counter() - learned_started

    if best_fixed_prediction is None or best_learned_prediction is None:
        raise RuntimeError("dictionary training produced no validation prediction")
    model.load_state_dict(best_learned_state)
    order = indices_from_intervals(validation_intervals)
    output = args.output_root / f"sub{args.subject}" / args.finger / f"fold{args.fold}" / f"seed{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "validation_indices.npy", order)
    np.save(output / "validation_raw_target.npy", raw[order])
    np.save(output / "validation_cleaned_target.npy", cleaned[order].cpu().numpy())
    np.save(output / "fixed_prediction.npy", best_fixed_prediction)
    np.save(output / "learned_prediction.npy", best_learned_prediction)
    plot_trajectories(
        output / "validation_trajectories.png",
        raw[order],
        cleaned[order].cpu().numpy(),
        best_fixed_prediction,
        best_learned_prediction,
    )
    plot_filter_responses(output / "learned_filter_responses.png", model)
    centers = model.morlet.center_frequencies_hz().detach().cpu().numpy()
    widths = model.morlet.temporal_widths_seconds().detach().cpu().numpy()
    selected = model.atom_gate.selected_indices().cpu().tolist() if model.atom_gate is not None else list(range(args.atoms))
    summary = {
        "protocol": "per-finger purged event-fold dictionary diagnostic; outer validation used only for exploratory curves",
        "subject": args.subject,
        "finger": args.finger,
        "fold": args.fold,
        "seed": args.seed,
        "official_final_validation_touched": False,
        "released_test_touched": False,
        "atoms": args.atoms,
        "top_k": args.top_k,
        "selected_atom_indices": selected,
        "selected_centers_hz": [float(centers[index]) for index in selected],
        "selected_widths_seconds": [float(widths[index]) for index in selected],
        "fixed_best_validation_raw_pcc": best_fixed_score,
        "learned_best_validation_raw_pcc": best_learned_score,
        "learned_improvement": best_learned_score - best_fixed_score,
        "runtime_seconds": {"fixed": fixed_seconds, "learned": learned_seconds},
        "fixed_history": fixed_history,
        "learned_history": learned_history,
    }
    torch.save({"model_state_dict": best_learned_state, "summary": summary}, output / "model.pt")
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"saved": str(output), **summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
