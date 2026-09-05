#!/usr/bin/env python3
"""Efficient inner-fold diagnostic for an overcomplete learnable Morlet dictionary.

The released test set and the final chronological validation block are never
loaded.  FastICA is fit on the inner-training interval only.  The fixed-bank
stage caches its continuous filter response once; the learnable stage filters
the contiguous training recording once per optimizer step rather than
reconstructing thousands of overlapping one-second windows.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from ecog_decoding.models import fit_fastica_spatial_weights
from ecog_decoding.spectrotemporal import ContinuousMorletDecoder
from ecog_decoding.training import FINGER_NAMES


TARGETS = {1: "local_w2_q10", 2: "local_w1_q10", 3: "local_w4_q20"}


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    x = x - x.mean()
    y = y - y.mean()
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    return float(x @ y / denominator) if denominator > 0 else 0.0


def cpu_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }


def blocked_fold(metadata: dict[str, object], history: int, fold: int) -> tuple[int, int]:
    rows = int(metadata["target_fit_samples_25hz"]) - (history - 1)
    boundaries = np.rint(np.linspace(0.5 * rows, rows, 4)).astype(int)
    if not 0 <= fold < 3:
        raise ValueError("fold must be 0, 1, or 2")
    return int(boundaries[fold]), int(boundaries[fold + 1])


def load_or_fit_ica(
    cache: Path,
    training_ecog: np.ndarray,
    components: int,
    seed: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    cache.parent.mkdir(parents=True, exist_ok=True)
    lock_path = cache.with_suffix(cache.suffix + ".lock")
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if cache.exists():
            values = np.load(cache)
            return values["weights"], values["mean"]
        started = time.perf_counter()
        mean = np.asarray(training_ecog, dtype=np.float64).mean(axis=0).astype(np.float32)
        weights = fit_fastica_spatial_weights(
            training_ecog,
            n_components=components,
            max_samples=50_000,
            random_state=seed,
            backend="torch",
            device=device,
        )
        temporary = cache.with_suffix(cache.suffix + ".tmp.npz")
        np.savez(temporary, weights=weights, mean=mean)
        temporary.replace(cache)
        print(
            json.dumps({"event": "ica_cached", "seconds": time.perf_counter() - started}),
            flush=True,
        )
        return weights, mean


def batch_slices(
    representation: torch.Tensor,
    broadband: torch.Tensor,
    starts: np.ndarray,
    sequence_steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    maps = torch.cat(
        [representation[..., start : start + sequence_steps] for start in starts],
        dim=0,
    )
    residual = torch.cat(
        [broadband[..., start : start + sequence_steps] for start in starts],
        dim=0,
    )
    return maps, residual


@torch.inference_mode()
def cached_prediction(
    model: ContinuousMorletDecoder,
    representation: torch.Tensor,
    broadband: torch.Tensor,
) -> np.ndarray:
    model.eval()
    return (
        model.forward_from_representation(representation, broadband)[0]
        .float()
        .cpu()
        .numpy()
    )


@torch.inference_mode()
def live_prediction(
    model: ContinuousMorletDecoder,
    recording: torch.Tensor,
    offset: int,
) -> np.ndarray:
    model.eval()
    return model(recording[None])[0, offset:].float().cpu().numpy()


def existing_fold_baseline(
    summary_path: Path,
    subject: int,
    finger_name: str,
    fold: int,
) -> dict[str, float]:
    if not summary_path.exists():
        return {}
    report = json.loads(summary_path.read_text())
    candidates = report["subjects"][str(subject)]["per_finger"][finger_name][
        "candidate_audit"
    ]
    return {
        str(candidate["config"]): float(candidate["fold_validation_r"][fold])
        for candidate in candidates
        if bool(candidate.get("eligible", True))
    }


def plot_trajectories(
    path: Path,
    raw: np.ndarray,
    cleaned: np.ndarray,
    fixed: np.ndarray,
    learned: np.ndarray,
    sample_rate_hz: float = 25.0,
) -> None:
    time_axis = np.arange(raw.size) / sample_rate_hz
    figure, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    axes[0].plot(time_axis, raw, color="black", linewidth=1.1, label="raw glove")
    axes[0].plot(time_axis, fixed, linewidth=1.0, label="fixed Morlet")
    axes[0].plot(time_axis, learned, linewidth=1.0, label="learned Top-K")
    axes[0].set_ylabel("raw coordinate")
    axes[0].legend(ncol=3, frameon=False)
    axes[1].plot(time_axis, cleaned, color="black", linewidth=1.1, label="cleaned target")
    axes[1].plot(time_axis, fixed, linewidth=1.0, label="fixed Morlet")
    axes[1].plot(time_axis, learned, linewidth=1.0, label="learned Top-K")
    axes[1].set_ylabel("movement")
    axes[1].set_xlabel("inner-validation time (s)")
    axes[1].legend(ncol=3, frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def plot_filter_responses(
    path: Path,
    model: ContinuousMorletDecoder,
) -> None:
    real, imaginary = model.morlet.complex_kernels()
    kernels = real.detach().cpu().numpy() + 1j * imaginary.detach().cpu().numpy()
    response = np.abs(np.fft.fft(kernels, n=8192, axis=-1))[:, :4097]
    frequency = np.linspace(0.0, model.morlet.sampling_rate_hz / 2.0, 4097)
    response /= np.maximum(response.max(axis=1, keepdims=True), 1.0e-12)
    selected = (
        set(model.atom_gate.selected_indices().cpu().tolist())
        if model.atom_gate is not None
        else set(range(model.atom_count))
    )
    figure, axis = plt.subplots(figsize=(12, 6))
    for atom, curve in enumerate(response):
        axis.plot(
            frequency,
            curve,
            linewidth=1.8 if atom in selected else 0.7,
            alpha=0.95 if atom in selected else 0.18,
        )
    axis.set_xlim(0, 500)
    axis.set_xlabel("frequency (Hz)")
    axis.set_ylabel("normalized magnitude")
    axis.set_title("Learned complex Morlet dictionary; emphasized curves are selected")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, default=1, choices=(1, 2, 3))
    parser.add_argument("--finger", choices=tuple(FINGER_NAMES), default="index")
    parser.add_argument("--fold", type=int, default=0, choices=(0, 1, 2))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prepared-root", type=Path, default=Path("outputs/preprocessed_v2"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/morlet_dictionary_inner_v1"))
    parser.add_argument("--cache-root", type=Path, default=Path("outputs/morlet_cache_v1"))
    parser.add_argument("--baseline-summary", type=Path, default=Path("outputs/nested_cv_selection_v1/summary.json"))
    parser.add_argument("--history", type=int, default=25)
    parser.add_argument("--stride-samples", type=int, default=40)
    parser.add_argument("--spatial-components", type=int, default=32)
    parser.add_argument("--atoms", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--kernel-size", type=int, default=1001)
    parser.add_argument("--convolution-units", type=int, default=4)
    parser.add_argument("--feature-width", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--sequence-steps", type=int, default=256)
    parser.add_argument("--sequence-stride", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--fixed-epochs", type=int, default=24)
    parser.add_argument("--learnable-epochs", type=int, default=60)
    parser.add_argument("--dense-epochs", type=int, default=15)
    parser.add_argument("--anneal-epochs", type=int, default=30)
    parser.add_argument("--head-learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--finetune-learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--atom-learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--gate-learning-rate", type=float, default=5.0e-3)
    parser.add_argument("--spatial-learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.atoms <= args.top_k:
        parser.error("atoms must exceed top-k for an overcomplete sparse dictionary")
    if args.kernel_size % 2 == 0:
        parser.error("kernel-size must be odd")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)

    prepared = args.prepared_root / f"sub{args.subject}"
    metadata = json.loads((prepared / "metadata.json").read_text())
    fit_end, validation_stop = blocked_fold(metadata, args.history, args.fold)
    offset = args.history - 1
    fit_raw_stop = (fit_end + offset) * args.stride_samples
    validation_raw_stop = (validation_stop + offset) * args.stride_samples
    ecog = np.load(prepared / "train_ecog.npy", mmap_mode="r")
    if validation_raw_stop > ecog.shape[0]:
        parser.error("requested inner fold exceeds the ECoG recording")

    cache_tag = (
        f"sub{args.subject}_fold{args.fold}_fit{fit_end}_"
        f"c{args.spatial_components}_seed{args.seed}.npz"
    )
    weights, ecog_mean = load_or_fit_ica(
        args.cache_root / cache_tag,
        np.asarray(ecog[:fit_raw_stop]),
        args.spatial_components,
        args.seed,
        device,
    )
    ecog_tensor = torch.from_numpy(np.asarray(ecog[:validation_raw_stop]).copy()).to(device)
    ecog_tensor = ecog_tensor - torch.from_numpy(ecog_mean).to(device)
    centered_recording = ecog_tensor.T.contiguous()
    spatial = torch.matmul(
        torch.from_numpy(weights).to(device), ecog_tensor.T
    )
    spatial_scale = spatial[:, :fit_raw_stop].std(dim=1, correction=0).clamp_min(1e-6)
    spatial = spatial / spatial_scale[:, None]
    del ecog_tensor

    finger = list(FINGER_NAMES).index(args.finger)
    cleaned_all = np.load(prepared / f"train_glove_{TARGETS[args.subject]}.npy")
    raw_all = np.load(prepared / "train_glove_25hz_raw.npy")
    cleaned = torch.from_numpy(
        np.asarray(cleaned_all[offset : offset + validation_stop, finger], dtype=np.float32)
    ).to(device)
    raw = np.asarray(
        raw_all[offset : offset + validation_stop, finger], dtype=np.float32
    )

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
    assert isinstance(model.spatial_projection, torch.nn.Conv1d)
    with torch.no_grad():
        normalized_weights = torch.from_numpy(weights).to(device) / spatial_scale[:, None]
        model.spatial_projection.weight[:, :, 0].copy_(normalized_weights)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    atom_parameters = [model.morlet.center_logits, model.morlet.width_logits]
    gate_parameters = [] if model.atom_gate is None else [model.atom_gate.logits]
    spatial_parameters = list(model.spatial_projection.parameters())
    for parameter in atom_parameters + gate_parameters + spatial_parameters:
        parameter.requires_grad_(False)
    head_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]

    with torch.inference_mode():
        representation_all, broadband_all = model.morlet(spatial[None])
    representation = representation_all[..., offset : offset + validation_stop]
    broadband = broadband_all[..., offset : offset + validation_stop]
    del representation_all, broadband_all

    fixed_optimizer = torch.optim.AdamW(
        head_parameters,
        lr=args.head_learning_rate,
        weight_decay=args.weight_decay,
    )
    cached_train = model.forward_from_representation
    if args.compile:
        cached_train = torch.compile(cached_train, mode="reduce-overhead")
    starts = np.arange(0, fit_end - args.sequence_steps + 1, args.sequence_stride)
    if starts.size == 0:
        parser.error("sequence is longer than the inner-training interval")
    generator = np.random.default_rng(args.seed)
    fixed_history: list[dict[str, float]] = []
    best_fixed_score = -np.inf
    best_fixed_state: dict[str, torch.Tensor] | None = None
    best_fixed_prediction: np.ndarray | None = None
    started = time.perf_counter()
    for epoch in range(1, args.fixed_epochs + 1):
        generator.shuffle(starts)
        padded_count = (-starts.size) % args.batch_size
        ordered = (
            np.concatenate((starts, starts[:padded_count])) if padded_count else starts
        )
        model.train()
        losses: list[float] = []
        for begin in range(0, ordered.size, args.batch_size):
            selected = ordered[begin : begin + args.batch_size]
            maps, residual = batch_slices(
                representation, broadband, selected, args.sequence_steps
            )
            prediction = cached_train(maps, residual)
            indices = torch.as_tensor(
                selected[:, None]
                + np.arange(args.sequence_steps, dtype=np.int64)[None],
                device=device,
            )
            loss = (prediction - cleaned[indices]).square().mean()
            fixed_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head_parameters, 1.0)
            fixed_optimizer.step()
            losses.append(float(loss.detach()))
        prediction = cached_prediction(model, representation, broadband)
        score = pearson(prediction[fit_end:validation_stop], raw[fit_end:validation_stop])
        record = {
            "epoch": float(epoch),
            "loss": float(np.mean(losses)),
            "inner_validation_raw_pcc": score,
            "elapsed_seconds": time.perf_counter() - started,
        }
        fixed_history.append(record)
        print(json.dumps({"stage": "fixed", **record}), flush=True)
        if score > best_fixed_score:
            best_fixed_score = score
            best_fixed_state = cpu_state(model)
            best_fixed_prediction = prediction.copy()

    if best_fixed_state is None or best_fixed_prediction is None:
        raise RuntimeError("fixed-bank training produced no result")
    fixed_seconds = time.perf_counter() - started
    model.load_state_dict(best_fixed_state)
    for parameter in atom_parameters + gate_parameters + spatial_parameters:
        parameter.requires_grad_(True)
    head_parameters = [
        parameter
        for parameter in model.parameters()
        if all(
            parameter is not other
            for other in atom_parameters + gate_parameters + spatial_parameters
        )
    ]
    finetune_optimizer = torch.optim.AdamW(
        [
            {"params": head_parameters, "lr": args.finetune_learning_rate},
            {"params": atom_parameters, "lr": args.atom_learning_rate},
            {"params": gate_parameters, "lr": args.gate_learning_rate},
            {"params": spatial_parameters, "lr": args.spatial_learning_rate},
        ],
        weight_decay=args.weight_decay,
    )
    live_train: torch.nn.Module = model
    if args.compile:
        live_train = torch.compile(model, mode="reduce-overhead")

    learned_history: list[dict[str, float]] = []
    best_learned_score = best_fixed_score
    best_learned_state = cpu_state(model)
    best_learned_prediction = best_fixed_prediction.copy()
    learned_started = time.perf_counter()
    for epoch in range(1, args.learnable_epochs + 1):
        if epoch <= args.dense_epochs:
            hard_fraction = 0.0
            temperature = 1.0
        else:
            progress = min(
                1.0,
                (epoch - args.dense_epochs) / max(1, args.anneal_epochs),
            )
            hard_fraction = progress
            temperature = 1.0 - 0.75 * progress
        model.set_gate_schedule(temperature, hard_fraction)
        model.train()
        prediction = live_train(centered_recording[None, :, :fit_raw_stop])[0, offset:]
        loss = (prediction - cleaned[:fit_end]).square().mean()
        finetune_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        finetune_optimizer.step()

        validation_prediction = live_prediction(model, centered_recording, offset)
        score = pearson(
            validation_prediction[fit_end:validation_stop],
            raw[fit_end:validation_stop],
        )
        record = {
            "epoch": float(epoch),
            "loss": float(loss.detach()),
            "temperature": float(temperature),
            "hard_fraction": float(hard_fraction),
            "inner_validation_raw_pcc": score,
            "elapsed_seconds": time.perf_counter() - learned_started,
        }
        learned_history.append(record)
        print(json.dumps({"stage": "learned", **record}), flush=True)
        if score > best_learned_score:
            best_learned_score = score
            best_learned_state = cpu_state(model)
            best_learned_prediction = validation_prediction.copy()

    learned_seconds = time.perf_counter() - learned_started
    model.load_state_dict(best_learned_state)
    output = (
        args.output_root
        / f"sub{args.subject}"
        / args.finger
        / f"fold{args.fold}"
        / f"seed{args.seed}"
    )
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "fastica_unmixing.npy", weights)
    np.save(output / "ecog_training_mean.npy", ecog_mean)
    np.save(output / "spatial_training_scale.npy", spatial_scale.cpu().numpy())
    validation_slice = slice(fit_end, validation_stop)
    np.save(output / "inner_validation_raw_target.npy", raw[validation_slice])
    np.save(output / "inner_validation_cleaned_target.npy", cleaned[validation_slice].cpu().numpy())
    np.save(output / "fixed_prediction.npy", best_fixed_prediction[validation_slice])
    np.save(output / "learned_prediction.npy", best_learned_prediction[validation_slice])
    plot_trajectories(
        output / "inner_validation_trajectories.png",
        raw[validation_slice],
        cleaned[validation_slice].cpu().numpy(),
        best_fixed_prediction[validation_slice],
        best_learned_prediction[validation_slice],
    )
    plot_filter_responses(output / "learned_filter_responses.png", model)

    centers = model.morlet.center_frequencies_hz().detach().cpu().numpy()
    widths = model.morlet.temporal_widths_seconds().detach().cpu().numpy()
    selected = (
        model.atom_gate.selected_indices().cpu().tolist()
        if model.atom_gate is not None
        else list(range(args.atoms))
    )
    gate_probabilities = (
        model.atom_gate.probabilities().detach().cpu().tolist()
        if model.atom_gate is not None
        else [1.0 / args.atoms] * args.atoms
    )
    baseline = existing_fold_baseline(
        args.baseline_summary, args.subject, args.finger, args.fold
    )
    summary = {
        "protocol": "rolling blocked inner-fold diagnostic",
        "subject": args.subject,
        "finger": args.finger,
        "fold": args.fold,
        "seed": args.seed,
        "fit_end_index": fit_end,
        "validation_end_index": validation_stop,
        "target": TARGETS[args.subject],
        "released_test_touched": False,
        "final_chronological_validation_touched": False,
        "dictionary": {
            "atom_count": args.atoms,
            "top_k": args.top_k,
            "initialization": "label-independent smooth full-spectrum coverage",
            "representations": list(model.morlet.representation_names),
            "broadband_residual_skip": True,
            "selected_atom_indices": selected,
            "selected_centers_hz": [float(centers[index]) for index in selected],
            "selected_widths_seconds": [float(widths[index]) for index in selected],
            "all_centers_hz": centers.tolist(),
            "all_widths_seconds": widths.tolist(),
            "gate_probabilities": gate_probabilities,
        },
        "parameter_count": parameter_count,
        "runtime_seconds": {
            "fixed_cached_stage": fixed_seconds,
            "learnable_stage": learned_seconds,
            "total_training": fixed_seconds + learned_seconds,
        },
        "fixed_morlet": {
            "best_inner_validation_raw_pcc": best_fixed_score,
            "history": fixed_history,
        },
        "learnable_morlet_topk": {
            "best_inner_validation_raw_pcc": best_learned_score,
            "improvement_over_fixed": best_learned_score - best_fixed_score,
            "history": learned_history,
        },
        "existing_fold_baselines": baseline,
    }
    torch.save({"model_state_dict": best_learned_state, "summary": summary}, output / "model.pt")
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"saved": str(output), "summary": summary}, indent=2), flush=True)


if __name__ == "__main__":
    main()
