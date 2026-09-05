#!/usr/bin/env python3
"""Leakage-safe reconstruction of the strong S3 little-finger CSP route.

The historical high-performing route combined a movement-state CSP/TCN model
with a beta/high-gamma gated model.  This script evaluates that deliberately
small family inside the existing purged event folds.  Target baselines and CSP
filters are re-fitted for every outer and inner split; released test arrays are
never opened.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import linalg
from torch.nn import functional as F

from benchmark_ridge_target_variants import lagged
from calibrate_state_gate import fit_gain, quality
from ecog_decoding.preprocessing import local_baseline_correct, paper_baseline_correct
from ecog_decoding.postprocessing import fit_nonnegative_gain, smooth_nonnegative
from ecog_decoding.training import FINGER_NAMES
from fit_oof_latent_movement_gate import (
    activity_emission,
    fit_classifier,
    intended_state,
    state_probabilities,
    temporal_features,
)
from train_beta_gamma_heads import BetaGammaHeads, transformed_energy
from train_csp_residual_ssm import CSPResidualSSM, correlation_loss, initialize_ridge
from train_event_grouped_lars_lstm import indices_from_intervals


SUBJECT = 3
LITTLE = list(FINGER_NAMES).index("little")
HISTORY = 25
OFFSET = HISTORY - 1
SAMPLES_PER_BIN = 40
VARIANTS = ("current_local", "paper_no_wta", "paper_wta_020")
FAMILIES = ("state_tcn", "beta_gamma")
GATE_STRENGTHS = (0.0, 0.25, 0.5, 0.75, 1.0)
GATE_TEMPERATURE = 2.0


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64) - np.mean(x)
    y = np.asarray(y, dtype=np.float64) - np.mean(y)
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    return float(x @ y / denominator) if denominator > 0 else 0.0


def temper_probability(probability: np.ndarray, temperature: float) -> np.ndarray:
    values = np.maximum(np.asarray(probability, dtype=np.float64), 1.0e-12)
    values = values ** (1.0 / temperature)
    return values / values.sum(axis=1, keepdims=True)


def fit_latent_little_gate(
    predictor_bank: list[np.ndarray],
    target: np.ndarray,
    training: np.ndarray,
    validation: np.ndarray,
    threshold: float = 0.10,
) -> tuple[np.ndarray, dict[str, object]]:
    """Fit a rest/finger classifier on training rows and return a soft little gate."""
    features = temporal_features(np.concatenate(predictor_bank, axis=1))
    state = intended_state(target, threshold)
    training_mask = np.zeros(target.shape[0], dtype=bool)
    training_mask[np.asarray(training, dtype=np.int64)] = True
    scaler, classifier = fit_classifier(features, state, training_mask)
    probability = temper_probability(
        state_probabilities(scaler, classifier, features), GATE_TEMPERATURE
    )
    activity = activity_emission(state, target, training_mask, threshold)
    little_gate = probability @ activity[:, LITTLE]
    audit = {
        "temperature": GATE_TEMPERATURE,
        "classifier_classes": classifier.classes_.astype(int).tolist(),
        "training_state_accuracy": float(
            np.mean(np.argmax(probability[training], axis=1) == state[training])
        ),
        "validation_state_accuracy": float(
            np.mean(np.argmax(probability[validation], axis=1) == state[validation])
        ),
        "training_state_counts": np.bincount(state[training], minlength=6).tolist(),
        "activity_emission": activity.tolist(),
        "validation_gate_mean": float(np.mean(little_gate[validation])),
    }
    return little_gate, audit


def apply_little_gate(
    prediction: np.ndarray, little_gate: np.ndarray, strength: float
) -> np.ndarray:
    factor = np.maximum((1.0 - strength) + strength * little_gate, 0.0)
    return np.asarray(prediction) * factor


def constrained_post_gate_gain(
    gated: np.ndarray,
    ungated: np.ndarray,
    target: np.ndarray,
    threshold: float = 0.10,
) -> tuple[float, dict[str, float]]:
    """Restore movement scale without undoing the gate's rest suppression."""
    moving = target >= threshold
    resting = ~moving
    movement_gain = fit_gain(gated[moving], target[moving]) if moving.any() else 1.0
    gated_rest = float(np.sqrt(np.mean(np.square(gated[resting]))))
    ungated_rest = float(np.sqrt(np.mean(np.square(ungated[resting]))))
    rest_limit = ungated_rest / max(gated_rest, 1.0e-8)
    predicted_peak = float(np.quantile(gated[moving], 0.95)) if moving.any() else 0.0
    target_peak = float(np.quantile(target[moving], 0.95)) if moving.any() else 0.0
    peak_limit = 1.10 * target_peak / max(predicted_peak, 1.0e-8)
    gain = float(np.clip(min(movement_gain, rest_limit, peak_limit), 0.25, 8.0))
    return gain, {
        "movement_ls_gain": float(movement_gain),
        "rest_rms_limit_gain": float(rest_limit),
        "peak_ratio_limit_gain": float(peak_limit),
    }


def intervals_from_mask(mask: np.ndarray) -> list[list[int]]:
    padded = np.r_[False, np.asarray(mask, dtype=bool), False].astype(np.int8)
    changes = np.diff(padded)
    return [
        [int(start), int(stop)]
        for start, stop in zip(
            np.flatnonzero(changes == 1), np.flatnonzero(changes == -1), strict=True
        )
    ]


def split_intervals(definition: dict[str, object], outer_fold: int) -> dict[str, dict[str, object]]:
    """Return one outer and two nested inner split definitions."""
    rows = int(definition["training_rows"])
    outer = definition["folds"][outer_fold]
    outer_training = np.zeros(rows, dtype=bool)
    outer_training[indices_from_intervals(outer["training_intervals_after_purge"])] = True
    result: dict[str, dict[str, object]] = {
        "outer": {
            "training_intervals": outer["training_intervals_after_purge"],
            "validation_intervals": outer["validation_intervals"],
        }
    }
    for inner_fold, inner in enumerate(definition["folds"]):
        if inner_fold == outer_fold:
            continue
        training = np.zeros(rows, dtype=bool)
        training[indices_from_intervals(inner["training_intervals_after_purge"])] = True
        training &= outer_training
        validation = np.zeros(rows, dtype=bool)
        validation[indices_from_intervals(inner["validation_intervals"])] = True
        validation &= outer_training
        result[f"inner{inner_fold}"] = {
            "training_intervals": intervals_from_mask(training),
            "validation_intervals": intervals_from_mask(validation),
        }
    if len(result) != 3:
        raise RuntimeError("expected one outer and two inner splits")
    return result


def correct_intervals(
    raw: np.ndarray, intervals: list[list[int]], method: str
) -> np.ndarray:
    corrected = np.full(raw.shape, np.nan, dtype=np.float64)
    for start, stop in intervals:
        segment = raw[start:stop]
        if segment.shape[0] < 3:
            raise RuntimeError(f"baseline interval is too short: {start}:{stop}")
        if method == "local":
            values = local_baseline_correct(
                segment,
                sampling_rate_hz=25.0,
                window_seconds=4.0,
                quantile=0.20,
                smoothing_seconds=0.16,
            ).corrected
        elif method == "paper":
            values = paper_baseline_correct(segment, smoothness=1.0e5).corrected
        else:
            raise ValueError(method)
        corrected[start:stop] = values
    return corrected


def scale_from_training(corrected: np.ndarray, training: np.ndarray) -> np.ndarray:
    scale = np.quantile(corrected[training], 0.995, axis=0)
    floor = max(float(np.max(scale)) * 0.01, np.finfo(np.float64).eps)
    return np.maximum(scale, floor)


def normalize_split(
    raw: np.ndarray,
    training_intervals: list[list[int]],
    validation_intervals: list[list[int]],
    method: str,
) -> np.ndarray:
    training = indices_from_intervals(training_intervals)
    validation = indices_from_intervals(validation_intervals)
    train_corrected = correct_intervals(raw, training_intervals, method)
    validation_corrected = correct_intervals(raw, validation_intervals, method)
    scale = scale_from_training(train_corrected, training)
    result = np.full(raw.shape, np.nan, dtype=np.float32)
    result[training] = np.clip(train_corrected[training] / scale, 0.0, 2.0)
    result[validation] = np.clip(validation_corrected[validation] / scale, 0.0, 2.0)
    return result


def make_target(
    raw: np.ndarray,
    training_intervals: list[list[int]],
    validation_intervals: list[list[int]],
    variant: str,
) -> np.ndarray:
    return make_targets(
        raw, training_intervals, validation_intervals, (variant,)
    )[variant]


def make_targets(
    raw: np.ndarray,
    training_intervals: list[list[int]],
    validation_intervals: list[list[int]],
    variants: tuple[str, ...] | list[str],
) -> dict[str, np.ndarray]:
    """Build requested targets while sharing their expensive baseline fits."""
    local = normalize_split(
        raw, training_intervals, validation_intervals, method="local"
    )
    results: dict[str, np.ndarray] = {}
    if "current_local" in variants:
        results["current_local"] = local
    paper_variants = [variant for variant in variants if variant != "current_local"]
    if paper_variants:
        paper = normalize_split(
            raw, training_intervals, validation_intervals, method="paper"
        )
        for variant in paper_variants:
            target = local.copy()
            little = paper[:, LITTLE].copy()
            if variant == "paper_wta_020":
                thresholded = paper.copy()
                thresholded[thresholded < 0.20] = 0.0
                winner = np.argmax(np.nan_to_num(thresholded, nan=-np.inf), axis=1)
                little[winner != LITTLE] = 0.0
            elif variant != "paper_no_wta":
                raise ValueError(variant)
            target[:, LITTLE] = little
            results[variant] = target
    return results


def regularized_covariance(values: np.ndarray, shrinkage: float = 0.05) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    values -= values.mean(axis=0, keepdims=True)
    covariance = values.T @ values / max(1, values.shape[0] - 1)
    isotropic = np.trace(covariance) / covariance.shape[0]
    return (1.0 - shrinkage) * covariance + shrinkage * isotropic * np.eye(
        covariance.shape[0]
    )


def csp_weights(
    filtered_bins: np.ndarray,
    target: np.ndarray,
    training: np.ndarray,
    components_per_tail: int = 2,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    # Rows outside this split are NaN and must be neither rest nor movement.
    rest = np.max(np.nan_to_num(target, nan=np.inf), axis=1) < 0.05
    rows: list[np.ndarray] = []
    audit: list[dict[str, object]] = []
    for finger, name in enumerate(FINGER_NAMES):
        active = target[:, finger] > 0.20
        active_rows = training[active[training]]
        rest_rows = training[rest[training]]
        if active_rows.size < 2 or rest_rows.size < 2:
            raise RuntimeError(f"too few CSP bins for {name}")
        active_values = filtered_bins[active_rows + OFFSET].reshape(
            -1, filtered_bins.shape[-1]
        )
        rest_values = filtered_bins[rest_rows + OFFSET].reshape(
            -1, filtered_bins.shape[-1]
        )
        active_covariance = regularized_covariance(active_values)
        rest_covariance = regularized_covariance(rest_values)
        eigenvalues, eigenvectors = linalg.eigh(
            active_covariance,
            active_covariance + rest_covariance,
            check_finite=False,
        )
        order = np.r_[
            np.arange(components_per_tail),
            np.arange(eigenvalues.size - components_per_tail, eigenvalues.size),
        ]
        weights = eigenvectors[:, order].T
        weights /= np.linalg.norm(weights, axis=1, keepdims=True).clip(min=1.0e-12)
        rows.append(weights.astype(np.float32))
        audit.append(
            {
                "finger": name,
                "active_bins": int(active_rows.size),
                "rest_bins": int(rest_rows.size),
                "eigenvalues": eigenvalues[order].tolist(),
            }
        )
    return np.concatenate(rows), audit


def binned_energy(
    filtered: np.ndarray, weights: np.ndarray, device: torch.device | None = None
) -> np.ndarray:
    if device is None:
        projected = np.asarray(filtered) @ weights.T
        bins = projected.shape[0] // SAMPLES_PER_BIN
        values = projected[: bins * SAMPLES_PER_BIN].reshape(
            bins, SAMPLES_PER_BIN, weights.shape[0]
        )
        return np.log1p(np.sqrt(np.sum(values * values, axis=1))).astype(np.float32)
    with torch.inference_mode():
        filtered_tensor = torch.tensor(
            np.asarray(filtered), dtype=torch.float32, device=device
        )
        weights_tensor = torch.as_tensor(weights, dtype=torch.float32, device=device)
        projected = filtered_tensor @ weights_tensor.T
        bins = projected.shape[0] // SAMPLES_PER_BIN
        values = projected[: bins * SAMPLES_PER_BIN].reshape(
            bins, SAMPLES_PER_BIN, weights.shape[0]
        )
        energy = torch.log1p(torch.sqrt(torch.sum(values.square(), dim=1)))
        result = energy.float().cpu().numpy()
    del filtered_tensor, weights_tensor, projected, values, energy
    torch.cuda.empty_cache()
    return result


def prepare(args: argparse.Namespace) -> None:
    definition = json.loads(
        (args.fold_root / "sub3" / "little" / "folds.json").read_text()
    )
    rows = int(definition["training_rows"])
    raw_full = np.load(args.prepared_root / "sub3" / "train_glove_25hz_raw.npy")
    raw = np.asarray(raw_full[OFFSET : OFFSET + rows], dtype=np.float64)
    filtered = np.load(
        args.band_cache / "sub3" / "train_filtered_bands.npy", mmap_mode="r"
    )
    if filtered.shape[0] != 7:
        raise RuntimeError(f"expected seven carrier bands, got {filtered.shape[0]}")
    bin_count = filtered.shape[1] // SAMPLES_PER_BIN
    if bin_count < rows + OFFSET:
        raise RuntimeError("carrier-band cache is shorter than the event-fold data")
    filtered_bins = filtered[:, : bin_count * SAMPLES_PER_BIN].reshape(
        filtered.shape[0], bin_count, SAMPLES_PER_BIN, filtered.shape[-1]
    )
    projection_device = torch.device(args.projection_device)
    records: list[dict[str, object]] = []
    for outer_fold in args.folds:
        for split_name, split in split_intervals(definition, outer_fold).items():
            missing_variants = []
            for variant in args.variants:
                destination = args.cache_root / variant / f"outer{outer_fold}" / split_name
                summary_path = destination / "summary.json"
                energy_path = destination / "energy.npy"
                target_path = destination / "target.npy"
                if summary_path.exists() and energy_path.exists() and target_path.exists():
                    print(f"cached {variant} outer={outer_fold} split={split_name}", flush=True)
                    continue
                missing_variants.append(variant)
            if not missing_variants:
                continue
            targets = make_targets(
                raw,
                split["training_intervals"],
                split["validation_intervals"],
                missing_variants,
            )
            for variant in missing_variants:
                destination = args.cache_root / variant / f"outer{outer_fold}" / split_name
                destination.mkdir(parents=True, exist_ok=True)
                training = indices_from_intervals(split["training_intervals"])
                validation = indices_from_intervals(split["validation_intervals"])
                records.append(
                    {
                        "variant": variant,
                        "outer_fold": outer_fold,
                        "split_name": split_name,
                        "split": split,
                        "destination": destination,
                        "training": training,
                        "validation": validation,
                        "target": targets[variant],
                        "energies": [],
                        "filter_audit": [],
                    }
                )
    if not records:
        return
    # Each split has different supervised CSP weights, but the expensive full
    # recording projection is shared. Concatenate all weights and perform one
    # matrix multiplication per carrier band rather than one per split.
    filters_per_split = len(FINGER_NAMES) * 4
    for band in range(filtered_bins.shape[0]):
        weight_parts = []
        for record in records:
            weights, audit = csp_weights(
                filtered_bins[band], record["target"], record["training"]
            )
            weight_parts.append(weights)
            record["filter_audit"].append({"band": band, "filters": audit})
        combined = np.concatenate(weight_parts, axis=0)
        combined_energy = binned_energy(filtered[band], combined, projection_device)
        for index, record in enumerate(records):
            start = index * filters_per_split
            stop = start + filters_per_split
            record["energies"].append(combined_energy[:, start:stop].copy())
        print(
            f"projected band={band} for {len(records)} split-specific CSP banks",
            flush=True,
        )
    for record in records:
        energy = np.concatenate(record["energies"], axis=1)
        destination = record["destination"]
        np.save(destination / "energy.npy", energy, allow_pickle=False)
        np.save(destination / "target.npy", record["target"], allow_pickle=False)
        split = record["split"]
        report = {
            "protocol": "split-local target baseline and training-only CSP",
            "subject": SUBJECT,
            "finger": "little",
            "variant": record["variant"],
            "outer_fold": record["outer_fold"],
            "split": record["split_name"],
            "training_intervals": split["training_intervals"],
            "validation_intervals": split["validation_intervals"],
            "training_bins": int(record["training"].size),
            "validation_bins": int(record["validation"].size),
            "energy_shape": list(energy.shape),
            "filter_audit": record["filter_audit"],
            "batched_projection": True,
            "released_test_touched": False,
        }
        (destination / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        print(
            f"prepared {record['variant']} outer={record['outer_fold']} "
            f"split={record['split_name']} energy={energy.shape}",
            flush=True,
        )


@torch.inference_mode()
def predict_components(
    model: torch.nn.Module, family: str, energy: torch.Tensor
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    if family == "state_tcn":
        amplitude, logits = model.forward_components(energy)
        primary = amplitude
    elif family == "beta_gamma":
        primary, _, logits, _ = model.forward_components(energy)
    else:
        raise ValueError(family)
    return (
        primary.squeeze(0).float().cpu().numpy(),
        torch.sigmoid(logits).squeeze(0).float().cpu().numpy(),
    )


def build_model(
    family: str,
    energy: np.ndarray,
    target: np.ndarray,
    training: np.ndarray,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, object]]:
    if family == "state_tcn":
        model = CSPResidualSSM(
            energy.shape[1], HISTORY, temporal_backbone="tcn"
        ).to(device)
    elif family == "beta_gamma":
        model = BetaGammaHeads(
            energy.shape[1], HISTORY, "gated", gate_floor=0.10
        ).to(device)
    else:
        raise ValueError(family)
    direct_energy = transformed_energy(model, energy) if family == "beta_gamma" else energy
    features = lagged(direct_energy, HISTORY)
    audit = initialize_ridge(
        model, features[training], target[training], 512, device
    )
    return model, audit


def train_model(
    *,
    family: str,
    energy: np.ndarray,
    target: np.ndarray,
    training: np.ndarray,
    validation: np.ndarray,
    raw: np.ndarray,
    epochs: int,
    seed: int,
    device: torch.device,
    compile_model: bool,
    select_best: bool = True,
) -> dict[str, object]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model, ridge_audit = build_model(family, energy, target, training, device)
    energy_tensor = torch.as_tensor(
        np.array(energy, dtype=np.float32, copy=True), device=device
    ).unsqueeze(0)
    target_filled = np.nan_to_num(target, nan=0.0).astype(np.float32)
    target_tensor = torch.as_tensor(target_filled, device=device).unsqueeze(0)
    training_tensor = torch.as_tensor(training, dtype=torch.long, device=device)
    initial, _ = predict_components(model, family, energy_tensor)
    initial_pcc = (
        pearson(initial[validation, LITTLE], raw[validation, LITTLE])
        if select_best
        else None
    )
    executable = torch.compile(model, mode="reduce-overhead") if compile_model else model
    if family == "state_tcn":
        direct_ids = {id(parameter) for parameter in model.direct.parameters()}
        residual = [parameter for parameter in model.parameters() if id(parameter) not in direct_ids]
        optimizer = torch.optim.AdamW(
            [
                {"params": model.direct.parameters(), "lr": 2.0e-5},
                {"params": residual, "lr": 5.0e-4},
            ],
            weight_decay=1.0e-4,
        )
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=5.0e-4, weight_decay=1.0e-4)
    state = (target_filled >= 0.10).astype(np.float32)
    training_state = state[training]
    state_tensor = torch.as_tensor(training_state, device=device).unsqueeze(0)
    movement_counts = state_tensor.sum(dim=1)
    pos_weight = (
        (state_tensor.shape[1] - movement_counts) / movement_counts.clamp_min(1)
    ).squeeze(0)
    best_epoch = 0
    best_score = (
        quality(initial[validation, LITTLE], target[validation, LITTLE], 0.10)[
            "quality_score"
        ]
        if select_best
        else float("nan")
    )
    best_state = copy.deepcopy(model.state_dict())
    history: list[dict[str, float]] = []
    stale = 0
    for epoch in range(1, epochs + 1):
        model.train()
        if family == "state_tcn":
            amplitude, logits = executable.forward_components(energy_tensor)
            estimate = amplitude.index_select(1, training_tensor)
            logits = logits.index_select(1, training_tensor)
            truth = target_tensor.index_select(1, training_tensor)
            weights = 1.0 + 2.0 * state_tensor
            level = torch.mean(weights * (estimate - truth).square())
            state_loss = F.binary_cross_entropy_with_logits(
                logits, state_tensor, pos_weight=pos_weight
            )
            derivative_correlation = correlation_loss(
                estimate[:, 1:] - estimate[:, :-1],
                truth[:, 1:] - truth[:, :-1],
            )
            loss = level + 0.25 * state_loss + 0.10 * derivative_correlation
        else:
            estimate_all, amplitude_all, logits_all, rest_all = executable.forward_components(
                energy_tensor
            )
            estimate = estimate_all.index_select(1, training_tensor)
            amplitude = amplitude_all.index_select(1, training_tensor)
            logits = logits_all.index_select(1, training_tensor)
            rest = rest_all.index_select(1, training_tensor)
            truth = target_tensor.index_select(1, training_tensor)
            weights = 1.0 + 2.0 * state_tensor
            level = torch.mean(weights * (estimate - truth).square())
            state_loss = F.binary_cross_entropy_with_logits(
                logits, state_tensor, pos_weight=pos_weight
            )
            amplitude_loss = torch.sum(
                state_tensor * (amplitude - truth).square()
            ) / state_tensor.sum().clamp_min(1)
            derivative_correlation = correlation_loss(
                estimate[:, 1:] - estimate[:, :-1],
                truth[:, 1:] - truth[:, :-1],
            )
            rest_penalty = torch.sum(
                (1.0 - state_tensor) * rest.square()
            ) / (1.0 - state_tensor).sum().clamp_min(1)
            loss = (
                level
                + 0.25 * state_loss
                + 0.25 * amplitude_loss
                + 0.10 * derivative_correlation
                + 0.05 * rest_penalty
            )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if select_best and (epoch == 1 or epoch % 5 == 0 or epoch == epochs):
            prediction, _ = predict_components(model, family, energy_tensor)
            metrics = quality(
                prediction[validation, LITTLE], target[validation, LITTLE], 0.10
            )
            score = float(metrics["quality_score"])
            history.append(
                {
                    "epoch": epoch,
                    "loss": float(loss.detach()),
                    "gradient_norm": float(gradient_norm),
                    "little_quality": score,
                    "little_raw_pcc": pearson(
                        prediction[validation, LITTLE], raw[validation, LITTLE]
                    ),
                }
            )
            if select_best and score > best_score:
                best_score = score
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            elif select_best:
                stale += 1
            if select_best and stale >= 8:
                break
    if not select_best:
        best_epoch = epochs
        best_state = copy.deepcopy(model.state_dict())
        final_prediction, _ = predict_components(model, family, energy_tensor)
        best_score = float(
            quality(
                final_prediction[validation, LITTLE],
                target[validation, LITTLE],
                0.10,
            )["quality_score"]
        )
    model.load_state_dict(best_state)
    prediction, probability = predict_components(model, family, energy_tensor)
    return {
        "model": model,
        "prediction": prediction,
        "probability": probability,
        "best_epoch": best_epoch,
        "best_quality": best_score,
        "initial_raw_pcc": initial_pcc,
        "ridge_audit": ridge_audit,
        "history": history,
    }


def calibration_candidates(
    family: str, prediction: np.ndarray, probability: np.ndarray
) -> dict[str, np.ndarray]:
    base = prediction[:, LITTLE]
    candidates = {"continuous": base}
    if family == "state_tcn":
        gate = 0.25 + 0.75 * np.sqrt(np.clip(probability[:, LITTLE], 0.0, 1.0))
        candidates["soft_f0.25_g0.5"] = base * gate
    return candidates


def choose_calibration(
    family: str,
    prediction: np.ndarray,
    probability: np.ndarray,
    target: np.ndarray,
) -> dict[str, object]:
    best: tuple[float, str, str, float] | None = None
    for mode, value in calibration_candidates(family, prediction, probability).items():
        moving = target >= 0.10
        gains = {"global_ls": fit_gain(value, target)}
        if moving.any():
            gains["movement_ls"] = fit_gain(value[moving], target[moving])
        for gain_name, gain in gains.items():
            score = quality(gain * value, target, 0.10)["quality_score"]
            candidate = (float(score), mode, gain_name, float(gain))
            if best is None or candidate[0] > best[0]:
                best = candidate
    assert best is not None
    return {
        "quality": best[0],
        "mode": best[1],
        "gain_method": best[2],
        "gain": best[3],
    }


def apply_calibration(
    spec: dict[str, object], family: str, prediction: np.ndarray, probability: np.ndarray
) -> np.ndarray:
    candidates = calibration_candidates(family, prediction, probability)
    return float(spec["gain"]) * candidates[str(spec["mode"])]


def load_split(cache_root: Path, variant: str, outer_fold: int, split: str):
    root = cache_root / variant / f"outer{outer_fold}" / split
    summary = json.loads((root / "summary.json").read_text())
    energy = np.load(root / "energy.npy", mmap_mode="r")
    target = np.load(root / "target.npy")
    training = indices_from_intervals(summary["training_intervals"])
    validation = indices_from_intervals(summary["validation_intervals"])
    return energy, target, training, validation, summary


def plot_outer(
    output: Path,
    order: np.ndarray,
    raw: np.ndarray,
    target: np.ndarray,
    candidates: dict[str, np.ndarray],
    base_blend: np.ndarray,
    gated_blend: np.ndarray,
    little_gate: np.ndarray,
) -> None:
    figure, axes = plt.subplots(4, 1, figsize=(15, 10), sharex=True)
    x = np.arange(order.size) / 25.0
    axes[0].plot(x, raw[order, LITTLE], color="#64748b", lw=0.8, label="raw glove")
    axes[0].plot(x, target[order, LITTLE], color="black", lw=0.9, label="split-local target")
    axes[0].legend(loc="upper right")
    for name, value in candidates.items():
        axes[1].plot(x, value, lw=0.8, label=name)
    axes[1].legend(loc="upper right")
    axes[2].plot(x, little_gate, color="#7c3aed", lw=0.8, label="little activity prior")
    axes[2].set_ylim(-0.02, 1.02)
    axes[2].legend(loc="upper right")
    axes[3].plot(x, target[order, LITTLE], color="black", lw=0.8, label="target")
    axes[3].plot(x, base_blend, color="#94a3b8", lw=0.7, label="ungated blend")
    axes[3].plot(x, gated_blend, color="#2563eb", lw=0.9, label="selected output")
    axes[3].legend(loc="upper right")
    for axis in axes:
        axis.grid(alpha=0.15)
    axes[-1].set_xlabel("concatenated held-out time (s)")
    figure.tight_layout()
    figure.savefig(output, dpi=170)
    plt.close(figure)


def run_fold(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)
    raw_full = np.load(args.prepared_root / "sub3" / "train_glove_25hz_raw.npy")
    raw = np.asarray(raw_full[OFFSET : OFFSET + args.rows], dtype=np.float32)
    inner_predictions: dict[str, list[np.ndarray]] = {family: [] for family in FAMILIES}
    inner_probabilities: dict[str, list[np.ndarray]] = {family: [] for family in FAMILIES}
    inner_targets: list[np.ndarray] = []
    inner_gate_values: list[np.ndarray] = []
    selected_epochs: dict[str, list[int]] = {family: [] for family in FAMILIES}
    inner_records: list[dict[str, object]] = []
    split_names = [
        path.name
        for path in sorted((args.cache_root / args.variant / f"outer{args.fold}").glob("inner*"))
    ]
    for inner_number, split_name in enumerate(split_names):
        energy, target, training, validation, _ = load_split(
            args.cache_root, args.variant, args.fold, split_name
        )
        inner_targets.append(target[validation, LITTLE])
        record: dict[str, object] = {"split": split_name, "families": {}}
        predictor_bank: list[np.ndarray] = []
        for family_index, family in enumerate(FAMILIES):
            result = train_model(
                family=family,
                energy=energy,
                target=target,
                training=training,
                validation=validation,
                raw=raw,
                epochs=args.max_epochs,
                seed=args.seed + 100 * inner_number + 10 * family_index,
                device=device,
                compile_model=args.compile,
                select_best=True,
            )
            inner_predictions[family].append(result["prediction"][validation])
            inner_probabilities[family].append(result["probability"][validation])
            predictor_bank.extend((result["prediction"], result["probability"]))
            selected_epochs[family].append(int(result["best_epoch"]))
            record["families"][family] = {
                key: result[key]
                for key in ("best_epoch", "best_quality", "initial_raw_pcc", "history")
            }
            del result["model"]
            torch.cuda.empty_cache()
        if args.latent_gate:
            little_gate, gate_audit = fit_latent_little_gate(
                predictor_bank, target, training, validation
            )
            inner_gate_values.append(little_gate[validation])
            record["latent_gate"] = gate_audit
        else:
            inner_gate_values.append(np.ones(validation.size, dtype=np.float64))
        inner_records.append(record)
    pooled_target = np.concatenate(inner_targets)
    calibration: dict[str, dict[str, object]] = {}
    calibrated_inner: dict[str, np.ndarray] = {}
    for family in FAMILIES:
        prediction = np.concatenate(inner_predictions[family])
        probability = np.concatenate(inner_probabilities[family])
        calibration[family] = choose_calibration(
            family, prediction, probability, pooled_target
        )
        calibrated_inner[family] = apply_calibration(
            calibration[family], family, prediction, probability
        )
    pooled_gate = np.concatenate(inner_gate_values)
    blend_scores: dict[str, float] = {}
    best_blend: tuple[float, float, float, str, float] | None = None
    for state_weight in np.linspace(0.0, 1.0, 5):
        base_estimate = (
            state_weight * calibrated_inner["state_tcn"]
            + (1.0 - state_weight) * calibrated_inner["beta_gamma"]
        )
        strengths = GATE_STRENGTHS if args.latent_gate else (0.0,)
        for strength in strengths:
            unscaled = apply_little_gate(base_estimate, pooled_gate, strength)
            if args.post_gate_gain:
                constrained_gain, _ = constrained_post_gate_gain(
                    unscaled, base_estimate, pooled_target
                )
                post_gains = {
                    "identity": 1.0,
                    "constrained_movement_ls": constrained_gain,
                }
            else:
                post_gains = {"identity": 1.0}
            for gain_name, post_gain in post_gains.items():
                estimate = post_gain * unscaled
                metrics = quality(estimate, pooled_target, 0.10)
                score = float(metrics["quality_score"])
                key = (
                    f"state={state_weight:.2f},gate={strength:.2f},"
                    f"gain={gain_name}"
                )
                blend_scores[key] = score
                if best_blend is None or score > best_blend[0]:
                    best_blend = (
                        score,
                        float(state_weight),
                        float(strength),
                        gain_name,
                        float(post_gain),
                    )
    assert best_blend is not None
    energy, target, training, validation, _ = load_split(
        args.cache_root, args.variant, args.fold, "outer"
    )
    outer_candidates: dict[str, np.ndarray] = {}
    outer_records: dict[str, object] = {}
    outer_predictor_bank: list[np.ndarray] = []
    for family_index, family in enumerate(FAMILIES):
        positive_epochs = [value for value in selected_epochs[family] if value > 0]
        epochs = int(np.rint(np.median(positive_epochs))) if positive_epochs else 0
        result = train_model(
            family=family,
            energy=energy,
            target=target,
            training=training,
            validation=validation,
            raw=raw,
            epochs=epochs,
            seed=args.seed + 1000 + 10 * family_index,
            device=device,
            compile_model=args.compile,
            select_best=False,
        )
        outer_candidates[family] = apply_calibration(
            calibration[family], family, result["prediction"][validation], result["probability"][validation]
        )
        outer_predictor_bank.extend((result["prediction"], result["probability"]))
        outer_records[family] = {
            "refit_epochs": epochs,
            "inner_selected_epochs": selected_epochs[family],
            "calibration": calibration[family],
            "initial_raw_pcc": result["initial_raw_pcc"],
            "outer_raw_pcc": pearson(
                outer_candidates[family], raw[validation, LITTLE]
            ),
            "outer_target_pcc": pearson(
                outer_candidates[family], target[validation, LITTLE]
            ),
            "history": result["history"],
        }
        del result["model"]
        torch.cuda.empty_cache()
    state_weight = best_blend[1]
    gate_strength = best_blend[2]
    post_gain_method = best_blend[3]
    post_gain = best_blend[4]
    base_blend = (
        state_weight * outer_candidates["state_tcn"]
        + (1.0 - state_weight) * outer_candidates["beta_gamma"]
    )
    if args.latent_gate:
        outer_gate, outer_gate_audit = fit_latent_little_gate(
            outer_predictor_bank, target, training, validation
        )
        validation_gate = outer_gate[validation]
    else:
        validation_gate = np.ones(validation.size, dtype=np.float64)
        outer_gate_audit = None
    gated_unscaled = apply_little_gate(base_blend, validation_gate, gate_strength)
    blend = post_gain * gated_unscaled
    _, outer_post_gain_limits = constrained_post_gate_gain(
        gated_unscaled, base_blend, target[validation, LITTLE]
    )
    output = args.output_root / args.variant / f"fold{args.fold}" / f"seed{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "validation_indices.npy", validation, allow_pickle=False)
    np.save(output / "validation_prediction.npy", blend.astype(np.float32), allow_pickle=False)
    np.save(output / "validation_prediction_ungated.npy", base_blend.astype(np.float32), allow_pickle=False)
    np.save(
        output / "validation_prediction_gated_unscaled.npy",
        gated_unscaled.astype(np.float32),
        allow_pickle=False,
    )
    np.save(output / "validation_little_gate.npy", validation_gate.astype(np.float32), allow_pickle=False)
    np.save(output / "validation_raw_target.npy", raw[validation, LITTLE], allow_pickle=False)
    np.save(output / "validation_cleaned_target.npy", target[validation, LITTLE], allow_pickle=False)
    for family, value in outer_candidates.items():
        np.save(output / f"validation_{family}.npy", value.astype(np.float32), allow_pickle=False)
    plot_outer(
        output / "validation_trajectories.png",
        validation,
        raw,
        target,
        outer_candidates,
        base_blend,
        blend,
        validation_gate,
    )
    report = {
        "protocol": "nested purged event folds; split-local baselines and CSP; no released test",
        "subject": SUBJECT,
        "finger": "little",
        "variant": args.variant,
        "fold": args.fold,
        "seed": args.seed,
        "families": outer_records,
        "inner_records": inner_records,
        "inner_blend_scores": blend_scores,
        "selected_state_tcn_weight": state_weight,
        "latent_gate_enabled": bool(args.latent_gate),
        "post_gate_gain_enabled": bool(args.post_gate_gain),
        "selected_latent_gate_strength": gate_strength,
        "selected_post_gate_gain_method": post_gain_method,
        "selected_post_gate_gain": post_gain,
        "outer_post_gate_gain_limits_for_audit_only": outer_post_gain_limits,
        "outer_latent_gate": outer_gate_audit,
        "outer_ungated_raw_pcc": pearson(base_blend, raw[validation, LITTLE]),
        "outer_ungated_target_pcc": pearson(base_blend, target[validation, LITTLE]),
        "outer_ungated_quality": quality(base_blend, target[validation, LITTLE], 0.10),
        "outer_raw_pcc": pearson(blend, raw[validation, LITTLE]),
        "outer_target_pcc": pearson(blend, target[validation, LITTLE]),
        "outer_quality": quality(blend, target[validation, LITTLE], 0.10),
        "runtime_seconds": time.perf_counter() - started,
        "torch_compile": bool(args.compile),
        "compile_mode": "reduce-overhead" if args.compile else None,
        "released_test_touched": False,
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "variant", "fold", "outer_raw_pcc", "outer_target_pcc", "runtime_seconds"
                )
            },
            indent=2,
        ),
        flush=True,
    )


def summarize(args: argparse.Namespace) -> None:
    report: dict[str, object] = {
        "protocol": "nested purged event OOF S3-little reconstruction",
        "released_test_touched": False,
        "variants": {},
    }
    for variant in args.variants:
        fold_reports = []
        predictions = []
        raw_targets = []
        cleaned_targets = []
        indices = []
        for fold in args.folds:
            root = args.output_root / variant / f"fold{fold}" / f"seed{args.seed}"
            fold_reports.append(json.loads((root / "summary.json").read_text()))
            predictions.append(np.load(root / "validation_prediction.npy"))
            raw_targets.append(np.load(root / "validation_raw_target.npy"))
            cleaned_targets.append(np.load(root / "validation_cleaned_target.npy"))
            indices.append(np.load(root / "validation_indices.npy"))
        order = np.argsort(np.concatenate(indices))
        prediction = np.concatenate(predictions)[order]
        raw = np.concatenate(raw_targets)[order]
        cleaned = np.concatenate(cleaned_targets)[order]
        report["variants"][variant] = {
            "mean_fold_raw_pcc": float(np.mean([item["outer_raw_pcc"] for item in fold_reports])),
            "fold_raw_pcc": [float(item["outer_raw_pcc"]) for item in fold_reports],
            "stitched_raw_pcc": pearson(prediction, raw),
            "stitched_target_pcc": pearson(prediction, cleaned),
            "folds": fold_reports,
        }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                name: {
                    key: values[key]
                    for key in ("mean_fold_raw_pcc", "stitched_raw_pcc", "stitched_target_pcc")
                }
                for name, values in report["variants"].items()
            },
            indent=2,
        ),
        flush=True,
    )


def prepare_full(args: argparse.Namespace) -> None:
    """Fit the selected target and CSP bank on all labeled development data."""
    destination = args.full_cache / args.variant
    if all((destination / name).exists() for name in ("train_energy.npy", "test_energy.npy", "target.npy", "summary.json")):
        print(f"cached full-development features: {destination}", flush=True)
        return
    destination.mkdir(parents=True, exist_ok=True)
    raw_full = np.load(args.prepared_root / "sub3" / "train_glove_25hz_raw.npy")
    raw = np.asarray(raw_full[OFFSET:], dtype=np.float64)
    rows = raw.shape[0]
    target = make_target(raw, [[0, rows]], [], args.variant)
    training = np.arange(rows, dtype=np.int64)
    band_root = args.band_cache / "sub3"
    train_bands = np.load(band_root / "train_filtered_bands.npy", mmap_mode="r")
    test_bands = np.load(band_root / "test_filtered_bands.npy", mmap_mode="r")
    train_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    filter_audit: list[dict[str, object]] = []
    projection_device = torch.device(args.projection_device)
    for band in range(train_bands.shape[0]):
        train_bins = np.asarray(train_bands[band]).reshape(
            -1, SAMPLES_PER_BIN, train_bands.shape[-1]
        )
        weights, audit = csp_weights(train_bins, target, training)
        train_parts.append(
            binned_energy(train_bands[band], weights, projection_device)
        )
        test_parts.append(
            binned_energy(test_bands[band], weights, projection_device)
        )
        filter_audit.append({"band": band, "filters": audit})
        print(f"prepared full-development CSP band={band}", flush=True)
    train_energy = np.concatenate(train_parts, axis=1)
    test_energy = np.concatenate(test_parts, axis=1)
    np.save(destination / "train_energy.npy", train_energy, allow_pickle=False)
    np.save(destination / "test_energy.npy", test_energy, allow_pickle=False)
    np.save(destination / "target.npy", target, allow_pickle=False)
    report = {
        "protocol": "full-development paper-baseline target and CSP; test ECoG transformed without test labels",
        "subject": SUBJECT,
        "finger": "little",
        "variant": args.variant,
        "development_rows": rows,
        "train_energy_shape": list(train_energy.shape),
        "test_energy_shape": list(test_energy.shape),
        "filter_audit": filter_audit,
        "released_test_labels_loaded": False,
    }
    (destination / "summary.json").write_text(json.dumps(report, indent=2) + "\n")


def frozen_refit_spec(cv_root: Path, variant: str) -> dict[str, object]:
    folds = [
        json.loads((cv_root / variant / f"fold{fold}" / "seed0" / "summary.json").read_text())
        for fold in (0, 1, 2)
    ]
    families: dict[str, object] = {}
    for family in FAMILIES:
        records = [fold["families"][family] for fold in folds]
        epochs = [int(record["refit_epochs"]) for record in records]
        modes = [str(record["calibration"]["mode"]) for record in records]
        mode = Counter(modes).most_common(1)[0][0]
        families[family] = {
            "epochs": int(np.rint(np.median(epochs))),
            "calibration": {
                "mode": mode,
                "gain_method": "median_nested_fold_gain",
                "gain": float(np.median([record["calibration"]["gain"] for record in records])),
            },
            "fold_epochs": epochs,
            "fold_modes": modes,
        }
    return {
        "source": str(cv_root),
        "variant": variant,
        "families": families,
        "state_tcn_weight": float(np.median([fold["selected_state_tcn_weight"] for fold in folds])),
        "latent_gate_strength": float(np.median([fold["selected_latent_gate_strength"] for fold in folds])),
        "latent_gate_temperature": GATE_TEMPERATURE,
        "post_gate_gain": 1.0,
        "released_test_used": False,
    }


def run_refit(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    torch.set_float32_matmul_precision("high")
    device = torch.device(args.device)
    cache = args.full_cache / args.variant
    train_energy = np.load(cache / "train_energy.npy", mmap_mode="r")
    test_energy = np.load(cache / "test_energy.npy", mmap_mode="r")
    target = np.load(cache / "target.npy")
    raw = np.load(args.prepared_root / "sub3" / "train_glove_25hz_raw.npy")[OFFSET:]
    training = np.arange(target.shape[0], dtype=np.int64)
    spec = frozen_refit_spec(args.cv_root, args.variant)
    train_bank: list[np.ndarray] = []
    test_bank: list[np.ndarray] = []
    train_candidates: dict[str, np.ndarray] = {}
    test_candidates: dict[str, np.ndarray] = {}
    family_reports: dict[str, object] = {}
    test_tensor = torch.as_tensor(
        np.array(test_energy, dtype=np.float32, copy=True), device=device
    ).unsqueeze(0)
    for family_index, family in enumerate(FAMILIES):
        family_spec = spec["families"][family]
        result = train_model(
            family=family,
            energy=train_energy,
            target=target,
            training=training,
            validation=training,
            raw=raw,
            epochs=int(family_spec["epochs"]),
            seed=args.seed + 10 * family_index,
            device=device,
            compile_model=args.compile,
            select_best=False,
        )
        test_prediction, test_probability = predict_components(
            result["model"], family, test_tensor
        )
        train_bank.extend((result["prediction"], result["probability"]))
        test_bank.extend((test_prediction, test_probability))
        train_candidates[family] = apply_calibration(
            family_spec["calibration"], family, result["prediction"], result["probability"]
        )
        test_candidates[family] = apply_calibration(
            family_spec["calibration"], family, test_prediction, test_probability
        )
        family_reports[family] = {
            "epochs": int(family_spec["epochs"]),
            "calibration": family_spec["calibration"],
            "training_history": result["history"],
        }
        del result["model"]
        torch.cuda.empty_cache()
    state = intended_state(target, 0.10)
    mask = np.ones(target.shape[0], dtype=bool)
    train_features = temporal_features(np.concatenate(train_bank, axis=1))
    test_features = temporal_features(np.concatenate(test_bank, axis=1))
    scaler, classifier = fit_classifier(train_features, state, mask)
    activity = activity_emission(state, target, mask, 0.10)
    train_probability = temper_probability(
        state_probabilities(scaler, classifier, train_features), GATE_TEMPERATURE
    )
    test_probability = temper_probability(
        state_probabilities(scaler, classifier, test_features), GATE_TEMPERATURE
    )
    train_gate = train_probability @ activity[:, LITTLE]
    test_gate = test_probability @ activity[:, LITTLE]
    weight = float(spec["state_tcn_weight"])
    train_base = weight * train_candidates["state_tcn"] + (1.0 - weight) * train_candidates["beta_gamma"]
    test_base = weight * test_candidates["state_tcn"] + (1.0 - weight) * test_candidates["beta_gamma"]
    strength = float(spec["latent_gate_strength"])
    train_prediction = apply_little_gate(train_base, train_gate, strength)
    test_prediction = apply_little_gate(test_base, test_gate, strength)
    output = args.output_root / f"seed{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "development_prediction.npy", train_prediction.astype(np.float32), allow_pickle=False)
    np.save(output / "released_test_prediction.npy", test_prediction.astype(np.float32), allow_pickle=False)
    np.save(output / "released_test_little_gate.npy", test_gate.astype(np.float32), allow_pickle=False)
    report = {
        "protocol": "six-seed candidate; full-development refit after frozen nested OOF selection",
        "subject": SUBJECT,
        "finger": "little",
        "seed": args.seed,
        "spec": spec,
        "families": family_reports,
        "development_raw_pcc": pearson(train_prediction, raw[:, LITTLE]),
        "development_prediction_sd": float(np.std(train_prediction)),
        "latent_training_state_accuracy": float(
            np.mean(np.argmax(train_probability, axis=1) == state)
        ),
        "runtime_seconds": time.perf_counter() - started,
        "released_test_labels_loaded": False,
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("seed", "development_raw_pcc", "runtime_seconds")}, indent=2), flush=True)


def run_refits(args: argparse.Namespace) -> None:
    tasks = []
    for seed in args.seeds:
        destination = args.output_root / f"seed{seed}" / "summary.json"
        if destination.exists():
            continue
        command = [
            sys.executable, __file__, "refit",
            "--variant", args.variant,
            "--seed", str(seed),
            "--prepared-root", str(args.prepared_root),
            "--full-cache", str(args.full_cache),
            "--cv-root", str(args.cv_root),
            "--output-root", str(args.output_root),
        ]
        if not args.compile:
            command.append("--no-compile")
        tasks.append((seed, command))
    log_root = args.output_root / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    pending = list(tasks)
    active: list[tuple[subprocess.Popen[bytes], object, int, str]] = []
    failures = []
    while pending or active:
        busy = {item[3] for item in active}
        for gpu in args.gpus:
            if not pending or gpu in busy:
                continue
            seed, command = pending.pop(0)
            log = open(log_root / f"seed{seed}.log", "wb")
            environment = os.environ.copy()
            environment["PYTHONPATH"] = "scripts:src"
            environment["CUDA_VISIBLE_DEVICES"] = gpu
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment)
            active.append((process, log, seed, gpu))
            print(f"started seed={seed} gpu={gpu} pid={process.pid}", flush=True)
        time.sleep(1.0)
        remaining = []
        for process, log, seed, gpu in active:
            code = process.poll()
            if code is None:
                remaining.append((process, log, seed, gpu))
                continue
            log.close()
            print(f"finished seed={seed} exit={code}", flush=True)
            if code:
                failures.append(seed)
        active = remaining
    if failures:
        raise RuntimeError(f"failed refit seeds: {failures}")


def plot_released_test(path: Path, target: np.ndarray, prediction: np.ndarray) -> None:
    time_axis = np.arange(target.size) / 25.0
    figure, axes = plt.subplots(2, 1, figsize=(15, 6), sharex=True)
    axes[0].plot(time_axis, target, color="black", lw=0.8, label="raw released glove")
    axes[0].legend(loc="upper right")
    axes[1].plot(time_axis, target, color="black", lw=0.7, alpha=0.7, label="raw glove")
    axes[1].plot(time_axis, prediction, color="#2563eb", lw=0.9, label="six-seed prediction")
    axes[1].legend(loc="upper right")
    axes[1].set_xlabel("released-test time (s)")
    for axis in axes:
        axis.grid(alpha=0.15)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_released_events(path: Path, target: np.ndarray, prediction: np.ndarray) -> None:
    active = target >= 0.10
    changes = np.diff(np.r_[False, active, False].astype(np.int8))
    events = list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1), strict=True))
    selected = sorted(events, key=lambda pair: float(np.max(target[pair[0]:pair[1]])), reverse=True)[:12]
    figure, axes = plt.subplots(4, 3, figsize=(14, 10))
    for axis, (start, stop) in zip(axes.flat, selected):
        left = max(0, int(start) - 25)
        right = min(target.size, int(stop) + 25)
        time_axis = (np.arange(left, right) - start) / 25.0
        axis.plot(time_axis, target[left:right], color="black", lw=0.9, label="cleaned target")
        axis.plot(time_axis, prediction[left:right], color="#2563eb", lw=0.9, label="prediction")
        axis.axvline(0.0, color="#94a3b8", lw=0.6)
    for axis in axes.flat[len(selected):]:
        axis.set_visible(False)
    if selected:
        axes.flat[0].legend(frameon=False, fontsize=8)
    figure.suptitle("S3 little: strongest released-test movement events")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def summarize_refits(args: argparse.Namespace) -> None:
    members = []
    development_members = []
    audits = []
    for seed in args.seeds:
        root = args.output_root / f"seed{seed}"
        report = json.loads((root / "summary.json").read_text())
        prediction = np.load(root / "released_test_prediction.npy")
        eligible = bool(
            np.isfinite(prediction).all()
            and float(np.std(prediction)) > 1.0e-6
            and float(report["development_raw_pcc"]) >= 0.20
        )
        audits.append({
            "seed": seed,
            "eligible": eligible,
            "development_raw_pcc": float(report["development_raw_pcc"]),
            "prediction_sd": float(np.std(prediction)),
            "runtime_seconds": float(report["runtime_seconds"]),
        })
        if eligible:
            members.append(prediction)
            development_members.append(np.load(root / "development_prediction.npy"))
    if not members:
        raise RuntimeError("all full-development refit seeds collapsed")
    stack = np.stack(members)
    development_stack = np.stack(development_members)
    prediction_clean_coordinate = np.mean(stack, axis=0)
    development_prediction = np.mean(development_stack, axis=0)
    development_raw = np.load(args.prepared_root / "sub3" / "train_glove_25hz_raw.npy")[OFFSET:]
    raw = np.load(args.prepared_root / "sub3" / "test_glove_25hz_raw.npy")[OFFSET:]
    design = np.column_stack((development_prediction, np.ones(development_prediction.size)))
    affine_scale, affine_offset = np.linalg.lstsq(
        design, development_raw[:, LITTLE], rcond=None
    )[0]
    prediction_raw_coordinate = affine_scale * prediction_clean_coordinate + affine_offset
    test_pcc = pearson(prediction_raw_coordinate, raw[:, LITTLE])
    target = np.load(args.full_cache / args.variant / "target.npy")[:, LITTLE]
    projected_development = smooth_nonnegative(development_prediction)
    cleaned_gain = fit_nonnegative_gain(projected_development, target)
    prediction_cleaned = cleaned_gain * smooth_nonnegative(prediction_clean_coordinate)
    paper_train = paper_baseline_correct(development_raw, smoothness=1.0e5).corrected
    paper_scale = max(float(np.quantile(paper_train[:, LITTLE], 0.995)), 1.0e-8)
    paper_test = paper_baseline_correct(raw, smoothness=1.0e5).corrected
    cleaned_test_target = np.clip(paper_test[:, LITTLE] / paper_scale, 0.0, 2.0)
    ensemble = args.output_root / "ensemble"
    ensemble.mkdir(parents=True, exist_ok=True)
    np.save(ensemble / "little_released_test_prediction.npy", prediction_raw_coordinate.astype(np.float32), allow_pickle=False)
    np.save(ensemble / "little_released_test_prediction_raw_coordinate.npy", prediction_raw_coordinate.astype(np.float32), allow_pickle=False)
    np.save(ensemble / "little_released_test_prediction_cleaned.npy", prediction_cleaned.astype(np.float32), allow_pickle=False)
    plot_released_test(ensemble / "little_full_trajectory.png", raw[:, LITTLE], prediction_raw_coordinate)
    plot_released_test(ensemble / "little_cleaned_full_trajectory.png", cleaned_test_target, prediction_cleaned)
    plot_released_events(ensemble / "little_cleaned_strongest_events.png", cleaned_test_target, prediction_cleaned)
    base_scores = None
    updated_scores = None
    baseline_files = [
        args.baseline_root / "sub3" / f"{finger}_released_test_prediction.npy"
        for finger in FINGER_NAMES
    ]
    hybrid_path = args.baseline_root / "sub3" / "hybrid_released_test_prediction.npy"
    if hybrid_path.exists():
        combined = np.load(hybrid_path).copy()
    elif all(path.exists() for path in baseline_files):
        combined = np.column_stack([np.load(path) for path in baseline_files])
    else:
        combined = None
    if combined is not None:
        base_scores = [pearson(combined[:, finger], raw[:, finger]) for finger in range(5)]
        combined[:, LITTLE] = prediction_raw_coordinate
        updated_scores = [pearson(combined[:, finger], raw[:, finger]) for finger in range(5)]
        np.save(ensemble / "released_test_prediction.npy", combined.astype(np.float32), allow_pickle=False)
    diversity = [
        pearson(stack[first], stack[second])
        for first in range(stack.shape[0])
        for second in range(first + 1, stack.shape[0])
    ]
    report = {
        "protocol": "six-seed full-development ensemble; eligibility uses development predictions only",
        "subject": SUBJECT,
        "finger": "little",
        "members": audits,
        "included_seeds": [item["seed"] for item in audits if item["eligible"]],
        "collapsed_seeds": [item["seed"] for item in audits if not item["eligible"]],
        "mean_pairwise_seed_prediction_pcc": float(np.mean(diversity)) if diversity else 1.0,
        "released_test_raw_pcc": test_pcc,
        "released_test_seed_pcc": [pearson(value, raw[:, LITTLE]) for value in stack],
        "released_test_seed_pcc_sd": float(np.std([pearson(value, raw[:, LITTLE]) for value in stack], ddof=1)) if stack.shape[0] > 1 else 0.0,
        "baseline_five_finger_raw_pcc": base_scores,
        "updated_five_finger_raw_pcc": updated_scores,
        "baseline_macro_five": float(np.mean(base_scores)) if base_scores is not None else None,
        "updated_macro_five": float(np.mean(updated_scores)) if updated_scores is not None else None,
        "development_fitted_raw_affine": {
            "scale": float(affine_scale),
            "offset": float(affine_offset),
        },
        "development_fitted_cleaned_gain": float(cleaned_gain),
        "released_test_cleaned_metrics": quality(prediction_cleaned, cleaned_test_target, 0.10),
        "released_test_cleaned_negative_fraction": float(np.mean(prediction_cleaned < 0.0)),
        "released_test_used_for_selection": False,
        "released_test_role": "one-time descriptive final evaluation after frozen OOF selection",
    }
    (ensemble / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


def run_cv(args: argparse.Namespace) -> None:
    tasks = []
    for variant in args.variants:
        for fold in args.folds:
            destination = args.output_root / variant / f"fold{fold}" / f"seed{args.seed}" / "summary.json"
            if destination.exists():
                continue
            command = [
                sys.executable,
                __file__,
                "fold",
                "--variant", variant,
                "--fold", str(fold),
                "--seed", str(args.seed),
                "--rows", str(args.rows),
                "--cache-root", str(args.cache_root),
                "--output-root", str(args.output_root),
                "--prepared-root", str(args.prepared_root),
                "--max-epochs", str(args.max_epochs),
            ]
            if not args.compile:
                command.append("--no-compile")
            if args.latent_gate:
                command.append("--latent-gate")
            if args.post_gate_gain:
                command.append("--post-gate-gain")
            tasks.append((variant, fold, command))
    log_root = args.output_root / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    pending = list(tasks)
    active: list[tuple[subprocess.Popen[bytes], object, str, int, str]] = []
    failures = []
    while pending or active:
        busy = {item[4] for item in active}
        for gpu in args.gpus:
            if not pending or gpu in busy:
                continue
            variant, fold, command = pending.pop(0)
            log = open(log_root / f"{variant}_fold{fold}_seed{args.seed}.log", "wb")
            environment = os.environ.copy()
            environment["PYTHONPATH"] = "scripts:src"
            environment["CUDA_VISIBLE_DEVICES"] = gpu
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment)
            active.append((process, log, variant, fold, gpu))
            print(f"started {variant} fold={fold} gpu={gpu} pid={process.pid}", flush=True)
        time.sleep(1.0)
        remaining = []
        for process, log, variant, fold, gpu in active:
            code = process.poll()
            if code is None:
                remaining.append((process, log, variant, fold, gpu))
                continue
            log.close()
            print(f"finished {variant} fold={fold} exit={code}", flush=True)
            if code:
                failures.append(f"{variant}/fold{fold}")
        active = remaining
    if failures:
        raise RuntimeError("failed tasks: " + ", ".join(failures))


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--prepared-root", type=Path, default=Path("outputs/preprocessed_v2"))
    parser.add_argument(
        "--fold-root",
        type=Path,
        default=Path("outputs/event_stratified_folds_fulldev_targetsafe_conservative_v1"),
    )
    parser.add_argument(
        "--cache-root", type=Path, default=Path("outputs/s3_little_nested_csp_cache_v1")
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("outputs/s3_little_reconstruction_nested_v1")
    )
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=("current_local",))
    parser.add_argument("--folds", type=int, nargs="+", choices=(0, 1, 2), default=(0, 1, 2))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    add_common(prepare_parser)
    prepare_parser.add_argument(
        "--band-cache", type=Path, default=Path("/dev/shm/ecog_csp_band_cache")
    )
    prepare_parser.add_argument("--projection-device", default="cuda")
    fold_parser = subparsers.add_parser("fold")
    add_common(fold_parser)
    fold_parser.add_argument("--variant", choices=VARIANTS, required=True)
    fold_parser.add_argument("--fold", type=int, choices=(0, 1, 2), required=True)
    fold_parser.add_argument("--seed", type=int, default=0)
    fold_parser.add_argument("--rows", type=int, default=9976)
    fold_parser.add_argument("--max-epochs", type=int, default=80)
    fold_parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    fold_parser.add_argument("--device", default="cuda")
    fold_parser.add_argument(
        "--latent-gate", action=argparse.BooleanOptionalAction, default=False
    )
    fold_parser.add_argument(
        "--post-gate-gain", action=argparse.BooleanOptionalAction, default=False
    )
    cv_parser = subparsers.add_parser("cv")
    add_common(cv_parser)
    cv_parser.add_argument("--seed", type=int, default=0)
    cv_parser.add_argument("--rows", type=int, default=9976)
    cv_parser.add_argument("--max-epochs", type=int, default=80)
    cv_parser.add_argument("--gpus", nargs="+", default=("0", "1", "2"))
    cv_parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    cv_parser.add_argument(
        "--latent-gate", action=argparse.BooleanOptionalAction, default=False
    )
    cv_parser.add_argument(
        "--post-gate-gain", action=argparse.BooleanOptionalAction, default=False
    )
    summary_parser = subparsers.add_parser("summarize")
    add_common(summary_parser)
    summary_parser.add_argument("--seed", type=int, default=0)
    full_prepare_parser = subparsers.add_parser("prepare-full")
    full_prepare_parser.add_argument("--prepared-root", type=Path, default=Path("outputs/preprocessed_v2"))
    full_prepare_parser.add_argument("--band-cache", type=Path, default=Path("/dev/shm/ecog_csp_band_cache"))
    full_prepare_parser.add_argument("--full-cache", type=Path, default=Path("outputs/s3_little_full_csp_cache_v1"))
    full_prepare_parser.add_argument("--variant", choices=VARIANTS, default="paper_no_wta")
    full_prepare_parser.add_argument("--projection-device", default="cuda")
    refit_parser = subparsers.add_parser("refit")
    refit_parser.add_argument("--prepared-root", type=Path, default=Path("outputs/preprocessed_v2"))
    refit_parser.add_argument("--full-cache", type=Path, default=Path("outputs/s3_little_full_csp_cache_v1"))
    refit_parser.add_argument("--cv-root", type=Path, default=Path("outputs/s3_little_reconstruction_nested_latent_v1"))
    refit_parser.add_argument("--output-root", type=Path, default=Path("outputs/s3_little_six_seed_refit_v1"))
    refit_parser.add_argument("--variant", choices=VARIANTS, default="paper_no_wta")
    refit_parser.add_argument("--seed", type=int, required=True)
    refit_parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    refit_parser.add_argument("--device", default="cuda")
    refits_parser = subparsers.add_parser("refits")
    refits_parser.add_argument("--prepared-root", type=Path, default=Path("outputs/preprocessed_v2"))
    refits_parser.add_argument("--full-cache", type=Path, default=Path("outputs/s3_little_full_csp_cache_v1"))
    refits_parser.add_argument("--cv-root", type=Path, default=Path("outputs/s3_little_reconstruction_nested_latent_v1"))
    refits_parser.add_argument("--output-root", type=Path, default=Path("outputs/s3_little_six_seed_refit_v1"))
    refits_parser.add_argument("--variant", choices=VARIANTS, default="paper_no_wta")
    refits_parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 2, 3, 4, 5))
    refits_parser.add_argument("--gpus", nargs="+", default=("0", "1", "2", "3", "4", "5"))
    refits_parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    ensemble_parser = subparsers.add_parser("ensemble")
    ensemble_parser.add_argument("--prepared-root", type=Path, default=Path("outputs/preprocessed_v2"))
    ensemble_parser.add_argument("--full-cache", type=Path, default=Path("outputs/s3_little_full_csp_cache_v1"))
    ensemble_parser.add_argument("--output-root", type=Path, default=Path("outputs/s3_little_six_seed_refit_v1"))
    ensemble_parser.add_argument("--baseline-root", type=Path, default=Path("outputs/heterogeneous_six_seed_refit_v1/ensemble"))
    ensemble_parser.add_argument("--variant", choices=VARIANTS, default="paper_no_wta")
    ensemble_parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 2, 3, 4, 5))
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "fold":
        run_fold(args)
    elif args.command == "cv":
        run_cv(args)
    elif args.command == "summarize":
        summarize(args)
    elif args.command == "prepare-full":
        prepare_full(args)
    elif args.command == "refit":
        run_refit(args)
    elif args.command == "refits":
        run_refits(args)
    else:
        summarize_refits(args)


if __name__ == "__main__":
    main()
