from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .models import AsymmetricWaveletPacketEnergy, WaveletPacketEnergy


@dataclass(frozen=True)
class FeatureGroup:
    component: int
    band: int
    band_name: str
    selected_columns: np.ndarray
    source_indices: np.ndarray


def decode_feature_groups(
    selected_indices: np.ndarray,
    component_count: int,
    band_names: tuple[str, ...],
    energy_bins: int,
) -> list[FeatureGroup]:
    """Map flattened selected features back to component-band coalitions."""
    selected = np.asarray(selected_indices, dtype=np.int64)
    band_count = len(band_names)
    feature_count = component_count * band_count * energy_bins
    if selected.ndim != 1 or np.any(selected < 0) or np.any(selected >= feature_count):
        raise ValueError("selected feature indices are outside the frontend output")
    component = selected // (band_count * energy_bins)
    band = (selected // energy_bins) % band_count
    groups: list[FeatureGroup] = []
    for component_index, band_index in sorted(set(zip(component, band))):
        columns = np.flatnonzero(
            (component == component_index) & (band == band_index)
        )
        groups.append(
            FeatureGroup(
                component=int(component_index),
                band=int(band_index),
                band_name=band_names[int(band_index)],
                selected_columns=columns,
                source_indices=selected[columns],
            )
        )
    return groups


def phase_masks(
    target: np.ndarray,
    prediction: np.ndarray,
    threshold: float = 0.1,
    transition_bins: int = 5,
) -> dict[str, np.ndarray]:
    """Separate onset, sustained movement, release, and rest false positives."""
    target = np.asarray(target)
    prediction = np.asarray(prediction)
    if target.shape != prediction.shape or target.ndim != 1:
        raise ValueError("target and prediction must be matching one-dimensional arrays")
    if transition_bins < 1:
        raise ValueError("transition_bins must be positive")
    moving = target >= threshold
    onset_event = moving & ~np.r_[False, moving[:-1]]
    release_event = ~moving & np.r_[False, moving[:-1]]
    onset = np.zeros_like(moving)
    release = np.zeros_like(moving)
    for shift in range(transition_bins):
        onset[shift:] |= onset_event[: target.size - shift]
        release[shift:] |= release_event[: target.size - shift]
    onset &= moving
    release &= ~moving
    return {
        "onset": onset,
        "sustained_movement": moving & ~onset,
        "release": release,
        "rest_false_positive": (~moving) & (prediction >= threshold),
    }


def matched_rest_donors(
    standardized: np.ndarray,
    all_finger_target: np.ndarray,
    finger: int,
    query_indices: np.ndarray,
    threshold: float = 0.1,
    maximum_pool: int = 1024,
    matching_dimensions: int = 64,
) -> np.ndarray:
    """Find in-distribution rest frames matched on other-finger state and energy."""
    standardized = np.asarray(standardized, dtype=np.float32)
    targets = np.asarray(all_finger_target)
    queries = np.asarray(query_indices, dtype=np.int64)
    if standardized.ndim != 2 or targets.ndim != 2:
        raise ValueError("features and targets must be two-dimensional")
    if standardized.shape[0] != targets.shape[0]:
        raise ValueError("features and targets must have the same number of rows")
    if not 0 <= finger < targets.shape[1]:
        raise ValueError("finger index is outside the target")
    if np.any(queries < 0) or np.any(queries >= standardized.shape[0]):
        raise ValueError("query index is outside the sequence")
    other = np.delete(targets >= threshold, finger, axis=1)
    powers = (1 << np.arange(other.shape[1], dtype=np.int64))[None]
    state_code = np.sum(other.astype(np.int64) * powers, axis=1)
    rest = targets[:, finger] < threshold
    rest_indices = np.flatnonzero(rest)
    if rest_indices.size < 2:
        raise ValueError("at least two rest frames are required for conditional donors")
    dimensions = np.linspace(
        0,
        standardized.shape[1] - 1,
        min(matching_dimensions, standardized.shape[1]),
        dtype=np.int64,
    )
    result = np.empty(queries.size, dtype=np.int64)
    for position, query in enumerate(queries):
        pool = rest_indices[state_code[rest_indices] == state_code[query]]
        if pool.size < 2:
            pool = rest_indices
        if pool.size > maximum_pool:
            pool = pool[np.linspace(0, pool.size - 1, maximum_pool, dtype=np.int64)]
        pool = pool[pool != query]
        difference = standardized[pool][:, dimensions] - standardized[query, dimensions]
        result[position] = pool[int(np.argmin(np.sum(difference * difference, axis=1)))]
    return result


def manual_lstm_trace(
    lstm: nn.LSTM,
    temporal: nn.Linear,
    direct: nn.Linear,
    standardized: torch.Tensor,
    output_activation: str = "relu",
    softplus_beta: float = 10.0,
) -> dict[str, torch.Tensor]:
    """Reconstruct a one-layer, forward LSTM and retain its internal states."""
    if lstm.num_layers != 1 or lstm.bidirectional or lstm.proj_size:
        raise ValueError("probe supports a one-layer, forward, unprojected LSTM")
    if standardized.ndim != 3 or standardized.shape[-1] != lstm.input_size:
        raise ValueError("standardized input has the wrong shape")
    batch, steps, _ = standardized.shape
    hidden = lstm.hidden_size
    h = standardized.new_zeros(batch, hidden)
    c = standardized.new_zeros(batch, hidden)
    traces: dict[str, list[torch.Tensor]] = {
        "input_logit": [],
        "forget_logit": [],
        "candidate_logit": [],
        "output_logit": [],
        "input_gate": [],
        "forget_gate": [],
        "candidate": [],
        "output_gate": [],
        "cell_write": [],
        "retention": [],
        "cell_state": [],
        "hidden_state": [],
    }
    bias = lstm.bias_ih_l0 + lstm.bias_hh_l0
    for step in range(steps):
        logits = (
            F.linear(standardized[:, step], lstm.weight_ih_l0, bias)
            + F.linear(h, lstm.weight_hh_l0)
        )
        input_logit, forget_logit, candidate_logit, output_logit = logits.chunk(4, 1)
        input_gate = torch.sigmoid(input_logit)
        forget_gate = torch.sigmoid(forget_logit)
        candidate = torch.tanh(candidate_logit)
        output_gate = torch.sigmoid(output_logit)
        retention = forget_gate * c
        cell_write = input_gate * candidate
        c = retention + cell_write
        h = output_gate * torch.tanh(c)
        values = (
            input_logit,
            forget_logit,
            candidate_logit,
            output_logit,
            input_gate,
            forget_gate,
            candidate,
            output_gate,
            cell_write,
            retention,
            c,
            h,
        )
        for name, value in zip(traces, values, strict=True):
            traces[name].append(value)
    output = {name: torch.stack(value, dim=1) for name, value in traces.items()}
    output["direct_term"] = direct(standardized).squeeze(-1)
    output["recurrent_term"] = temporal(output["hidden_state"]).squeeze(-1)
    output["pre_relu"] = output["direct_term"] + output["recurrent_term"]
    if output_activation == "relu":
        output["prediction"] = torch.relu(output["pre_relu"])
    elif output_activation == "softplus":
        output["prediction"] = F.softplus(output["pre_relu"], beta=softplus_beta)
    elif output_activation == "linear":
        output["prediction"] = output["pre_relu"]
    else:
        raise ValueError(f"unsupported output activation {output_activation!r}")
    return output


def _linear_same_filter(x: torch.Tensor, layer: nn.Conv1d) -> torch.Tensor:
    dilation = int(layer.dilation[0])
    padding = dilation * (layer.kernel_size[0] - 1)
    left = padding // 2
    right = padding - left
    return F.conv1d(
        F.pad(x, (left, right), mode="constant"),
        layer.weight,
        bias=None,
        dilation=dilation,
    )


@torch.inference_mode()
def linearized_wavelet_impulse_response(
    frontend: WaveletPacketEnergy | AsymmetricWaveletPacketEnergy,
    length: int = 4096,
) -> np.ndarray:
    """Return the small-signal, pre-energy impulse response of every path."""
    if length < frontend.effective_kernel_size + 2:
        raise ValueError("impulse length is shorter than the effective filter")
    parameter = next(frontend.parameters())
    impulse = torch.zeros(1, 1, length, dtype=parameter.dtype, device=parameter.device)
    impulse[..., length // 2] = 1.0
    slope = 1.7156 * (2.0 / 3.0)
    if isinstance(frontend, WaveletPacketEnergy):
        bands = impulse
        for layer in frontend.layers:
            bands = slope * _linear_same_filter(bands, layer)
        return bands[0].float().cpu().numpy()
    bands = impulse
    for layer in frontend.base.layers:
        bands = slope * _linear_same_filter(bands, layer)
    retained = bands[:, frontend.retained_parents]
    split = slope * _linear_same_filter(bands, frontend.split_layer)
    half = frontend.lmp_kernel_size // 2
    lmp = F.conv1d(F.pad(impulse, (half, half)), frontend.lmp.weight, bias=None)
    return torch.cat((retained, split, lmp), dim=1)[0].float().cpu().numpy()


def frequency_response_summary(
    impulse_response: np.ndarray,
    sampling_rate_hz: float = 1000.0,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float]]]:
    """Measure magnitude, phase-derived delay, and out-of-band leakage."""
    impulse_response = np.asarray(impulse_response, dtype=np.float64)
    # Move the injected impulse to sample zero so reported delay is the
    # filter's computational delay rather than the arbitrary probe offset.
    centered = np.roll(impulse_response, -(impulse_response.shape[1] // 2), axis=1)
    spectrum = np.fft.rfft(centered, axis=1)
    frequencies = np.fft.rfftfreq(impulse_response.shape[1], 1.0 / sampling_rate_hz)
    magnitude = np.abs(spectrum)
    power = magnitude**2
    phase = np.unwrap(np.angle(spectrum), axis=1)
    omega = 2.0 * np.pi * frequencies
    group_delay_seconds = -np.gradient(phase, omega, axis=1)
    summaries: list[dict[str, float]] = []
    for band in range(impulse_response.shape[0]):
        weights = power[band]
        total = float(weights.sum())
        if total <= 0:
            summaries.append({"center_hz": 0.0, "low_hz": 0.0, "high_hz": 0.0, "group_delay_ms": 0.0, "stopband_leakage": 0.0})
            continue
        active_mask = magnitude[band] >= magnitude[band].max() / np.sqrt(2.0)
        active_indices = np.flatnonzero(active_mask)
        low_index = int(active_indices[0])
        high_index = int(active_indices[-1])
        active_weight = weights[active_mask]
        center = float(np.sum(frequencies * weights) / total)
        delay = float(
            np.sum(group_delay_seconds[band, active_mask] * active_weight)
            / max(active_weight.sum(), np.finfo(float).eps)
        )
        summaries.append(
            {
                "center_hz": center,
                "low_hz": float(frequencies[low_index]),
                "high_hz": float(frequencies[high_index]),
                "group_delay_ms": 1000.0 * delay,
                "stopband_leakage": float(1.0 - active_weight.sum() / total),
            }
        )
    return frequencies, magnitude, summaries
