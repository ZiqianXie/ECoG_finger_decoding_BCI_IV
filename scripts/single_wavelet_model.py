#!/usr/bin/env python3
"""Trainable components for the released single-wavelet decoder."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ecog_decoding.models import WaveletPacketEnergy
from single_wavelet_support import FREQUENCY_ORDER, HISTORY, SAMPLES_PER_BIN


class SingleWaveletDecoder(nn.Module):
    """One spatial convolution, one wavelet tree, and one nonlinear LSTM.

    The sparse LARS predictor is embedded in one LSTM unit so that the network
    starts near a useful linear solution. It is retained separately only as an
    immutable audit baseline; the trainable prediction passes through the LSTM.
    """

    def __init__(
        self,
        spatial_weights: np.ndarray,
        initialization: dict[str, np.ndarray],
        hidden_size: int,
        lars_candidate_scale: float = 1.0,
        near_zero_std: float = 1.0e-3,
        open_gate_bias: float = 5.0,
        forget_gate_bias: float = -5.0,
        output_activation: str = "softplus",
        softplus_beta: float = 10.0,
        energy_window_samples: int = SAMPLES_PER_BIN,
        tap_resample_up: int = 5,
        tap_resample_down: int = 2,
    ) -> None:
        super().__init__()
        components, channels = spatial_weights.shape
        self.spatial = nn.Conv1d(channels, components, 1, bias=False)
        with torch.no_grad():
            self.spatial.weight.copy_(torch.from_numpy(spatial_weights)[:, :, None])

        self.wavelet = WaveletPacketEnergy(
            wavelet="bior6.8",
            levels=3,
            kernel_size=17,
            trainable=True,
            padding_mode="constant",
            energy_window_samples=energy_window_samples,
            energy_stride_samples=energy_window_samples,
            tap_resample_up=tap_resample_up,
            tap_resample_down=tap_resample_down,
        )
        self.samples_per_bin = int(energy_window_samples)
        self.register_buffer(
            "selected_indices",
            torch.as_tensor(initialization["selected_indices"], dtype=torch.long),
        )
        self.register_buffer(
            "frequency_order", torch.as_tensor(FREQUENCY_ORDER, dtype=torch.long)
        )
        self.register_buffer(
            "feature_mean",
            torch.as_tensor(initialization["feature_mean"], dtype=torch.float32),
        )
        self.register_buffer(
            "feature_scale",
            torch.as_tensor(initialization["feature_scale"], dtype=torch.float32),
        )

        if output_activation not in ("linear", "softplus"):
            raise ValueError(f"unsupported output activation {output_activation!r}")
        if softplus_beta <= 0:
            raise ValueError("softplus beta must be positive")
        self.output_activation = output_activation
        self.softplus_beta = float(softplus_beta)
        self.head_initialization = "lars_linear_regime"

        feature_count = int(self.selected_indices.numel())
        self.direct = nn.Linear(feature_count, 1)
        self.lstm = nn.LSTM(feature_count, hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, 1)
        with torch.no_grad():
            self.direct.weight.copy_(
                torch.as_tensor(initialization["coefficients"])[None]
            )
            self.direct.bias.fill_(float(initialization["intercept"]))
        self.direct.requires_grad_(False)
        self._initialize_lars_linear_regime(
            coefficients=np.asarray(initialization["coefficients"]),
            intercept=float(initialization["intercept"]),
            candidate_scale=lars_candidate_scale,
            near_zero_std=near_zero_std,
            open_gate_bias=open_gate_bias,
            forget_gate_bias=forget_gate_bias,
        )

    @torch.no_grad()
    def _initialize_lars_linear_regime(
        self,
        coefficients: np.ndarray,
        intercept: float,
        candidate_scale: float,
        near_zero_std: float,
        open_gate_bias: float,
        forget_gate_bias: float,
        unit: int = 0,
    ) -> None:
        """Embed LARS in one standard LSTM unit without disabling nonlinearity."""
        if not 0 <= unit < self.lstm.hidden_size:
            raise ValueError("LARS unit is outside the LSTM hidden state")
        if candidate_scale <= 0:
            raise ValueError("lars_candidate_scale must be positive")
        if near_zero_std < 0:
            raise ValueError("near_zero_std must be nonnegative")

        coefficient = torch.as_tensor(
            coefficients,
            dtype=self.lstm.weight_ih_l0.dtype,
            device=self.lstm.weight_ih_l0.device,
        )
        if coefficient.shape != (self.lstm.input_size,):
            raise ValueError("LARS coefficient count must equal the LSTM input size")

        for parameter in (
            self.lstm.weight_ih_l0,
            self.lstm.weight_hh_l0,
            self.output.weight,
        ):
            parameter.normal_(0.0, near_zero_std)
        self.lstm.bias_ih_l0.zero_()
        self.lstm.bias_hh_l0.zero_()
        self.output.bias.zero_()

        hidden = self.lstm.hidden_size
        input_row = unit
        forget_row = hidden + unit
        candidate_row = 2 * hidden + unit
        output_row = 3 * hidden + unit
        self.lstm.bias_ih_l0[input_row] = open_gate_bias
        self.lstm.bias_ih_l0[forget_row] = forget_gate_bias
        self.lstm.bias_ih_l0[output_row] = open_gate_bias
        self.lstm.bias_ih_l0[candidate_row] = candidate_scale * intercept
        self.lstm.weight_ih_l0[candidate_row].copy_(candidate_scale * coefficient)

        open_gate = torch.sigmoid(
            torch.as_tensor(
                open_gate_bias,
                dtype=self.output.weight.dtype,
                device=self.output.weight.device,
            )
        )
        self.output.weight[0, unit] = 1.0 / (
            candidate_scale * open_gate.square()
        )

    @property
    def context_samples(self) -> int:
        return self.wavelet.effective_kernel_size // 2

    def activate_output(self, prediction: torch.Tensor) -> torch.Tensor:
        if self.output_activation == "softplus":
            return F.softplus(prediction, beta=self.softplus_beta)
        return prediction

    def decode_features(self, features: torch.Tensor) -> torch.Tensor:
        standardized = (features - self.feature_mean) / self.feature_scale
        recurrent, _ = self.lstm(standardized)
        return self.activate_output(self.output(recurrent)).squeeze(-1)

    def direct_features(self, features: torch.Tensor) -> torch.Tensor:
        """Return the immutable LARS audit prediction for the same features."""
        standardized = (features - self.feature_mean) / self.feature_scale
        return self.activate_output(self.direct(standardized)).squeeze(-1)

    def extract_sequences(
        self,
        padded_ecog: torch.Tensor,
        starts: torch.Tensor,
        steps: int,
    ) -> torch.Tensor:
        """Extract continuous-filter histories for several binned sequences."""
        context = self.context_samples
        filtered_bins = steps + HISTORY - 1
        samples = filtered_bins * self.samples_per_bin
        length = samples + 2 * context
        offsets = torch.arange(length, device=starts.device)
        sample_indices = starts[:, None] * self.samples_per_bin + offsets[None]
        segments = padded_ecog.index_select(0, sample_indices.reshape(-1)).reshape(
            starts.numel(), length, padded_ecog.shape[1]
        )

        spatial = self.spatial(segments.transpose(1, 2))
        batch, components, _ = spatial.shape
        bands = spatial.reshape(batch * components, 1, length)
        for layer in self.wavelet.layers:
            bands = self.wavelet._same_filter(bands, layer)
            bands = 1.7156 * torch.tanh((2.0 / 3.0) * bands)
        bands = bands[..., context : context + samples]

        energy = self.wavelet._energy(bands)
        energy = energy.reshape(batch, components, 8, filtered_bins)
        per_bin = (
            energy.index_select(2, self.frequency_order)
            .permute(0, 3, 1, 2)
            .flatten(2)
        )
        history = per_bin.unfold(1, HISTORY, 1).permute(0, 1, 3, 2).flatten(2)
        return history.index_select(2, self.selected_indices)

    def forward(
        self,
        padded_ecog: torch.Tensor,
        starts: torch.Tensor,
        steps: int,
    ) -> torch.Tensor:
        return self.decode_features(self.extract_sequences(padded_ecog, starts, steps))


@torch.inference_mode()
def predict_intervals(
    model: SingleWaveletDecoder,
    features: torch.Tensor,
    intervals: list[list[int]],
    rows: int,
) -> np.ndarray:
    model.eval()
    prediction = np.full(rows, np.nan, dtype=np.float32)
    for begin, end in intervals:
        prediction[begin:end] = (
            model.decode_features(features[begin:end][None])[0].float().cpu().numpy()
        )
    return prediction


@torch.inference_mode()
def extract_all(
    model: SingleWaveletDecoder,
    padded_ecog: torch.Tensor,
    rows: int,
    chunk_steps: int,
) -> torch.Tensor:
    model.eval()
    chunks = []
    for begin in range(0, rows, chunk_steps):
        steps = min(chunk_steps, rows - begin)
        chunks.append(
            model.extract_sequences(
                padded_ecog,
                torch.as_tensor([begin], device=padded_ecog.device),
                steps,
            )[0]
        )
    return torch.cat(chunks, dim=0)
