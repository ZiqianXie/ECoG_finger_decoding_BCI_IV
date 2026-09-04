"""Neural-network components for ECoG decoding."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pywt
import torch
from sklearn.decomposition import FastICA
from torch import nn
from torch.nn import functional as F


_WAVELET_BRANCHES = {
    "decomposition_lowpass": "dec_lo",
    "decomposition_highpass": "dec_hi",
    "reconstruction_lowpass": "rec_lo",
    "reconstruction_highpass": "rec_hi",
}


def fit_fastica_spatial_weights(
    ecog: np.ndarray,
    n_components: int | None = None,
    max_samples: int | None = 50_000,
    random_state: int = 0,
    backend: str = "sklearn",
    device: str | torch.device | None = None,
) -> np.ndarray:
    """Fit a FastICA unmixing matrix using training data only.

    Long recordings are evenly subsampled for initialization efficiency. The
    returned rows can be copied directly into a bias-free 1x1 convolution.
    """
    ecog = np.asarray(ecog, dtype=np.float64)
    if ecog.ndim != 2 or ecog.shape[0] < 2:
        raise ValueError(f"ecog must have shape (time, channels); got {ecog.shape}")
    if not np.isfinite(ecog).all():
        raise ValueError("ecog contains non-finite values")
    component_count = ecog.shape[1] if n_components is None else int(n_components)
    if not 1 <= component_count <= ecog.shape[1]:
        raise ValueError("n_components must be between 1 and the channel count")
    if max_samples is not None:
        if max_samples < 2:
            raise ValueError("max_samples must be at least 2 or None")
        if ecog.shape[0] > max_samples:
            indices = np.linspace(0, ecog.shape[0] - 1, max_samples, dtype=int)
            ecog = ecog[indices]
    if backend == "sklearn":
        estimator = FastICA(
            n_components=component_count,
            algorithm="parallel",
            whiten="unit-variance",
            fun="logcosh",
            max_iter=1000,
            tol=1.0e-4,
            random_state=random_state,
        )
        estimator.fit(ecog)
        return np.asarray(estimator.components_, dtype=np.float32)
    if backend != "torch":
        raise ValueError("backend must be 'sklearn' or 'torch'")

    target_device = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    values = torch.as_tensor(ecog, dtype=torch.float32, device=target_device)
    values = values - values.mean(dim=0, keepdim=True)
    covariance = values.T @ values / values.shape[0]
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    order = torch.argsort(eigenvalues, descending=True)[:component_count]
    eigenvalues = eigenvalues[order].clamp_min(torch.finfo(values.dtype).eps)
    eigenvectors = eigenvectors[:, order]
    whitening = (eigenvectors * eigenvalues.rsqrt()).T
    whitened = values @ whitening.T

    generator = torch.Generator(device=target_device)
    generator.manual_seed(random_state)
    weights = torch.randn(
        component_count,
        component_count,
        generator=generator,
        dtype=values.dtype,
        device=target_device,
    )

    def symmetric_decorrelation(matrix: torch.Tensor) -> torch.Tensor:
        spectrum, basis = torch.linalg.eigh(matrix @ matrix.T)
        inverse_root = spectrum.clamp_min(1.0e-7).rsqrt()
        return (basis * inverse_root) @ basis.T @ matrix

    weights = symmetric_decorrelation(weights)
    for _ in range(1000):
        old_weights = weights
        projected = whitened @ weights.T
        nonlinear = torch.tanh(projected)
        derivative_mean = (1.0 - nonlinear.square()).mean(dim=0)
        weights = nonlinear.T @ whitened / whitened.shape[0]
        weights = weights - derivative_mean[:, None] * old_weights
        weights = symmetric_decorrelation(weights)
        alignment = torch.diagonal(weights @ old_weights.T).abs()
        if (1.0 - alignment).abs().max().item() < 1.0e-4:
            break

    unmixing = weights @ whitening
    source_scale = (values @ unmixing.T).std(dim=0, correction=0).clamp_min(1.0e-7)
    unmixing = unmixing / source_scale[:, None]
    return unmixing.detach().cpu().numpy().astype(np.float32, copy=False)


def compact_wavelet_taps(
    wavelet: str = "bior2.2",
    branch: str = "decomposition_highpass",
    trim_boundary_zeros: bool = True,
    zero_tolerance: float = 1.0e-12,
) -> np.ndarray:
    """Return convolution-ordered wavelet taps without inert edge padding.

    PyWavelets stores filter-bank coefficients in convolution order. PyTorch's
    ``conv1d`` performs cross-correlation, so the taps are reversed here. Only
    leading and trailing zeros are removed; an interior zero would encode a
    real lag and is therefore preserved.
    """
    if branch not in _WAVELET_BRANCHES:
        choices = ", ".join(sorted(_WAVELET_BRANCHES))
        raise ValueError(f"unknown wavelet branch {branch!r}; choose from {choices}")
    if zero_tolerance < 0:
        raise ValueError("zero_tolerance must be nonnegative")

    coefficients = np.asarray(
        getattr(pywt.Wavelet(wavelet), _WAVELET_BRANCHES[branch]),
        dtype=np.float64,
    )
    if trim_boundary_zeros:
        nonzero = np.flatnonzero(np.abs(coefficients) > zero_tolerance)
        if nonzero.size == 0:
            raise ValueError("wavelet branch contains no nonzero coefficients")
        coefficients = coefficients[nonzero[0] : nonzero[-1] + 1]
    return coefficients[::-1].copy()


def fixed_length_wavelet_taps(
    wavelet: str,
    branch: str,
    kernel_size: int,
    zero_tolerance: float = 1.0e-12,
) -> np.ndarray:
    """Remove only inert boundary padding to obtain a requested kernel size.

    When more than one boundary-zero removal is possible, prefer the window
    that best preserves the wavelet filter's symmetry. This selects taps 1:18
    for the bior6.8 low-pass filter and taps 0:17 for its high-pass filter.
    Interior zero coefficients remain in the kernel and can become nonzero
    during training, matching the cross-band adaptation described in the paper.
    """
    if kernel_size < 1:
        raise ValueError("kernel_size must be positive")
    if branch not in _WAVELET_BRANCHES:
        choices = ", ".join(sorted(_WAVELET_BRANCHES))
        raise ValueError(f"unknown wavelet branch {branch!r}; choose from {choices}")
    coefficients = np.asarray(
        getattr(pywt.Wavelet(wavelet), _WAVELET_BRANCHES[branch]),
        dtype=np.float64,
    )
    if coefficients.size < kernel_size:
        raise ValueError(
            f"{wavelet} {branch} has {coefficients.size} taps, fewer than {kernel_size}"
        )

    candidates: list[tuple[float, int, np.ndarray]] = []
    removed_count = coefficients.size - kernel_size
    for start in range(removed_count + 1):
        stop = start + kernel_size
        removed = np.concatenate((coefficients[:start], coefficients[stop:]))
        if removed.size and np.any(np.abs(removed) > zero_tolerance):
            continue
        candidate = coefficients[start:stop]
        symmetry_error = float(np.max(np.abs(candidate - candidate[::-1])))
        candidates.append((symmetry_error, start, candidate))
    if not candidates:
        raise ValueError(
            f"cannot reduce {wavelet} {branch} to {kernel_size} taps by removing boundary zeros"
        )
    _, _, selected = min(candidates, key=lambda item: (item[0], item[1]))
    return selected[::-1].copy()


class DilatedWaveletFilterBank(nn.Module):
    """A learnable multiscale FIR bank initialized from a wavelet filter.

    The compact tap vector is copied once per dilation. Filters are shared over
    electrodes but can adapt independently across scales. Dilation supplies the
    gaps between taps without allocating trainable zero coefficients.

    Input shape is ``(batch, electrode, time)`` and output shape is
    ``(batch, electrode, scale, time)``.
    """

    def __init__(
        self,
        wavelet: str = "bior2.2",
        branch: str = "decomposition_highpass",
        dilations: Sequence[int] = (1, 2, 4, 8, 16, 32, 64),
        trim_boundary_zeros: bool = True,
        trainable: bool = True,
        padding_mode: str = "reflect",
    ) -> None:
        super().__init__()
        parsed_dilations = tuple(int(value) for value in dilations)
        if not parsed_dilations or any(value < 1 for value in parsed_dilations):
            raise ValueError("dilations must contain positive integers")
        if len(set(parsed_dilations)) != len(parsed_dilations):
            raise ValueError("dilations must be unique")
        if padding_mode not in {"constant", "reflect", "replicate", "circular"}:
            raise ValueError(f"unsupported padding mode {padding_mode!r}")

        taps = compact_wavelet_taps(
            wavelet=wavelet,
            branch=branch,
            trim_boundary_zeros=trim_boundary_zeros,
        )
        initial = torch.tensor(taps, dtype=torch.float32)[None, None, :]
        initial = initial.repeat(len(parsed_dilations), 1, 1)
        self.kernel_taps = nn.Parameter(initial, requires_grad=trainable)
        self.wavelet = wavelet
        self.branch = branch
        self.dilations = parsed_dilations
        self.padding_mode = padding_mode

    @property
    def effective_kernel_sizes(self) -> tuple[int, ...]:
        tap_count = int(self.kernel_taps.shape[-1])
        return tuple((tap_count - 1) * dilation + 1 for dilation in self.dilations)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                f"x must have shape (batch, electrode, time); got {tuple(x.shape)}"
            )
        batch, electrodes, time = x.shape
        flattened = x.reshape(batch * electrodes, 1, time)
        filtered: list[torch.Tensor] = []
        for index, dilation in enumerate(self.dilations):
            padding = dilation * (self.kernel_taps.shape[-1] - 1)
            left = int(padding // 2)
            right = int(padding - left)
            mode = self.padding_mode
            if mode == "reflect" and (left >= time or right >= time):
                mode = "replicate"
            padded = F.pad(flattened, (left, right), mode=mode)
            scale = F.conv1d(
                padded,
                self.kernel_taps[index : index + 1],
                dilation=dilation,
            )
            filtered.append(scale.reshape(batch, electrodes, time))
        return torch.stack(filtered, dim=2)


class WaveletPacketEnergy(nn.Module):
    """Paper-defined undecimated wavelet-packet CNN and log-energy pooling.

    Three levels produce 2, 4, and 8 paths. At each level every prior band is
    split into low/high children using bior6.8 analysis filters, with dilation
    1, 2, and 4 and same-length padding. Connections outside the wavelet tree
    start at zero but remain trainable, allowing cross-band interactions.

    Input shape is ``(batch, electrode, time)`` and output shape is
    ``(batch, electrode, band, energy_time)``.
    """

    def __init__(
        self,
        wavelet: str = "bior6.8",
        levels: int = 3,
        kernel_size: int = 17,
        trainable: bool = True,
        padding_mode: str = "reflect",
        energy_window_samples: int = 40,
        energy_stride_samples: int = 40,
        log_epsilon: float = 0.0,
    ) -> None:
        super().__init__()
        if levels < 1:
            raise ValueError("levels must be positive")
        if energy_window_samples < 1 or energy_stride_samples < 1:
            raise ValueError("energy window and stride must be positive")
        if log_epsilon < 0:
            raise ValueError("log_epsilon must be nonnegative")
        if padding_mode not in {"constant", "reflect", "replicate", "circular"}:
            raise ValueError(f"unsupported padding mode {padding_mode!r}")

        lowpass = fixed_length_wavelet_taps(
            wavelet,
            "decomposition_lowpass",
            kernel_size,
        )
        highpass = fixed_length_wavelet_taps(
            wavelet,
            "decomposition_highpass",
            kernel_size,
        )
        low_initial = torch.tensor(lowpass, dtype=torch.float32)
        high_initial = torch.tensor(highpass, dtype=torch.float32)
        self.layers = nn.ModuleList()
        for level in range(levels):
            parent_count = 2**level
            layer = nn.Conv1d(
                parent_count,
                2 * parent_count,
                kernel_size=kernel_size,
                dilation=2**level,
                padding=0,
                bias=True,
            )
            with torch.no_grad():
                layer.weight.zero_()
                layer.bias.zero_()
                for parent in range(parent_count):
                    layer.weight[2 * parent, parent] = low_initial
                    layer.weight[2 * parent + 1, parent] = high_initial
            layer.weight.requires_grad_(trainable)
            layer.bias.requires_grad_(trainable)
            self.layers.append(layer)
        self.wavelet = wavelet
        self.levels = int(levels)
        self.kernel_size = int(kernel_size)
        self.dilations = tuple(2**level for level in range(levels))
        self.padding_mode = padding_mode
        self.energy_window_samples = int(energy_window_samples)
        self.energy_stride_samples = int(energy_stride_samples)
        self.log_epsilon = float(log_epsilon)

    @property
    def band_names(self) -> tuple[str, ...]:
        return tuple(
            "".join("L" if (band >> bit) & 1 == 0 else "H" for bit in reversed(range(self.levels)))
            for band in range(2**self.levels)
        )

    def _same_filter(
        self, x: torch.Tensor, layer: nn.Conv1d
    ) -> torch.Tensor:
        dilation = int(layer.dilation[0])
        padding = dilation * (layer.kernel_size[0] - 1)
        left = int(padding // 2)
        right = int(padding - left)
        mode = self.padding_mode
        if mode == "reflect" and (left >= x.shape[-1] or right >= x.shape[-1]):
            mode = "replicate"
        return layer(F.pad(x, (left, right), mode=mode))

    def _energy(self, x: torch.Tensor) -> torch.Tensor:
        squared_sum = F.avg_pool1d(
            x.square(),
            kernel_size=self.energy_window_samples,
            stride=self.energy_stride_samples,
        ) * self.energy_window_samples
        l2_norm = torch.sqrt(torch.clamp_min(squared_sum, 0.0))
        return torch.log1p(l2_norm + self.log_epsilon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                f"x must have shape (batch, electrode, time); got {tuple(x.shape)}"
            )
        batch, electrodes, time = x.shape
        if time < self.energy_window_samples:
            raise ValueError("time dimension is shorter than the energy window")
        bands = x.reshape(batch * electrodes, 1, time)
        for layer in self.layers:
            bands = self._same_filter(bands, layer)
            bands = 1.7156 * torch.tanh((2.0 / 3.0) * bands)
        energy = self._energy(bands)
        return energy.reshape(batch, electrodes, 2**self.levels, energy.shape[-1])


class DiagonalSSMBlock(nn.Module):
    """Stable diagonal state-space block with parallel causal convolution."""

    def __init__(
        self,
        width: int,
        state_size: int = 16,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if width < 1 or state_size < 1:
            raise ValueError("width and state_size must be positive")
        self.width = int(width)
        self.state_size = int(state_size)
        self.norm = nn.LayerNorm(width)
        self.input_projection = nn.Linear(width, 2 * width)
        decay_rates = torch.logspace(-2.0, 0.0, state_size)
        inverse_softplus = torch.log(torch.expm1(decay_rates))
        self.log_decay = nn.Parameter(inverse_softplus.repeat(width, 1))
        scale = state_size**-0.5
        self.input_scale = nn.Parameter(scale * torch.randn(width, state_size))
        self.output_scale = nn.Parameter(scale * torch.randn(width, state_size))
        self.skip = nn.Parameter(torch.ones(width))
        self.output_projection = nn.Linear(width, width)
        self.dropout = nn.Dropout(dropout)

    def _kernel(
        self, length: int, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        decay = torch.exp(-F.softplus(self.log_decay.float()))
        steps = torch.arange(length, device=device, dtype=torch.float32)
        powers = torch.pow(decay[..., None], steps)
        kernel = torch.sum(
            self.input_scale.float()[..., None]
            * self.output_scale.float()[..., None]
            * powers,
            dim=1,
        )
        return kernel.to(dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != self.width:
            raise ValueError(
                f"x must have shape (batch, time, {self.width}); got {tuple(x.shape)}"
            )
        residual = x
        value, gate = self.input_projection(self.norm(x)).chunk(2, dim=-1)
        value = F.silu(value)
        length = value.shape[1]
        kernel = self._kernel(length, value.dtype, value.device)
        fft_size = 1 << max(1, (2 * length - 1).bit_length())
        value_frequency = torch.fft.rfft(value.transpose(1, 2), n=fft_size)
        kernel_frequency = torch.fft.rfft(kernel, n=fft_size)
        filtered = torch.fft.irfft(
            value_frequency * kernel_frequency[None, :, :], n=fft_size
        )[..., :length]
        filtered = filtered.transpose(1, 2) + value * self.skip
        update = self.output_projection(filtered * F.silu(gate))
        return residual + self.dropout(update)


class DiagonalSSM(nn.Module):
    """Stack of FFT-trained diagonal SSM blocks."""

    def __init__(
        self,
        width: int,
        layers: int = 3,
        state_size: int = 16,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError("layers must be positive")
        self.blocks = nn.ModuleList(
            DiagonalSSMBlock(width, state_size=state_size, dropout=dropout)
            for _ in range(layers)
        )
        self.final_norm = nn.LayerNorm(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.final_norm(x)


class CausalLinearAttentionBlock(nn.Module):
    """Parallel causal linear attention with gating and a nonlinear MLP.

    This is an input-dependent modern sequence block, unlike the fixed-kernel
    diagonal SSM above.  ELU+1 feature maps turn causal attention into prefix
    sums, retaining linear complexity and full-sequence parallel training.
    """

    def __init__(
        self,
        width: int,
        heads: int = 4,
        expansion: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if width < 1 or heads < 1 or width % heads:
            raise ValueError("width must be positive and divisible by heads")
        self.width = int(width)
        self.heads = int(heads)
        self.head_width = width // heads
        self.norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, 3 * width)
        self.output_gate = nn.Linear(width, width)
        self.output_projection = nn.Linear(width, width)
        self.attention_dropout = nn.Dropout(dropout)
        self.mlp_norm = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, expansion * width),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(expansion * width, width),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != self.width:
            raise ValueError(
                f"x must have shape (batch, time, {self.width}); got {tuple(x.shape)}"
            )
        normalized = self.norm(x)
        query, key, value = self.qkv(normalized).chunk(3, dim=-1)
        shape = (*query.shape[:2], self.heads, self.head_width)
        query = F.elu(query.reshape(shape)) + 1.0
        key = F.elu(key.reshape(shape)) + 1.0
        value = value.reshape(shape)
        key_value = torch.einsum("bthd,bthe->bthde", key, value)
        key_value_prefix = torch.cumsum(key_value, dim=1)
        key_prefix = torch.cumsum(key, dim=1)
        numerator = torch.einsum("bthd,bthde->bthe", query, key_value_prefix)
        denominator = torch.einsum("bthd,bthd->bth", query, key_prefix).unsqueeze(-1)
        attended = (numerator / denominator.clamp_min(1.0e-6)).reshape_as(x)
        gate = torch.sigmoid(self.output_gate(normalized))
        x = x + self.attention_dropout(self.output_projection(attended * gate))
        return x + self.mlp(self.mlp_norm(x))


class CausalLinearAttention(nn.Module):
    """Stack of gated causal linear-attention blocks."""

    def __init__(
        self,
        width: int,
        layers: int = 3,
        heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError("layers must be positive")
        self.blocks = nn.ModuleList(
            CausalLinearAttentionBlock(
                width,
                heads=heads,
                dropout=dropout,
            )
            for _ in range(layers)
        )
        self.final_norm = nn.LayerNorm(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.final_norm(x)


class CausalDilatedTCNBlock(nn.Module):
    """Gated causal temporal convolution that preserves short movement pulses."""

    def __init__(self, width: int, dilation: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.dilation = int(dilation)
        self.norm = nn.LayerNorm(width)
        self.conv = nn.Conv1d(width, 2 * width, kernel_size=3, dilation=dilation)
        self.output = nn.Conv1d(width, width, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        value = self.norm(x).transpose(1, 2)
        value = F.pad(value, (2 * self.dilation, 0))
        value = F.glu(self.conv(value), dim=1)
        value = self.output(value).transpose(1, 2)
        return residual + self.dropout(value)


class CausalDilatedTCN(nn.Module):
    """Parallel gated TCN with exponentially increasing causal dilation."""

    def __init__(self, width: int, layers: int = 5, dropout: float = 0.0) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            CausalDilatedTCNBlock(width, dilation=2**level, dropout=dropout)
            for level in range(layers)
        )
        self.final_norm = nn.LayerNorm(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.final_norm(x)


class MambaSequence(nn.Module):
    """Stack of native selective state-space (Mamba) mixer blocks.

    The import is intentionally delayed so the core package remains usable on
    machines without the optional CUDA extension.  Unlike ``DiagonalSSM``,
    Mamba's recurrence parameters depend on the input at every time point.
    """

    def __init__(
        self,
        width: int,
        layers: int = 2,
        state_size: int = 16,
        convolution_width: int = 4,
        expansion: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError("layers must be positive")
        try:
            # mamba-ssm 2.3 imports its optional Mamba-3 module eagerly.  Older
            # Triton releases used by this PyTorch environment do not expose
            # the allocator hook, although the Mamba-1 block below does not
            # need it.  Supplying the no-op hook keeps that optional import
            # from blocking the compatible selective-scan implementation.
            import triton

            if not hasattr(triton, "set_allocator"):
                triton.set_allocator = lambda _allocator: None
            from mamba_ssm import Mamba
        except ImportError as error:
            raise ImportError(
                "MambaSequence requires the optional mamba-ssm CUDA package"
            ) from error
        self.blocks = nn.ModuleList(
            Mamba(
                d_model=width,
                d_state=state_size,
                d_conv=convolution_width,
                expand=expansion,
            )
            for _ in range(layers)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(width) for _ in range(layers))
        self.dropout = nn.Dropout(dropout)
        self.final_norm = nn.LayerNorm(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for norm, block in zip(self.norms, self.blocks, strict=True):
            x = x + self.dropout(block(norm(x)))
        return self.final_norm(x)


class EcogTrajectoryDecoder(nn.Module):
    """Joint five-finger decoder with the paper's front end and a modern backbone."""

    def __init__(
        self,
        input_channels: int,
        spatial_components: int | None = None,
        feature_width: int = 64,
        temporal_backbone: str = "diagonal_ssm",
        temporal_layers: int = 3,
        state_size: int = 16,
        dropout: float = 0.1,
        output_fingers: int = 5,
        cnn_history_bins: int = 25,
        wavelet_frontend: WaveletPacketEnergy | None = None,
        direct_linear_head: bool = False,
        output_activation: str = "sigmoid",
        zero_initialize_residual: bool = False,
    ) -> None:
        super().__init__()
        if input_channels < 1 or output_fingers < 1:
            raise ValueError("channel and output dimensions must be positive")
        if cnn_history_bins < 1:
            raise ValueError("cnn_history_bins must be positive")
        spatial_components = input_channels if spatial_components is None else spatial_components
        if not 1 <= spatial_components <= input_channels:
            raise ValueError("spatial_components must be between 1 and input_channels")
        self.spatial_projection = nn.Conv1d(
            input_channels, spatial_components, kernel_size=1, bias=False
        )
        nn.init.orthogonal_(self.spatial_projection.weight[:, :, 0])
        self.wavelet_frontend = wavelet_frontend or WaveletPacketEnergy()
        band_count = 2**self.wavelet_frontend.levels
        self.cnn_history_bins = int(cnn_history_bins)
        flattened_features = spatial_components * band_count * self.cnn_history_bins
        self.feature_projection = nn.Sequential(
            nn.LayerNorm(flattened_features),
            nn.Linear(
                flattened_features,
                feature_width,
            ),
            nn.SiLU(),
        )
        if temporal_backbone == "diagonal_ssm":
            self.temporal = DiagonalSSM(
                feature_width,
                layers=temporal_layers,
                state_size=state_size,
                dropout=dropout,
            )
        elif temporal_backbone in {"gru", "lstm"}:
            recurrent = nn.GRU if temporal_backbone == "gru" else nn.LSTM
            self.temporal = recurrent(
                feature_width,
                feature_width,
                num_layers=temporal_layers,
                dropout=dropout if temporal_layers > 1 else 0.0,
                batch_first=True,
            )
        elif temporal_backbone == "linear_attention":
            self.temporal = CausalLinearAttention(
                feature_width,
                layers=temporal_layers,
                heads=4,
                dropout=dropout,
            )
        elif temporal_backbone == "mamba":
            self.temporal = MambaSequence(
                feature_width,
                layers=temporal_layers,
                state_size=state_size,
                dropout=dropout,
            )
        else:
            raise ValueError(
                "temporal_backbone must be 'diagonal_ssm', 'gru', 'lstm', "
                "'linear_attention', or 'mamba'"
            )
        self.temporal_backbone = temporal_backbone
        self.output = nn.Linear(feature_width, output_fingers)
        self.direct_output = (
            nn.Linear(flattened_features, output_fingers)
            if direct_linear_head
            else None
        )
        if output_activation not in {"sigmoid", "relu", "identity"}:
            raise ValueError("output_activation must be 'sigmoid', 'relu', or 'identity'")
        self.output_activation = output_activation
        if zero_initialize_residual:
            nn.init.zeros_(self.output.weight)
            nn.init.zeros_(self.output.bias)

    def initialize_direct_output_from_ridge(
        self,
        fits: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]],
    ) -> None:
        """Make the direct path exactly reproduce standardized ridge fits.

        Each tuple contains selected feature indices, their training mean and
        scale, the target mean, and standardized ridge weights. Unselected
        coefficients are zero, so a zero temporal branch starts from a working
        linear decoder instead of a random bottleneck.
        """
        if self.direct_output is None:
            raise RuntimeError("direct_linear_head was not enabled")
        if len(fits) != self.direct_output.out_features:
            raise ValueError("one ridge fit is required per output finger")
        with torch.no_grad():
            self.direct_output.weight.zero_()
            self.direct_output.bias.zero_()
            for finger, (indices, mean, scale, target_mean, weight) in enumerate(fits):
                indices = np.asarray(indices, dtype=np.int64)
                mean = np.asarray(mean, dtype=np.float64)
                scale = np.asarray(scale, dtype=np.float64)
                weight = np.asarray(weight, dtype=np.float64)
                if not (indices.shape == mean.shape == scale.shape == weight.shape):
                    raise ValueError("ridge fit arrays must have matching one-dimensional shapes")
                raw_weight = weight / scale
                self.direct_output.weight[finger, indices] = torch.as_tensor(
                    raw_weight,
                    dtype=self.direct_output.weight.dtype,
                    device=self.direct_output.weight.device,
                )
                self.direct_output.bias[finger] = float(
                    target_mean - np.dot(mean, raw_weight)
                )

    def initialize_spatial_from_fastica(
        self,
        training_ecog: np.ndarray,
        max_samples: int | None = 50_000,
        random_state: int = 0,
        backend: str = "sklearn",
        device: str | torch.device | None = None,
    ) -> np.ndarray:
        """Initialize the spatial convolution from a FastICA unmixing matrix."""
        weights = fit_fastica_spatial_weights(
            training_ecog,
            n_components=self.spatial_projection.out_channels,
            max_samples=max_samples,
            random_state=random_state,
            backend=backend,
            device=device,
        )
        if weights.shape[1] != self.spatial_projection.in_channels:
            raise ValueError("training_ecog channel count does not match the model")
        with torch.no_grad():
            tensor = torch.as_tensor(
                weights,
                dtype=self.spatial_projection.weight.dtype,
                device=self.spatial_projection.weight.device,
            )
            self.spatial_projection.weight[:, :, 0].copy_(tensor)
        return weights

    def forward(self, ecog: torch.Tensor) -> torch.Tensor:
        spatial = self.spatial_projection(ecog)
        energy = self.wavelet_frontend(spatial)
        per_bin = energy.permute(0, 3, 1, 2).flatten(start_dim=2)
        if per_bin.shape[1] < self.cnn_history_bins:
            raise ValueError("input is too short for the configured CNN history")
        history = per_bin.unfold(1, self.cnn_history_bins, 1)
        history_features = history.permute(0, 1, 3, 2).flatten(start_dim=2)
        features = self.feature_projection(history_features)
        if self.temporal_backbone in {"gru", "lstm"}:
            features, _ = self.temporal(features)
        else:
            features = self.temporal(features)
        prediction = self.output(features)
        if self.direct_output is not None:
            prediction = prediction + self.direct_output(history_features)
        if self.output_activation == "sigmoid":
            return torch.sigmoid(prediction)
        if self.output_activation == "relu":
            return torch.relu(prediction)
        return prediction
