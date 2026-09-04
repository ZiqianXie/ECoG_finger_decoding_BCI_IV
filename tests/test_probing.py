import numpy as np
import torch
from torch import nn

from ecog_decoding.probing import (
    decode_feature_groups,
    frequency_response_summary,
    manual_lstm_trace,
    matched_rest_donors,
    phase_masks,
)


def test_manual_lstm_trace_matches_pytorch() -> None:
    torch.manual_seed(3)
    lstm = nn.LSTM(4, 3, batch_first=True)
    temporal = nn.Linear(3, 1)
    direct = nn.Linear(4, 1)
    values = torch.randn(2, 7, 4)

    recurrent, _ = lstm(values)
    expected = torch.relu(direct(values) + temporal(recurrent)).squeeze(-1)
    trace = manual_lstm_trace(lstm, temporal, direct, values)

    assert torch.allclose(trace["hidden_state"], recurrent, atol=1.0e-6)
    assert torch.allclose(trace["prediction"], expected, atol=1.0e-6)
    assert torch.allclose(
        trace["cell_state"], trace["retention"] + trace["cell_write"], atol=1.0e-7
    )


def test_feature_groups_follow_component_band_time_flattening() -> None:
    groups = decode_feature_groups(
        np.asarray([0, 1, 5, 6, 17]),
        component_count=2,
        band_names=("low", "mid", "high"),
        energy_bins=3,
    )

    assert [(group.component, group.band) for group in groups] == [
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 2),
    ]
    assert groups[0].selected_columns.tolist() == [0, 1]
    assert groups[-1].source_indices.tolist() == [17]


def test_phase_masks_separate_transitions_and_false_positives() -> None:
    target = np.asarray([0, 0, 0.2, 0.4, 0.3, 0, 0, 0, 0.2])
    prediction = np.asarray([0, 0.2, 0.2, 0.4, 0.3, 0.2, 0, 0, 0.2])
    phases = phase_masks(target, prediction, threshold=0.1, transition_bins=2)

    assert np.flatnonzero(phases["onset"]).tolist() == [2, 3, 8]
    assert np.flatnonzero(phases["sustained_movement"]).tolist() == [4]
    assert np.flatnonzero(phases["release"]).tolist() == [5, 6]
    assert np.flatnonzero(phases["rest_false_positive"]).tolist() == [1, 5]


def test_matched_rest_donors_preserve_other_finger_state() -> None:
    features = np.asarray([[0], [1], [2], [3], [4], [5]], dtype=np.float32)
    targets = np.asarray(
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
            [0, 0],
            [0, 1],
        ],
        dtype=np.float32,
    )
    donors = matched_rest_donors(features, targets, 0, np.asarray([2, 3]))

    assert donors.tolist() == [0, 1]


def test_frequency_response_removes_probe_impulse_offset() -> None:
    impulse = np.zeros((1, 128), dtype=np.float32)
    impulse[0, 64] = 1.0
    _, magnitude, summary = frequency_response_summary(impulse, sampling_rate_hz=1000.0)

    assert np.allclose(magnitude, 1.0)
    assert abs(summary[0]["group_delay_ms"]) < 1.0e-9
    assert summary[0]["stopband_leakage"] == 0.0
