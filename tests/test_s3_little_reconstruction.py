import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_s3_little_reconstruction.py"
SPEC = importlib.util.spec_from_file_location("s3_little_reconstruction", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_split_intervals_never_use_outer_validation_for_inner_training():
    definition = {
        "training_rows": 30,
        "folds": [
            {
                "training_intervals_after_purge": [[10, 30]],
                "validation_intervals": [[0, 8]],
            },
            {
                "training_intervals_after_purge": [[0, 8], [20, 30]],
                "validation_intervals": [[10, 18]],
            },
            {
                "training_intervals_after_purge": [[0, 18]],
                "validation_intervals": [[20, 30]],
            },
        ],
    }
    splits = MODULE.split_intervals(definition, 0)
    outer_validation = set(range(0, 8))
    for name, split in splits.items():
        training = set(MODULE.indices_from_intervals(split["training_intervals"]).tolist())
        assert training.isdisjoint(outer_validation), name


def test_paper_variant_changes_only_little(monkeypatch):
    raw = np.arange(100, dtype=np.float64).reshape(20, 5)
    local = np.full_like(raw, 2.0, dtype=np.float32)
    paper = np.full_like(raw, 3.0, dtype=np.float32)
    calls = iter((local, paper))
    monkeypatch.setattr(MODULE, "normalize_split", lambda *args, **kwargs: next(calls))
    target = MODULE.make_target(raw, [[0, 10]], [[10, 20]], "paper_no_wta")
    np.testing.assert_array_equal(target[:, :4], local[:, :4])
    np.testing.assert_array_equal(target[:, 4], paper[:, 4])


def test_paper_variants_share_baseline_fits(monkeypatch):
    raw = np.arange(100, dtype=np.float64).reshape(20, 5)
    calls = []

    def fake_normalize(*args, method, **kwargs):
        calls.append(method)
        value = 2.0 if method == "local" else 3.0
        return np.full_like(raw, value, dtype=np.float32)

    monkeypatch.setattr(MODULE, "normalize_split", fake_normalize)
    result = MODULE.make_targets(
        raw,
        [[0, 10]],
        [[10, 20]],
        ("paper_no_wta", "paper_wta_020"),
    )
    assert calls == ["local", "paper"]
    assert set(result) == {"paper_no_wta", "paper_wta_020"}


def test_soft_calibration_is_smooth_and_has_floor():
    prediction = np.zeros((4, 5), dtype=np.float32)
    prediction[:, 4] = 2.0
    probability = np.zeros((4, 5), dtype=np.float32)
    probability[:, 4] = [0.0, 0.25, 0.64, 1.0]
    candidates = MODULE.calibration_candidates("state_tcn", prediction, probability)
    np.testing.assert_allclose(
        candidates["soft_f0.25_g0.5"], [0.5, 1.25, 1.7, 2.0], atol=1.0e-6
    )


def test_csp_uses_training_bins_only():
    rng = np.random.default_rng(3)
    filtered = rng.normal(size=(70, 40, 3)).astype(np.float32)
    target = np.zeros((42, 5), dtype=np.float32)
    target[6:12, :] = 0.4
    target[12:18, 0] = 0.5
    target[18:24, 1] = 0.5
    target[24:30, 2] = 0.5
    target[30:36, 3] = 0.5
    target[36:42, 4] = 0.5
    training = np.arange(42)
    weights, audit = MODULE.csp_weights(filtered, target, training, components_per_tail=1)
    assert weights.shape == (10, 3)
    assert len(audit) == 5


def test_cpu_binned_energy_matches_manual_calculation():
    filtered = np.arange(24, dtype=np.float32).reshape(8, 3)
    weights = np.asarray([[1.0, 0.0, 0.0], [0.0, 0.5, 0.0]], dtype=np.float32)
    original = MODULE.SAMPLES_PER_BIN
    MODULE.SAMPLES_PER_BIN = 4
    try:
        actual = MODULE.binned_energy(filtered, weights)
    finally:
        MODULE.SAMPLES_PER_BIN = original
    projected = filtered @ weights.T
    expected = np.log1p(np.sqrt(np.sum(projected.reshape(2, 4, 2) ** 2, axis=1)))
    np.testing.assert_allclose(actual, expected, rtol=1.0e-6)


def test_beta_gamma_direct_initializer_uses_only_its_selected_bands(monkeypatch):
    captured = {}

    def fake_initialize(model, train_x, train_y, top_features, device):
        captured["shape"] = train_x.shape
        return {}

    monkeypatch.setattr(MODULE, "initialize_ridge", fake_initialize)
    energy = np.zeros((40, 140), dtype=np.float32)
    target = np.zeros((16, 5), dtype=np.float32)
    MODULE.build_model(
        "beta_gamma",
        energy,
        target,
        np.arange(16),
        MODULE.torch.device("cpu"),
    )
    assert captured["shape"] == (16, 60 * MODULE.HISTORY)


def test_probability_tempering_preserves_rows_and_softens_at_temperature_two():
    probability = np.asarray([[0.81, 0.19], [0.25, 0.75]], dtype=np.float64)
    tempered = MODULE.temper_probability(probability, 2.0)
    np.testing.assert_allclose(tempered.sum(axis=1), 1.0)
    assert tempered[0, 0] < probability[0, 0]
    assert tempered[1, 1] < probability[1, 1]


def test_zero_strength_latent_gate_is_exact_identity():
    prediction = np.asarray([0.2, 0.7, -0.1])
    little_gate = np.asarray([0.0, 0.5, 1.0])
    np.testing.assert_array_equal(
        MODULE.apply_little_gate(prediction, little_gate, 0.0), prediction
    )


def test_latent_classifier_and_emission_are_fit_on_training_only(monkeypatch):
    captured = {}

    class Classifier:
        classes_ = np.arange(6)

    def fake_fit(features, state, mask):
        captured["classifier_mask"] = mask.copy()
        return object(), Classifier()

    def fake_probabilities(scaler, classifier, features):
        return np.full((features.shape[0], 6), 1.0 / 6.0)

    def fake_emission(state, target, mask, threshold):
        captured["emission_mask"] = mask.copy()
        return np.full((6, 5), 0.5)

    monkeypatch.setattr(MODULE, "fit_classifier", fake_fit)
    monkeypatch.setattr(MODULE, "state_probabilities", fake_probabilities)
    monkeypatch.setattr(MODULE, "activity_emission", fake_emission)
    target = np.zeros((12, 5), dtype=np.float32)
    predictors = [np.zeros((12, 5), dtype=np.float32)]
    training = np.arange(0, 8)
    validation = np.arange(8, 12)
    gate, _ = MODULE.fit_latent_little_gate(
        predictors, target, training, validation
    )
    expected = np.zeros(12, dtype=bool)
    expected[training] = True
    np.testing.assert_array_equal(captured["classifier_mask"], expected)
    np.testing.assert_array_equal(captured["emission_mask"], expected)
    np.testing.assert_allclose(gate, 0.5)


def test_post_gate_gain_cannot_restore_more_rest_noise_than_ungated():
    target = np.asarray([0.0, 0.0, 1.0, 1.0])
    ungated = np.asarray([0.2, 0.2, 0.4, 0.4])
    gated = np.asarray([0.05, 0.05, 0.2, 0.2])
    gain, limits = MODULE.constrained_post_gate_gain(gated, ungated, target)
    gated_rest = np.sqrt(np.mean((gain * gated[target < 0.1]) ** 2))
    ungated_rest = np.sqrt(np.mean(ungated[target < 0.1] ** 2))
    assert gated_rest <= ungated_rest + 1.0e-12
    assert gain <= limits["rest_rms_limit_gain"] + 1.0e-12


def test_post_gate_gain_caps_peak_ratio_at_one_point_one():
    target = np.asarray([0.0, 0.0, 1.0, 1.0])
    ungated = np.asarray([0.2, 0.2, 0.4, 0.4])
    gated = np.asarray([0.01, 0.01, 0.3, 0.3])
    gain, _ = MODULE.constrained_post_gate_gain(gated, ungated, target)
    ratio = np.quantile(gain * gated[target >= 0.1], 0.95) / np.quantile(
        target[target >= 0.1], 0.95
    )
    assert ratio <= 1.10 + 1.0e-12
