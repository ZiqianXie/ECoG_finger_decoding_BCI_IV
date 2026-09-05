import numpy as np

from render_extension_report import load_source_audit, normalize_for_display


def test_display_normalization_is_nonnegative_and_matches_development_scale():
    prediction = np.linspace(-0.3, 0.7, 401)
    development = np.linspace(0.0, 1.2, 401)

    display, audit = normalize_for_display(prediction, development)

    assert np.isfinite(display).all()
    assert np.min(display) >= 0.0
    assert audit["gain"] > 0.0
    np.testing.assert_allclose(
        np.quantile(display, 0.995),
        np.quantile(development, 0.995),
        rtol=1.0e-6,
    )


def test_source_audit_surfaces_test_informed_blend(tmp_path):
    subject = tmp_path / "sub1"
    subject.mkdir()
    (tmp_path / "summary.json").write_text(
        '{"protocol":"retrospective",'
        '"released_test_used_for_weight_selection":true,'
        '"suitable_as_confirmatory_benchmark":false,'
        '"right_weight":0.096,"blend_pcc":0.752}'
    )

    audit = load_source_audit(str(subject / "test_prediction.npy"))

    assert audit["released_test_used_for_weight_selection"] is True
    assert audit["suitable_as_confirmatory_benchmark"] is False
    assert audit["right_weight"] == 0.096
