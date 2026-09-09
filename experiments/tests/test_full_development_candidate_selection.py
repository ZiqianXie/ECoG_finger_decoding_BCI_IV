from __future__ import annotations

import pytest

from scripts.select_full_development_candidates import (
    select_candidates,
    validate_report,
)


def fake_report(thumb_score: float, index_score: float) -> dict[str, object]:
    per_finger = {}
    for finger in ("thumb", "index", "middle", "ring", "little"):
        score = thumb_score if finger == "thumb" else index_score
        per_finger[finger] = {
            "ensemble_oof_pcc": score,
            "included_seeds": [0, 1],
        }
    return {
        "released_test_touched": False,
        "selection_scopes": ["full-development"],
        "subjects": {"1": {"per_finger": per_finger}},
    }


def test_candidate_selection_is_per_finger() -> None:
    reports = {
        "short": fake_report(0.6, 0.4),
        "long": fake_report(0.5, 0.7),
    }
    options = {
        "short": {"summary": "short.json", "input_root": "short", "sequence_steps": 50},
        "long": {"summary": "long.json", "input_root": "long", "sequence_steps": 100},
    }
    ensemble_map, audit = select_candidates(
        reports, options, {"learning_rate": 1.0e-4}, subjects=(1,)
    )
    assert audit["subjects"]["1"]["per_finger"]["thumb"]["selected"] == "short"
    assert audit["subjects"]["1"]["per_finger"]["index"]["selected"] == "long"
    assert ensemble_map["subjects"][1]["thumb"]["sequence_steps"] == 50
    assert ensemble_map["subjects"][1]["index"]["sequence_steps"] == 100


def test_test_touched_summary_is_rejected() -> None:
    report = fake_report(0.5, 0.5)
    report["released_test_touched"] = True
    with pytest.raises(ValueError, match="released test"):
        validate_report(report, "full-development")
