import json

import numpy as np

from scripts.refit_frozen_event_model import frozen_epoch, lars_chunks, resolve_options


def test_options_merge_per_finger_override() -> None:
    mapping = {
        "default": {"input_root": "base", "seeds": [0, 1], "sequence_steps": 50},
        "subjects": {1: {"index": {"input_root": "long", "sequence_steps": 100}}},
    }
    options = resolve_options(mapping, 1, "index")
    assert str(options["input_root"]) == "long"
    assert options["seeds"] == (0, 1)
    assert options["epoch_reference_seeds"] == (0, 1)
    assert options["sequence_steps"] == 100


def test_lars_chunks_cover_every_row_once() -> None:
    chunks = lars_chunks(9976)
    indices = np.concatenate([np.arange(start, stop) for start, stop in chunks])
    assert np.array_equal(indices, np.arange(9976))


def test_frozen_epoch_uses_one_pooled_median_for_all_members(tmp_path) -> None:
    expected = {"0": [3, 11, 7], "1": [5, 13, 9]}
    for seed, epochs in expected.items():
        for fold, epoch in enumerate(epochs):
            path = tmp_path / "sub1" / "ring" / f"fold{fold}" / f"seed{seed}"
            path.mkdir(parents=True)
            (path / "summary.json").write_text(json.dumps({"selected_epoch": epoch}))
    selected, values = frozen_epoch(tmp_path, 1, "ring", (0, 1))
    assert values == expected
    assert selected == 8
