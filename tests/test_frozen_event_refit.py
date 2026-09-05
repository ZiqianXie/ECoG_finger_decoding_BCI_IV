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
    assert options["sequence_steps"] == 100


def test_lars_chunks_cover_every_row_once() -> None:
    chunks = lars_chunks(9976)
    indices = np.concatenate([np.arange(start, stop) for start, stop in chunks])
    assert np.array_equal(indices, np.arange(9976))


def test_frozen_epoch_uses_rounded_outer_median(tmp_path) -> None:
    for fold, epoch in enumerate((3, 11, 7)):
        path = tmp_path / "sub1" / "ring" / f"fold{fold}" / "seed0"
        path.mkdir(parents=True)
        (path / "summary.json").write_text(json.dumps({"selected_epoch": epoch}))
    selected, values = frozen_epoch(tmp_path, 1, "ring", 0)
    assert values == [3, 11, 7]
    assert selected == 7
