from refit_frozen_event_model import update_equivalent_schedule


def test_update_schedule_scales_for_larger_full_refit() -> None:
    definition = {
        "folds": [
            {"training_intervals_after_purge": [[0, 500]]},
            {"training_intervals_after_purge": [[0, 500]]},
            {"training_intervals_after_purge": [[0, 500]]},
        ]
    }
    total, warmup, audit = update_equivalent_schedule(
        epochs_by_seed={"0": [20, 8, 12]},
        seed=0,
        definition=definition,
        sequence_steps=100,
        sequence_stride=25,
        batch_size=24,
        unfrozen_batch_size=4,
        warmup_epochs=8,
        full_rows=1000,
    )
    assert warmup == 4
    assert total == 6
    assert audit["transferred_end_to_end_epochs"] == 2
    assert audit["median_frozen_updates"] == 8
    assert audit["median_end_to_end_updates"] == 20


def test_update_schedule_pools_reference_seeds_for_new_seed() -> None:
    definition = {
        "folds": [
            {"training_intervals_after_purge": [[0, 500]]},
            {"training_intervals_after_purge": [[0, 500]]},
            {"training_intervals_after_purge": [[0, 500]]},
        ]
    }
    total, warmup, audit = update_equivalent_schedule(
        epochs_by_seed={"0": [20, 8, 12], "1": [16, 8, 10]},
        seed=5,
        definition=definition,
        sequence_steps=100,
        sequence_stride=25,
        batch_size=24,
        unfrozen_batch_size=4,
        warmup_epochs=8,
        full_rows=1000,
    )
    assert warmup == 4
    assert total == 6
    assert len(audit["reference"]) == 6
