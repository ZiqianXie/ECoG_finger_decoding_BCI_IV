import importlib.util
import sys
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "select_single_wavelet_lr_by_inner.py"
)
SPEC = importlib.util.spec_from_file_location("select_single_wavelet_lr_by_inner", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def report(selected_schedule: str, mean_pcc: float, sem_pcc: float):
    return {
        "outer_folds": [
            {
                "outer_fold": 0,
                "selected_schedule": selected_schedule,
                "selection_summary": {
                    selected_schedule: {
                        "mean_raw_pcc": mean_pcc,
                        "sem_raw_pcc": sem_pcc,
                    }
                },
            }
        ]
    }


def test_choose_candidate_uses_selected_schedule_inner_pcc_only(tmp_path):
    low = MODULE.Candidate("low", tmp_path / "low", report("frozen_50", 0.41, 0.02))
    high = MODULE.Candidate(
        "high", tmp_path / "high", report("unfrozen_100_20", 0.47, 0.03)
    )

    selected, record, mean_pcc, sem_pcc = MODULE.choose_candidate([low, high], 0)

    assert selected.name == "high"
    assert record["selected_schedule"] == "unfrozen_100_20"
    assert mean_pcc == 0.47
    assert sem_pcc == 0.03


def test_candidate_argument_preserves_equals_in_path():
    name, path = MODULE.candidate_argument("lr=folder=variant")
    assert name == "lr"
    assert path == Path("folder=variant")
