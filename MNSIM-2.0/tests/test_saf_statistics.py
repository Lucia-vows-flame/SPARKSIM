import numpy as np
import pytest

from MNSIM.Accuracy_Model.saf_statistics import (
    MAX_SEED,
    parse_seed_list,
    summarize_weight_changes,
    temporary_numpy_seed,
)


def test_parse_seed_list_accepts_repeated_and_comma_separated_values():
    assert parse_seed_list(["1, 2", "2", "3"]) == [1, 2, 3]


@pytest.mark.parametrize("values", [[""], ["-1"], [str(MAX_SEED + 1)], ["x"]])
def test_parse_seed_list_rejects_invalid_values(values):
    with pytest.raises(ValueError):
        parse_seed_list(values)


def test_summarize_weight_changes_counts_effective_fault_directions():
    original = [{"a": np.array([[0, 1, 3], [4, 5, 6]], dtype=float)}, None]
    updated = [{"a": np.array([[0, 0, 3], [4, 7, 6]], dtype=float)}, None]

    summary = summarize_weight_changes(original, updated, max_code=7)

    assert summary["total_codes"] == 6
    assert summary["changed_codes"] == 2
    assert summary["changed_to_zero"] == 1
    assert summary["changed_to_max"] == 1
    assert summary["changed_fraction"] == pytest.approx(2 / 6)


def test_temporary_numpy_seed_is_reproducible_and_restores_state():
    np.random.seed(1234)
    before = np.random.get_state()
    with temporary_numpy_seed(20260809):
        first = np.random.random_sample()
    after = np.random.get_state()
    with temporary_numpy_seed(20260809):
        second = np.random.random_sample()

    assert first == second
    assert after[0] == before[0]
    assert np.array_equal(after[1], before[1])
    assert after[2:] == before[2:]
