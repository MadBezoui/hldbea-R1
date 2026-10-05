import numpy as np
import numpy.testing as npt
import pytest


def rank_and_crowding_indices(*args, **kwargs):
    from hldbea.selection import rank_and_crowding_indices as implementation

    return implementation(*args, **kwargs)


def select_survivor_indices(*args, **kwargs):
    from hldbea.selection import select_survivor_indices as implementation

    return implementation(*args, **kwargs)


def test_rank_and_crowding_includes_a_complete_front():
    """Catches dropping members when an entire front fits."""
    objectives = np.array([[0.0, 3.0], [1.0, 2.0], [2.0, 1.0], [3.0, 0.0]])

    selected = rank_and_crowding_indices(objectives, 4)

    assert set(selected) == {0, 1, 2, 3}


def test_rank_and_crowding_partial_front_preserves_extremes():
    """Catches truncation that ignores crowding-distance boundary points."""
    objectives = np.array([[0.0, 3.0], [1.0, 2.0], [2.0, 1.0], [3.0, 0.0]])

    selected = rank_and_crowding_indices(objectives, 2)

    npt.assert_array_equal(selected, [0, 3])


def test_rank_and_crowding_resolves_equal_crowding_by_original_index():
    """Catches nondeterministic randomized tie handling in confirmatory runs."""
    objectives = np.array(
        [[0.0, 4.0], [1.0, 3.0], [2.0, 2.0], [3.0, 1.0], [4.0, 0.0]]
    )

    first = rank_and_crowding_indices(objectives, 3)
    second = rank_and_crowding_indices(objectives, 3)

    npt.assert_array_equal(first, [0, 4, 1])
    npt.assert_array_equal(second, first)


def test_rank_and_crowding_returns_unique_valid_indices_with_duplicates():
    """Catches duplicate-row bookkeeping returning repeated or invalid indices."""
    objectives = np.array([[0.0, 1.0], [0.0, 1.0], [1.0, 0.0], [2.0, 2.0]])

    selected = rank_and_crowding_indices(objectives, 3)

    assert len(selected) == len(set(selected.tolist())) == 3
    assert np.all((0 <= selected) & (selected < len(objectives)))


@pytest.mark.parametrize("n_survive", [0, -1, 4])
def test_rank_and_crowding_rejects_invalid_survivor_count(n_survive):
    """Catches silent underflow or overflow of the survivor population."""
    with pytest.raises(ValueError):
        rank_and_crowding_indices(np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 2.0]]), n_survive)


def test_local_selection_keeps_zero_stratum_and_fills_from_rest():
    """Catches replacing raw-score-zero individuals because their fitness differs."""
    objectives = np.array(
        [[0.0, 5.0], [1.0, 4.0], [5.0, 0.0], [2.0, 3.0], [3.0, 2.0]]
    )
    raw_scores = np.array([0, 2, 0, 1, 3])

    selected = select_survivor_indices(objectives, raw_scores, 4, mode="local")

    npt.assert_array_equal(selected, [0, 2, 1, 4])


def test_local_selection_trims_only_within_overfull_zero_stratum():
    """Catches a nonzero-score point entering while zero-score candidates overflow."""
    objectives = np.array([[0.0, 3.0], [1.0, 2.0], [3.0, 0.0], [-1.0, -1.0]])
    raw_scores = np.array([0, 0, 0, 1])

    selected = select_survivor_indices(objectives, raw_scores, 2, mode="local")

    npt.assert_array_equal(selected, [0, 2])


def test_global_selection_ignores_raw_score_strata():
    """Catches the local-versus-global ablation accidentally using one implementation."""
    objectives = np.array([[1.0, 1.0], [0.0, 0.0], [2.0, 2.0]])
    raw_scores = np.array([0, 1, 0])

    local = select_survivor_indices(objectives, raw_scores, 1, mode="local")
    global_selection = select_survivor_indices(objectives, raw_scores, 1, mode="global")

    npt.assert_array_equal(local, [0])
    npt.assert_array_equal(global_selection, [1])


def test_local_selection_preserves_all_zero_indices_when_population_fits():
    """Catches reordering that loses a zero-score member during fill."""
    objectives = np.array([[0.0, 2.0], [1.0, 1.0], [2.0, 0.0]])
    raw_scores = np.array([0, 1, 0])

    selected = select_survivor_indices(objectives, raw_scores, 3, mode="local")

    assert set(selected[:2]) == {0, 2}
    assert set(selected) == {0, 1, 2}


@pytest.mark.parametrize(
    ("raw_scores", "n_survive", "mode"),
    [
        (np.array([0, 1]), 1, "local"),
        (np.array([0, 1, 2]), 0, "local"),
        (np.array([0, 1, 2]), 1, "unknown"),
    ],
)
def test_survivor_selection_rejects_invalid_contract(raw_scores, n_survive, mode):
    """Catches malformed score vectors and undeclared selection variants."""
    objectives = np.array([[0.0, 2.0], [1.0, 1.0], [2.0, 0.0]])
    with pytest.raises(ValueError):
        select_survivor_indices(objectives, raw_scores, n_survive, mode=mode)
