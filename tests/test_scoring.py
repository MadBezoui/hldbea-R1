import numpy as np
import numpy.testing as npt
import pytest


def compute_local_scores(*args, **kwargs):
    from hldbea.scoring import compute_local_scores as implementation

    return implementation(*args, **kwargs)


def test_strict_score_matches_hand_computed_three_objective_neighborhoods():
    """Catches collapsing directional sub-neighborhoods into global dominance."""
    objectives = np.array(
        [
            [1.0, 1.0, 1.0],
            [0.7, 0.5, 0.5],
            [0.5, 0.7, 0.5],
            [0.5, 0.5, 0.7],
            [2.0, 2.0, 2.0],
        ]
    )

    result = compute_local_scores(objectives, k=1.0, cone_epsilon=0.0, lmbda=0.0)

    npt.assert_allclose(result.steps, [0.3, 0.3, 0.3])
    npt.assert_array_equal(result.per_axis_counts[0], [1, 1, 1])
    assert result.raw_scores[0] == 3


def test_axis_union_counts_a_neighbour_only_once_across_directional_bands():
    objectives = np.array(
        [
            [1.0, 1.0, 1.0],
            [0.7, 0.7, 0.7],
            [2.0, 2.0, 2.0],
        ]
    )

    summed = compute_local_scores(
        objectives,
        k=1.0,
        lmbda=0.0,
        neighborhood_mode="axis",
        score_aggregation="sum",
    )
    union = compute_local_scores(
        objectives,
        k=1.0,
        lmbda=0.0,
        neighborhood_mode="axis",
        score_aggregation="union",
    )

    npt.assert_array_equal(summed.per_axis_counts[0], [1, 1, 1])
    assert summed.raw_scores[0] == 3
    assert union.raw_scores[0] == 1


def test_full_box_excludes_axis_band_neighbours_that_are_far_on_other_axes():
    objectives = np.array(
        [
            [1.0, 1.0, 1.0],
            [0.7, 0.5, 0.5],
            [0.5, 0.7, 0.5],
            [0.5, 0.5, 0.7],
            [2.0, 2.0, 2.0],
        ]
    )

    axis = compute_local_scores(
        objectives,
        k=1.0,
        lmbda=0.0,
        neighborhood_mode="axis",
        score_aggregation="union",
    )
    box = compute_local_scores(
        objectives,
        k=1.0,
        lmbda=0.0,
        neighborhood_mode="box",
        score_aggregation="union",
    )

    assert axis.raw_scores[0] == 3
    assert box.raw_scores[0] == 0


def test_knn_neighbourhood_counts_only_local_relaxed_dominators():
    objectives = np.array(
        [
            [1.0, 1.0],
            [0.9, 0.9],
            [0.8, 1.2],
            [0.0, 0.0],
            [2.0, 2.0],
        ]
    )

    result = compute_local_scores(
        objectives,
        k=1.0,
        lmbda=0.0,
        neighborhood_mode="knn",
        score_aggregation="union",
    )

    # ceil(sqrt(5)) nearest neighbours include the close dominator but the
    # globally dominating origin is outside the local k-nearest set.
    assert result.raw_scores[0] == 1


def test_boundary_on_selected_axis_is_inclusive():
    """Catches changing the lower selected-axis bound from <= to <."""
    objectives = np.array(
        [
            [1.0, 1.0],
            [0.5, 0.0],
            [2.0, 2.0],
            [2.0, 0.5],
        ]
    )

    result = compute_local_scores(objectives, k=1.0, cone_epsilon=0.0, lmbda=0.0)

    npt.assert_allclose(result.steps, [0.375, 0.5])
    # Scale k so the first-axis lower bound is exactly B[0] = 0.5.
    boundary = compute_local_scores(objectives, k=4.0 / 3.0, cone_epsilon=0.0, lmbda=0.0)
    assert boundary.per_axis_counts[0, 0] == 1


def test_strict_excluded_axis_rejects_equality():
    """Catches legacy <= comparisons on axes excluded from the local band."""
    objectives = np.array(
        [
            [1.0, 1.0],
            [0.5, 1.0],
            [0.0, 0.0],
            [2.0, 2.0],
        ]
    )

    result = compute_local_scores(objectives, k=1.0, cone_epsilon=0.0, lmbda=0.0)

    assert result.per_axis_counts[0, 0] == 0


def test_duplicate_vectors_do_not_strictly_score_each_other():
    """Catches treating duplicate objective vectors as strict dominators."""
    objectives = np.array(
        [
            [1.0, 1.0],
            [1.0, 1.0],
            [0.0, 0.0],
            [2.0, 2.0],
        ]
    )

    result = compute_local_scores(objectives, k=1.0, cone_epsilon=0.0, lmbda=0.0)

    npt.assert_array_equal(result.per_axis_counts[0], [0, 0])
    npt.assert_array_equal(result.per_axis_counts[1], [0, 0])


def test_cone_relaxation_can_score_globally_nondominated_neighbor():
    """Catches cone scoring that is incorrectly prefiltered by Pareto dominance."""
    objectives = np.array(
        [
            [1.0, 1.0],
            [0.9, 1.05],
            [0.0, 2.0],
            [2.0, 0.0],
        ]
    )

    strict = compute_local_scores(objectives, cone_epsilon=0.0, lmbda=0.0)
    relaxed = compute_local_scores(objectives, cone_epsilon=0.2, lmbda=0.0)

    assert strict.raw_scores[0] == 0
    assert relaxed.raw_scores[0] == 1
    assert relaxed.nondominated[0]


def test_false_zero_is_zero_score_minus_global_nd():
    """Catches reporting every local zero as globally nondominated."""
    objectives = np.array(
        [
            [1.0, 1.0],
            [0.0, 0.0],
            [2.0, 2.0],
            [2.0, 0.5],
        ]
    )

    result = compute_local_scores(objectives, cone_epsilon=0.0, lmbda=0.0)

    assert result.raw_scores[0] == 0
    assert not result.nondominated[0]
    assert result.false_zero[0]


def test_augmented_fitness_keeps_distance_separate_from_raw_score():
    """Catches inferring raw-score strata from augmented-fitness equality."""
    objectives = np.array([[0.0, 1.0], [1.0, 0.0]])

    result = compute_local_scores(objectives, lmbda=0.1)

    npt.assert_array_equal(result.raw_scores, [0, 0])
    npt.assert_allclose(result.normalized_distances, [1.0, 1.0])
    npt.assert_allclose(result.fitness, [-0.1, -0.1])


def test_singleton_is_finite():
    """Catches division by zero for a one-individual population."""
    result = compute_local_scores(np.array([[3.0, 4.0]]), lmbda=0.1)

    npt.assert_array_equal(result.raw_scores, [0])
    npt.assert_array_equal(result.normalized_distances, [0.0])
    npt.assert_array_equal(result.fitness, [0.0])
    npt.assert_array_equal(result.nondominated, [True])
    npt.assert_array_equal(result.false_zero, [False])


def test_constant_population_has_zero_distance():
    """Catches unstable normalization when all objective vectors coincide."""
    result = compute_local_scores(np.ones((3, 2)), lmbda=0.1)

    npt.assert_array_equal(result.raw_scores, [0, 0, 0])
    npt.assert_array_equal(result.normalized_distances, [0.0, 0.0, 0.0])
    npt.assert_array_equal(result.fitness, [0.0, 0.0, 0.0])
    npt.assert_array_equal(result.nondominated, [True, True, True])


@pytest.mark.parametrize(
    ("objectives", "kwargs"),
    [
        (np.array([1.0, 2.0]), {}),
        (np.empty((0, 2)), {}),
        (np.ones((2, 1)), {}),
        (np.array([[0.0, np.nan], [1.0, 2.0]]), {}),
        (np.ones((2, 2)), {"k": -1.0}),
        (np.ones((2, 2)), {"cone_epsilon": -0.1}),
        (np.ones((2, 2)), {"lmbda": -0.1}),
        (np.ones((2, 2)), {"safe_epsilon": 0.0}),
        (np.ones((2, 2)), {"neighborhood_mode": "unknown"}),
        (np.ones((2, 2)), {"score_aggregation": "unknown"}),
        (
            np.ones((2, 2)),
            {"neighborhood_mode": "box", "score_aggregation": "sum"},
        ),
    ],
)
def test_invalid_score_inputs_raise_value_error(objectives, kwargs):
    """Catches silent propagation of malformed experimental data."""
    with pytest.raises(ValueError):
        compute_local_scores(objectives, **kwargs)
