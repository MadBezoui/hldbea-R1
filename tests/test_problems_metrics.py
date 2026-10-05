import numpy as np
import numpy.testing as npt
import pytest
from pymoo.problems import get_problem


def _spec(name="dtlz2", n_obj=3, n_var=12, transform=None):
    from hldbea.problems import ProblemSpec

    return ProblemSpec(
        id=f"{name}-m{n_obj}",
        name=name,
        n_obj=n_obj,
        n_var=n_var,
        parameters={},
        transform=transform or {"kind": "identity"},
    )


@pytest.mark.parametrize(
    ("name", "n_obj", "n_var"),
    [("zdt3", 2, 30), ("dtlz2", 5, 14), ("wfg2", 5, 24)],
)
def test_problem_registry_builds_supported_families(name, n_obj, n_var):
    from hldbea.problems import build_problem

    problem = build_problem(_spec(name, n_obj, n_var))
    x = np.row_stack((problem.xl, problem.xu))
    objectives = problem.evaluate(x, return_values_of=["F"])

    assert objectives.shape == (2, n_obj)
    assert np.all(np.isfinite(objectives))


@pytest.mark.parametrize(
    ("name", "n_obj", "expected"),
    [
        ("imop3", 2, [1.2, 0.0]),
        ("imop4", 3, [1.0, 1.0, 0.0]),
        ("imop7", 3, [0.0, 0.0, 1.0]),
    ],
)
def test_imop_ports_match_source_formula_at_known_pareto_decisions(
    name, n_obj, expected
):
    from hldbea.problems import build_problem

    spec = _spec(name, n_obj, 10)
    problem = build_problem(spec)
    decisions = np.full((1, 10), 0.5)
    if name == "imop3":
        decisions[0, :5] = 0.0
    else:
        decisions[0, :5] = 1.0

    objectives = problem.evaluate(decisions, return_values_of=["F"])

    npt.assert_allclose(objectives[0], expected, atol=1e-12)


@pytest.mark.parametrize(("name", "n_obj"), [("imop3", 2), ("imop4", 3), ("imop7", 3)])
def test_imop_reference_geometry_is_finite_nontrivial_and_deterministic(name, n_obj):
    from hldbea.metrics import build_reference_geometry

    spec = _spec(name, n_obj, 10)
    first = build_reference_geometry(spec)
    second = build_reference_geometry(spec)

    assert first.reference_set.shape[1] == n_obj
    assert len(first.reference_set) >= 100
    assert np.all(np.isfinite(first.reference_set))
    assert np.all(first.nadir > first.ideal)
    npt.assert_array_equal(first.reference_set, second.reference_set)
    assert first.geometry_hash == second.geometry_hash


def test_imop_problem_rejects_incompatible_objective_count():
    from hldbea.problems import build_problem

    with pytest.raises(ValueError, match="dimensions"):
        build_problem(_spec("imop3", 3, 10))


def test_identity_problem_preserves_base_objectives_exactly():
    from hldbea.problems import build_problem

    spec = _spec("dtlz2", 3, 12)
    problem = build_problem(spec)
    base = get_problem("dtlz2", n_var=12, n_obj=3)
    x = np.random.default_rng(4).random((7, 12))

    npt.assert_allclose(
        problem.evaluate(x, return_values_of=["F"]),
        base.evaluate(x, return_values_of=["F"]),
    )
    npt.assert_allclose(problem.objective_transform["matrix"], np.eye(3))


def test_seeded_rotation_is_shared_reproducible_and_invertible():
    from hldbea.problems import build_problem

    first = build_problem(
        _spec("dtlz2", 5, 14, {"kind": "rotation", "seed": 912})
    )
    second = build_problem(
        _spec("dtlz2", 5, 14, {"kind": "rotation", "seed": 912})
    )
    different = build_problem(
        _spec("dtlz2", 5, 14, {"kind": "rotation", "seed": 913})
    )
    matrix = first.objective_transform["matrix"]
    npt.assert_allclose(matrix @ matrix.T, np.eye(5), atol=1e-12)
    npt.assert_allclose(matrix, second.objective_transform["matrix"])
    assert not np.allclose(matrix, different.objective_transform["matrix"])

    original = np.random.default_rng(8).normal(size=(9, 5))
    transformed = first.transform_objectives(original)
    recovered = first.inverse_transform_objectives(transformed)
    npt.assert_allclose(recovered, original, atol=1e-12)
    assert first.objective_transform["hash"] == second.objective_transform["hash"]


def test_angle_rotation_uses_declared_givens_angle():
    from hldbea.problems import build_problem

    problem = build_problem(
        _spec("dtlz2", 3, 12, {"kind": "rotation", "angle_degrees": 30.0})
    )
    matrix = problem.objective_transform["matrix"]

    npt.assert_allclose(matrix @ matrix.T, np.eye(3), atol=1e-12)
    npt.assert_allclose(matrix[2], [0.0, 0.0, 1.0], atol=1e-12)
    npt.assert_allclose(matrix[0, :2], [np.sqrt(3) / 2, -0.5], atol=1e-12)


def test_reference_geometry_matches_stored_transformed_front_and_hash():
    from hldbea.metrics import build_reference_geometry
    from hldbea.problems import build_problem

    spec = _spec("dtlz2", 3, 12, {"kind": "rotation", "seed": 77})
    problem = build_problem(spec)
    geometry = build_reference_geometry(spec)
    front = problem.pareto_front(
        ref_dirs=geometry.reference_directions
    )

    npt.assert_allclose(geometry.reference_set, front)
    assert geometry.geometry_hash == build_reference_geometry(spec).geometry_hash
    assert geometry.transform_hash == problem.objective_transform["hash"]
    npt.assert_array_equal(geometry.hv_reference_point, np.full(3, 1.1))


def test_wfg_reference_geometry_is_deterministic_and_preserves_global_rng():
    from hldbea.metrics import build_reference_geometry

    spec = _spec("wfg2", 3, 24)
    expected_draws = np.random.RandomState(9182).random_sample(5)
    np.random.seed(9182)
    first = build_reference_geometry(spec)
    observed_draws = np.random.random(5)
    second = build_reference_geometry(spec)

    npt.assert_array_equal(observed_draws, expected_draws)
    npt.assert_array_equal(first.reference_set, second.reference_set)
    assert first.geometry_hash == second.geometry_hash


def test_many_objective_wfg_reference_front_uses_bounded_pareto_set(
    monkeypatch,
):
    from hldbea.metrics import metric_reference_directions
    from hldbea.problems import _base_problem, _pareto_front, _wfg_reference_decisions

    spec = _spec("wfg1", 15, 38)
    problem = _base_problem(spec)

    def reject_exponential_extremes():
        raise AssertionError("the 2**k pymoo extreme-set path must not be used")

    monkeypatch.setattr(
        problem,
        "_calc_pareto_set_extremes",
        reject_exponential_extremes,
    )
    directions = metric_reference_directions(15)
    decisions = _wfg_reference_decisions(problem, directions)
    front = _pareto_front(problem, ref_dirs=directions)

    assert 512 <= len(front) <= 2048
    assert front.shape == (len(decisions), 15)
    assert np.all(np.isfinite(front))
    assert np.all(np.ptp(front, axis=0) > 0)
    npt.assert_allclose(
        front,
        problem.evaluate(decisions, return_values_of=["F"]),
        atol=0.0,
        rtol=0.0,
    )


def test_normalized_metrics_are_fixed_and_algorithm_independent():
    from hldbea.metrics import build_reference_geometry, compute_metrics, normalize_objectives

    geometry = build_reference_geometry(_spec("dtlz2", 3, 12))
    approximation = geometry.reference_set[::5]
    first = compute_metrics(approximation, geometry)
    second = compute_metrics(approximation.copy(), geometry)
    normalized = normalize_objectives(geometry.reference_set, geometry)

    assert first == second
    assert np.isfinite(first["hv"])
    assert np.isfinite(first["igd_plus"])
    npt.assert_allclose(np.min(normalized, axis=0), 0.0, atol=1e-12)
    npt.assert_allclose(np.max(normalized, axis=0), 1.0, atol=1e-12)


def test_many_objective_hypervolume_uses_reproducible_sobol_estimator():
    from hldbea.metrics import compute_hypervolume, hypervolume_estimator_metadata

    points = np.array(
        [
            [0.20] * 10,
            [0.35, 0.15] * 5,
        ]
    )
    ideal = np.zeros(10)
    reference = np.ones(10)

    first = compute_hypervolume(points, reference, ideal_point=ideal)
    second = compute_hypervolume(points.copy(), reference, ideal_point=ideal)
    augmented = compute_hypervolume(
        np.row_stack((points, np.full(10, 0.10))),
        reference,
        ideal_point=ideal,
    )
    metadata = hypervolume_estimator_metadata(10)

    assert first == second
    assert 0.0 < first < 1.0
    assert augmented >= first
    assert metadata == {
        "kind": "sobol_qmc",
        "exact_dimension_limit": 5,
        "samples": 65536,
        "scramble": True,
        "seed": 20261003,
        "lower_bound": "reference_ideal",
    }


def test_numba_dominance_counter_matches_vectorized_definition():
    from hldbea.metrics import _dominated_sample_count

    samples = np.array(
        [
            [0.10, 0.10, 0.10],
            [0.50, 0.50, 0.50],
            [0.90, 0.20, 0.80],
            [0.95, 0.95, 0.95],
        ]
    )
    points = np.array([[0.20, 0.20, 0.20], [0.80, 0.10, 0.70]])
    expected = int(
        np.count_nonzero(
            np.any(np.all(points[:, None, :] <= samples[None, :, :], axis=2), axis=0)
        )
    )

    assert _dominated_sample_count(samples, points) == expected


def test_low_dimensional_hypervolume_remains_exact():
    from pymoo.indicators.hv import HV

    from hldbea.metrics import compute_hypervolume, hypervolume_estimator_metadata

    points = np.array([[0.2, 0.8, 0.4], [0.7, 0.3, 0.5]])
    reference = np.ones(3)

    observed = compute_hypervolume(
        points,
        reference,
        ideal_point=np.zeros(3),
    )

    assert observed == float(HV(ref_point=reference)(points))
    assert hypervolume_estimator_metadata(3) == {"kind": "exact"}


@pytest.mark.parametrize("n_obj", [2, 3, 5, 8, 10, 15])
def test_reference_geometry_supports_all_declared_dimensions(n_obj):
    from hldbea.metrics import build_reference_geometry

    geometry = build_reference_geometry(_spec("dtlz2", n_obj, n_obj + 9))

    assert geometry.reference_set.ndim == 2
    assert geometry.reference_set.shape[1] == n_obj
    assert geometry.ideal.shape == (n_obj,)
    assert geometry.nadir.shape == (n_obj,)
    assert geometry.hv_reference_point.shape == (n_obj,)


@pytest.mark.parametrize(
    ("name", "n_obj", "n_var"),
    [("dtlz5", 5, 14), ("dtlz6", 10, 19)],
)
def test_degenerate_dtlz_reference_front_uses_exact_g_zero_curve(
    name, n_obj, n_var
):
    from hldbea.metrics import build_reference_geometry, compute_metrics
    from hldbea.problems import build_problem

    spec = _spec(name, n_obj, n_var)
    geometry = build_reference_geometry(spec)
    problem = build_problem(spec)
    decisions = np.full((len(geometry.reference_set), n_var), 0.5)
    decisions[:, 0] = np.linspace(0.0, 1.0, len(decisions))
    if name == "dtlz6":
        decisions[:, n_obj - 1 :] = 0.0
    expected = problem.evaluate(decisions, return_values_of=["F"])

    npt.assert_allclose(geometry.reference_set, expected, atol=1e-12, rtol=1e-12)
    assert np.all(geometry.nadir > geometry.ideal)
    assert all(
        np.isfinite(value)
        for value in compute_metrics(geometry.reference_set[::7], geometry).values()
    )


def test_degenerate_dtlz_reference_front_supports_seeded_rotation():
    from hldbea.metrics import build_reference_geometry

    geometry = build_reference_geometry(
        _spec("dtlz5", 5, 14, {"kind": "rotation", "seed": 441})
    )

    assert geometry.reference_set.shape[1] == 5
    assert np.all(np.isfinite(geometry.reference_set))
    assert np.all(geometry.nadir > geometry.ideal)


def test_metric_dimension_mismatch_is_rejected():
    from hldbea.metrics import build_reference_geometry, compute_metrics

    geometry = build_reference_geometry(_spec("dtlz2", 3, 12))
    with pytest.raises(ValueError, match="dimension"):
        compute_metrics(np.ones((4, 2)), geometry)


def test_score_diagnostics_report_nonzero_pressure_and_false_zeros():
    from hldbea.metrics import compute_score_diagnostics

    objectives = np.array([[0.2, 0.2], [0.8, 0.8]])
    diagnostics = compute_score_diagnostics(objectives, k=0.1, cone_epsilon=0.0)

    assert diagnostics == {"phi_nz": 0.0, "r_fz": 0.5}


@pytest.mark.parametrize(
    "transform",
    [
        {"kind": "rotation"},
        {"kind": "rotation", "seed": 1, "angle_degrees": 30.0},
        {"kind": "rotation", "matrix": [[1, 0], [0, 2]]},
    ],
)
def test_invalid_rotation_contract_is_rejected(transform):
    from hldbea.problems import build_problem

    with pytest.raises(ValueError):
        build_problem(_spec("dtlz2", 2, 11, transform))


def test_rotated_front_dtlz2_has_analytical_front_and_uniform_distance_penalty():
    """A rotated front must not rotate the dominance-preserving distance term."""

    from hldbea.problems import ProblemSpec, build_problem

    spec = ProblemSpec(
        id="rdtlz2-m3-a30",
        name="rdtlz2",
        n_obj=3,
        n_var=12,
        parameters={"angle_degrees": 30.0},
        transform={"kind": "identity"},
    )
    problem = build_problem(spec)
    on_front = np.full((2, 12), 0.5)
    on_front[0, :2] = [0.0, 0.0]
    on_front[1, :2] = [0.0, 1.0]
    off_front = on_front.copy()
    off_front[:, 2:] = 0.6

    observed = problem.evaluate(on_front, return_values_of=["F"])
    degraded = problem.evaluate(off_front, return_values_of=["F"])

    npt.assert_allclose(
        observed,
        [
            [np.sqrt(3.0) / 2.0 + 0.5, 0.5, 0.0],
            [0.0, np.sqrt(3.0) / 2.0, 0.0],
        ],
        atol=1e-12,
    )
    npt.assert_allclose(degraded - observed, np.full((2, 3), 0.1), atol=1e-12)


def test_rotated_front_dtlz2_reference_is_nondominated_and_deterministic():
    from hldbea.metrics import build_reference_geometry
    from hldbea.problems import ProblemSpec
    from hldbea.scoring import pareto_nondominated_mask

    spec = ProblemSpec(
        id="rdtlz2-m3-a45",
        name="rdtlz2",
        n_obj=3,
        n_var=12,
        parameters={"angle_degrees": 45.0},
        transform={"kind": "identity"},
    )
    first = build_reference_geometry(spec)
    second = build_reference_geometry(spec)

    assert len(first.reference_set) >= 10
    assert np.all(pareto_nondominated_mask(first.reference_set))
    assert np.all(first.reference_set >= -1e-12)
    npt.assert_array_equal(first.reference_set, second.reference_set)
    assert first.geometry_hash == second.geometry_hash


@pytest.mark.parametrize(
    "parameters",
    [{}, {"angle_degrees": "30"}, {"angle_degrees": float("nan")}, {"angle_degrees": 95.0}],
)
def test_rotated_front_dtlz2_rejects_invalid_parameters(parameters):
    from hldbea.problems import ProblemSpec, build_problem

    spec = ProblemSpec(
        id="rdtlz2-invalid",
        name="rdtlz2",
        n_obj=3,
        n_var=12,
        parameters=parameters,
        transform={"kind": "identity"},
    )
    with pytest.raises(ValueError, match="angle_degrees"):
        build_problem(spec)


def test_dtlz7_reference_front_is_local_exact_and_disconnected():
    from hldbea.metrics import build_reference_geometry

    spec = _spec("dtlz7", 3, 22)
    first = build_reference_geometry(spec)
    second = build_reference_geometry(spec)
    front = np.asarray(first.reference_set)

    assert first.geometry_hash == second.geometry_hash
    assert front.shape[1] == 3 and len(front) > 1_000
    npt.assert_allclose(front[:, 2], 2 * (3 - np.sum(front[:, :2] / 2 * (1 + np.sin(3 * np.pi * front[:, :2])), axis=1)))
    for column in (0, 1):
        assert not np.any((front[:, column] > 0.26) & (front[:, column] < 0.63))
    assert front[:, 2].min() == pytest.approx(2.614, abs=0.01)
