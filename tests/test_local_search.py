import numpy as np
import numpy.testing as npt
import pytest


def select_objective(*args, **kwargs):
    from hldbea.local_search import select_objective as implementation

    return implementation(*args, **kwargs)


def solve_epsilon_constraint(*args, **kwargs):
    from hldbea.local_search import solve_epsilon_constraint as implementation

    return implementation(*args, **kwargs)


class QuadraticProblem:
    n_obj = 2
    n_eq_constr = 0
    xl = np.array([-2.0])
    xu = np.array([2.0])

    def __init__(self, *, impossible_constraint=False, fail_after=None):
        self.n_ieq_constr = 1 if impossible_constraint else 0
        self.impossible_constraint = impossible_constraint
        self.fail_after = fail_after
        self.evaluations = 0

    def evaluate(self, x, return_values_of=None):
        values = np.asarray(x, dtype=float)
        rows = 1 if values.ndim == 1 else len(values)
        self.evaluations += rows
        if self.fail_after is not None and self.evaluations > self.fail_after:
            raise RuntimeError("toy solver evaluation failed")
        if return_values_of == ["F"]:
            point = values if values.ndim == 1 else values[:, 0]
            scalar = point[0] if values.ndim == 1 else point
            result = np.stack((scalar**2, scalar**2), axis=-1)
            return result
        if return_values_of == ["G"]:
            if values.ndim == 1:
                return np.array([1.0])
            return np.ones((len(values), 1))
        raise ValueError(f"unsupported return_values_of={return_values_of}")


def test_round_robin_objective_uses_generation_modulo_objectives():
    """Catches the legacy random choice under the declared round-robin policy."""
    assert select_objective("round_robin", n_obj=3, generation=0) == 0
    assert select_objective("round_robin", n_obj=3, generation=4) == 1


def test_adaptive_objective_uses_largest_axis_count_with_stable_tie():
    """Catches selecting a sparse axis or resolving equal violations randomly."""
    selected = select_objective(
        "adaptive", n_obj=3, generation=7, per_axis_counts=np.array([2, 5, 5])
    )

    assert selected == 1


def test_random_objective_is_reproducible_from_supplied_generator():
    """Catches use of NumPy's unseeded global random state."""
    left_rng = np.random.default_rng(42)
    right_rng = np.random.default_rng(42)

    left = [
        select_objective("random", n_obj=5, generation=i, rng=left_rng)
        for i in range(8)
    ]
    right = [
        select_objective("random", n_obj=5, generation=i, rng=right_rng)
        for i in range(8)
    ]

    assert left == right
    assert all(0 <= value < 5 for value in left)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"policy": "unknown", "n_obj": 3, "generation": 0},
        {"policy": "round_robin", "n_obj": 1, "generation": 0},
        {"policy": "round_robin", "n_obj": 3, "generation": -1},
        {
            "policy": "adaptive",
            "n_obj": 3,
            "generation": 0,
            "per_axis_counts": np.array([1, 2]),
        },
        {"policy": "random", "n_obj": 3, "generation": 0, "rng": None},
    ],
)
def test_objective_policy_rejects_invalid_contract(kwargs):
    """Catches undeclared or under-specified objective-selection behavior."""
    with pytest.raises(ValueError):
        select_objective(**kwargs)


def test_epsilon_search_accepts_a_feasible_improvement_and_counts_every_call():
    """Catches an uncharged SLSQP path or an accepted worsening candidate."""
    from hldbea.evaluation import EvaluationLedger

    problem = QuadraticProblem()
    ledger = EvaluationLedger(max_evaluations=100)

    result = solve_epsilon_constraint(
        problem,
        np.array([1.0]),
        main_obj_index=0,
        max_iter=20,
        ledger=ledger,
    )

    assert result.success
    assert result.feasible
    assert result.accepted
    assert result.objective_index == 0
    assert result.f[0] < 1e-8
    assert abs(result.x[0]) < 1e-4
    assert result.evaluations == ledger.used == problem.evaluations


def test_epsilon_search_rejects_an_infeasible_candidate_and_retains_start():
    """Catches accepting a solver point that violates original constraints."""
    from hldbea.evaluation import EvaluationLedger

    problem = QuadraticProblem(impossible_constraint=True)
    ledger = EvaluationLedger(max_evaluations=100)
    start = np.array([1.0])

    result = solve_epsilon_constraint(
        problem,
        start,
        main_obj_index=0,
        max_iter=3,
        ledger=ledger,
    )

    assert not result.accepted
    assert not result.feasible
    npt.assert_array_equal(result.x, start)
    npt.assert_array_equal(result.f, [1.0, 1.0])
    assert result.evaluations == ledger.used == problem.evaluations


def test_epsilon_search_catches_problem_exception_and_retains_start():
    """Catches a black-box exception escaping without an auditable result."""
    from hldbea.evaluation import EvaluationLedger

    problem = QuadraticProblem(fail_after=1)
    ledger = EvaluationLedger(max_evaluations=100)
    start = np.array([1.0])

    result = solve_epsilon_constraint(
        problem,
        start,
        main_obj_index=0,
        max_iter=5,
        ledger=ledger,
    )

    assert not result.success
    assert not result.accepted
    assert "toy solver evaluation failed" in result.message
    npt.assert_array_equal(result.x, start)
    npt.assert_array_equal(result.f, [1.0, 1.0])
    assert result.evaluations == ledger.used == problem.evaluations == 2


def test_epsilon_search_stops_at_last_evaluation_without_overspending():
    """Catches SLSQP dispatch after the final available FE."""
    from hldbea.evaluation import EvaluationLedger

    problem = QuadraticProblem()
    ledger = EvaluationLedger(max_evaluations=1)
    start = np.array([1.0])

    result = solve_epsilon_constraint(
        problem,
        start,
        main_obj_index=0,
        max_iter=5,
        ledger=ledger,
    )

    assert not result.success
    assert not result.accepted
    assert "budget" in result.message.lower()
    npt.assert_array_equal(result.x, start)
    npt.assert_array_equal(result.f, [1.0, 1.0])
    assert result.evaluations == ledger.used == problem.evaluations == 1


@pytest.mark.parametrize(
    ("acceptance", "expected"), [("converged", False), ("feasible_improvement", True)]
)
def test_epsilon_search_acceptance_of_iteration_limited_improvement(acceptance, expected):
    """Algorithm 2 keeps a feasible k_max-iterate even without SLSQP convergence."""
    from hldbea.evaluation import EvaluationLedger

    problem = QuadraticProblem()
    ledger = EvaluationLedger(max_evaluations=100)
    start = np.array([1.5])

    result = solve_epsilon_constraint(
        problem,
        start,
        main_obj_index=0,
        max_iter=2,
        ledger=ledger,
        acceptance=acceptance,
    )

    assert not result.success
    assert result.feasible
    assert result.accepted is expected
    if expected:
        assert result.f[0] < 1.5**2
    else:
        npt.assert_array_equal(result.x, start)
    assert result.evaluations == ledger.used == problem.evaluations


def test_epsilon_search_rejects_unknown_acceptance_rule():
    from hldbea.evaluation import EvaluationLedger

    with pytest.raises(ValueError, match="acceptance"):
        solve_epsilon_constraint(
            QuadraticProblem(),
            np.array([1.0]),
            main_obj_index=0,
            max_iter=1,
            ledger=EvaluationLedger(max_evaluations=10),
            acceptance="always",
        )
