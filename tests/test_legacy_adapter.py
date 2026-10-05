from types import SimpleNamespace
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import numpy.testing as npt
import pytest
from pymoo.core.population import Population
from pymoo.core.evaluator import Evaluator
from pymoo.core.problem import Problem

from hldbea.evaluation import EvaluationLedger, LedgerEvaluator
from hldbea.local_search import LocalSearchResult
from hldbea.scoring import compute_local_scores
from ibeas.nibea.util import Advance4Util


class ToyBiObjectiveProblem(Problem):
    def __init__(self):
        super().__init__(n_var=1, n_obj=2, xl=np.array([0.0]), xu=np.array([1.0]))

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = np.column_stack((x[:, 0], x[:, 0]))


def test_adapter_state_stores_raw_score_diagnostics_separately_from_fitness():
    """Catches the legacy adapter exposing only augmented `Fit` values."""
    objectives = np.array(
        [[1.0, 1.0], [0.9, 1.05], [0.0, 2.0], [2.0, 0.0]], dtype=float
    )
    population = Population.new("F", objectives)
    algorithm = SimpleNamespace(
        K=1.0,
        cone_epsilon=0.2,
        lmbda=0.1,
        neighborhood_mode="axis",
        score_aggregation="union",
        pop=population,
        sc_method="0",
    )
    expected = compute_local_scores(
        objectives,
        k=1.0,
        cone_epsilon=0.2,
        lmbda=0.1,
        neighborhood_mode="axis",
        score_aggregation="union",
    )

    fitness = Advance4Util.calc_scores(algorithm, pop=population)

    npt.assert_allclose(fitness, expected.fitness)
    npt.assert_array_equal(population.get("ScoreRaw"), expected.raw_scores)
    npt.assert_array_equal(population.get("ScoreAxis"), expected.per_axis_counts)
    npt.assert_allclose(population.get("FitDistance"), expected.normalized_distances)
    npt.assert_array_equal(population.get("GlobalND"), expected.nondominated)
    npt.assert_array_equal(population.get("FalseZero"), expected.false_zero)


def _survival_algorithm(mode):
    objectives = np.array([[1.0, 1.0], [0.0, 0.0], [2.0, 2.0]])
    population = Population.new(
        "X",
        np.array([[0.1], [0.2], [0.3]]),
        "F",
        objectives,
        "ScoreRaw",
        np.array([0, 1, 0]),
        "Fit",
        np.array([-0.1, 0.0, -0.2]),
    )
    return SimpleNamespace(
        pop=population,
        init_pop_size=1 if mode == "global" else 2,
        selection_mode=mode,
        eliminated_tab=None,
    )


def test_adapter_survival_uses_raw_zero_stratum_not_maximal_fitness_tie():
    """Catches the legacy `get_sorted(Fit)` partition masquerading as score zero."""
    from ibeas.nibea.ibea_advance_4 import _apply_environmental_selection

    algorithm = _survival_algorithm("local")

    _apply_environmental_selection(algorithm)

    assert {tuple(row) for row in algorithm.pop.get("F")} == {(1.0, 1.0), (2.0, 2.0)}
    npt.assert_array_equal(algorithm.pop.get("ScoreRaw"), [0, 0])


def test_adapter_global_selection_is_a_real_ablation():
    """Catches the global-selection variant still preserving the local zero stratum."""
    from ibeas.nibea.ibea_advance_4 import _apply_environmental_selection

    algorithm = _survival_algorithm("global")

    _apply_environmental_selection(algorithm)

    npt.assert_array_equal(algorithm.pop.get("F"), [[0.0, 0.0]])


def _restart_algorithm(with_reference=True):
    ledger = EvaluationLedger(50)
    problem = ToyBiObjectiveProblem()
    x = np.array([[0.2], [0.4], [0.6]])
    population = Population.new(
        "X",
        x,
        "F",
        np.column_stack((x[:, 0], x[:, 0])),
        "ScoreRaw",
        np.zeros(3, dtype=int),
        "Fit",
        np.array([-3.0, -2.0, -1.0]),
        "FitMin",
        np.array([-3.0, -2.0, -1.0]),
        "FitLife",
        np.zeros(3),
    )
    values = dict(
        pop=population,
        init_pop_size=3,
        problem=problem,
        evaluator=LedgerEvaluator(ledger),
        evaluation_ledger=ledger,
        random_state=np.random.default_rng(7),
        n_gen=1,
        restart_theta_zero=1.0,
        restart_window=2,
        restart_delta_hv=1.0,
        restart_fraction=1.0 / 3.0,
        restart_events=[],
    )
    if with_reference:
        values["metric_ref_point"] = np.array([2.0, 2.0])
    return SimpleNamespace(**values)


def test_adapter_restart_uses_raw_scores_and_explicit_metric_reference():
    """Catches both `Fit >= 0` restart gating and implicit nadir reference points."""
    from ibeas.nibea.ibea_advance_4 import _check_and_apply_restart

    algorithm = _restart_algorithm(with_reference=True)

    _check_and_apply_restart(algorithm)
    algorithm.n_gen = 2
    _check_and_apply_restart(algorithm)

    assert algorithm.restart_events[-1]["status"] == "triggered"
    assert algorithm.restart_events[-1]["zero_fraction"] == 1.0
    assert algorithm.restart_events[-1]["replace_indices"] == [0]
    assert algorithm.restart_events[-1]["replacement_count"] == 1
    assert algorithm.restart_events[-1]["evaluation"] == 0
    assert algorithm.restart_events[-1]["evaluation_after"] == 1
    assert np.isfinite(algorithm.restart_events[-1]["hv"])
    assert len(algorithm.pop) == 3


def test_adapter_restart_records_missing_reference_instead_of_guessing():
    """Catches silently falling back to a problem-dependent nadir point."""
    from ibeas.nibea.ibea_advance_4 import _check_and_apply_restart

    algorithm = _restart_algorithm(with_reference=False)
    before = algorithm.pop.get("X").copy()

    _check_and_apply_restart(algorithm)

    assert algorithm.restart_events[-1]["status"] == "skipped_missing_reference"
    npt.assert_array_equal(algorithm.pop.get("X"), before)


def test_adapter_restart_skips_cleanly_when_shared_budget_is_exhausted():
    from hldbea.evaluation import EvaluationLedger, LedgerEvaluator
    from ibeas.nibea.ibea_advance_4 import _check_and_apply_restart

    algorithm = _restart_algorithm(with_reference=True)
    ledger = EvaluationLedger(1, used=1, evolutionary=1)
    algorithm.evaluation_ledger = ledger
    algorithm.evaluator = LedgerEvaluator(ledger)
    before = algorithm.pop.get("X").copy()

    _check_and_apply_restart(algorithm)
    algorithm.n_gen = 2
    _check_and_apply_restart(algorithm)

    assert algorithm.restart_events[-1]["status"] == "skipped_budget"
    npt.assert_array_equal(algorithm.pop.get("X"), before)


def _local_search_algorithm(policy, generation=3):
    return SimpleNamespace(
        problem=ToyBiObjectiveProblem(),
        objective_policy=policy,
        n_gen=generation,
        random_state=np.random.default_rng(11),
        evaluation_ledger=EvaluationLedger(50),
        metric_ref_point=np.array([2.0, 2.0]),
        local_search_max_iter=2,
        local_search_events=[],
    )


def test_local_search_adapter_uses_declared_policy_ledger_and_cached_objectives(
    monkeypatch,
):
    """Catches random objective choice, hidden FEs, and duplicate elite evaluation."""
    import ibeas.nibea.custominfill as module

    algorithm = _local_search_algorithm("round_robin", generation=3)
    elite = Population.new(
        "X",
        np.array([[0.6]]),
        "F",
        np.array([[0.6, 0.6]]),
        "ScoreAxis",
        np.array([[4, 1]]),
    )
    calls = []

    def fake_solver(problem, x0, **kwargs):
        calls.append((problem, x0.copy(), kwargs))
        assert kwargs["ledger"] is algorithm.evaluation_ledger
        algorithm.evaluation_ledger.charge(7, source="solver")
        return LocalSearchResult(
            x=np.array([0.5]),
            f=np.array([0.5, 0.5]),
            success=True,
            feasible=True,
            accepted=True,
            evaluations=7,
            message="accepted fixture",
            objective_index=kwargs["main_obj_index"],
        )

    monkeypatch.setattr(module, "solve_epsilon_constraint", fake_solver)

    refined = module._refine_elite(algorithm, elite)

    assert calls[0][2]["main_obj_index"] == 1
    npt.assert_array_equal(calls[0][2]["initial_f"], [0.6, 0.6])
    npt.assert_array_equal(refined.get("X"), [[0.5]])
    npt.assert_array_equal(refined.get("F"), [[0.5, 0.5]])
    assert algorithm.local_search_events == [
        {
            "generation": 3,
            "elite_index": 0,
            "objective_index": 1,
            "success": True,
            "feasible": True,
            "accepted": True,
            "evaluations": 7,
            "evaluation_before": 0,
            "evaluation_after": 7,
            "hv_gain": pytest.approx(0.29),
            "gain_per_evaluation": pytest.approx(0.29 / 7.0),
            "message": "accepted fixture",
            "x_before": [0.6],
            "x_after": [0.5],
            "f_before": [0.6, 0.6],
            "f_after": [0.5, 0.5],
        }
    ]


def test_local_search_adapter_adaptive_policy_uses_per_axis_counts(monkeypatch):
    import ibeas.nibea.custominfill as module

    algorithm = _local_search_algorithm("adaptive", generation=0)
    elite = Population.new(
        "X",
        np.array([[0.4]]),
        "F",
        np.array([[0.4, 0.4]]),
        "ScoreAxis",
        np.array([[1, 5]]),
    )

    def fake_solver(problem, x0, **kwargs):
        algorithm.evaluation_ledger.charge(2, source="solver")
        return LocalSearchResult(
            x=x0.copy(),
            f=np.array([0.4, 0.4]),
            success=False,
            feasible=True,
            accepted=False,
            evaluations=2,
            message="rejected fixture",
            objective_index=kwargs["main_obj_index"],
        )

    monkeypatch.setattr(module, "solve_epsilon_constraint", fake_solver)

    module._refine_elite(algorithm, elite)

    assert algorithm.local_search_events[0]["objective_index"] == 1
    assert not algorithm.local_search_events[0]["accepted"]
    assert algorithm.local_search_events[0]["evaluation_before"] == 0
    assert algorithm.local_search_events[0]["evaluation_after"] == 2
    assert algorithm.local_search_events[0]["hv_gain"] == 0.0
    assert algorithm.local_search_events[0]["gain_per_evaluation"] == 0.0


def test_local_search_adapter_zero_evaluation_rejection_has_zero_efficiency(
    monkeypatch,
):
    import ibeas.nibea.custominfill as module

    algorithm = _local_search_algorithm("round_robin", generation=0)
    elite = Population.new(
        "X",
        np.array([[0.4]]),
        "F",
        np.array([[0.4, 0.4]]),
        "ScoreAxis",
        np.array([[1, 1]]),
    )

    def fake_solver(problem, x0, **kwargs):
        return LocalSearchResult(
            x=x0.copy(),
            f=np.array([0.4, 0.4]),
            success=False,
            feasible=False,
            accepted=False,
            evaluations=0,
            message="budget boundary fixture",
            objective_index=kwargs["main_obj_index"],
        )

    monkeypatch.setattr(module, "solve_epsilon_constraint", fake_solver)

    module._refine_elite(algorithm, elite)

    event = algorithm.local_search_events[0]
    assert event["evaluation_before"] == event["evaluation_after"] == 0
    assert event["hv_gain"] == 0.0
    assert event["gain_per_evaluation"] == 0.0


def test_zero_local_search_depth_disables_solver_calls():
    import ibeas.nibea.custominfill as module

    algorithm = _local_search_algorithm("round_robin", generation=0)
    algorithm.local_search_max_iter = 0
    elite = Population.new(
        "X",
        np.array([[0.4]]),
        "F",
        np.array([[0.4, 0.4]]),
        "ScoreAxis",
        np.array([[1, 1]]),
    )

    refined = module._refine_elite(algorithm, elite)

    npt.assert_array_equal(refined.get("X"), elite.get("X"))
    npt.assert_array_equal(refined.get("F"), elite.get("F"))
    assert algorithm.evaluation_ledger.used == 0
    assert algorithm.local_search_events == []


def test_ibea_exposes_all_confirmatory_controls_and_budget_ledger():
    from ibea import IBEA

    algorithm = IBEA(
        SimpleNamespace(),
        pop_size=4,
        cone_epsilon=0.15,
        selection_mode="global",
        objective_policy="adaptive",
        evaluation_budget=123,
        metric_ref_point=np.array([1.1, 1.1]),
        use_restart=True,
        restart_theta_zero=0.9,
        restart_window=7,
        restart_delta_hv=1e-5,
        restart_fraction=0.25,
        local_search_max_iter=3,
    )

    assert algorithm.cone_epsilon == 0.15
    assert algorithm.selection_mode == "global"
    assert algorithm.objective_policy == "adaptive"
    assert algorithm.evaluation_ledger.max_evaluations == 123
    npt.assert_array_equal(algorithm.metric_ref_point, [1.1, 1.1])
    assert algorithm.use_restart
    assert algorithm.restart_window == 7
    assert algorithm.local_search_max_iter == 3
    assert algorithm.restart_events == []
    assert algorithm.local_search_events == []


def test_pymoo_evaluator_charges_evolutionary_rows_to_the_shared_budget():
    from hldbea.evaluation import BudgetExhausted, EvaluationLedger, LedgerEvaluator

    ledger = EvaluationLedger(3)
    evaluator = LedgerEvaluator(ledger)
    problem = ToyBiObjectiveProblem()
    first = Population.new("X", np.array([[0.1], [0.2]]))
    evaluator.eval(problem, first)

    assert evaluator.n_eval == 2
    assert ledger.used == 2
    assert ledger.evolutionary == 2

    second = Population.new("X", np.array([[0.3], [0.4]]))
    with np.testing.assert_raises(BudgetExhausted):
        evaluator.eval(problem, second)

    assert ledger.used == 2
    assert evaluator.n_eval == 2


def test_many_objective_reference_points_are_finite_and_dimensionally_valid():
    from hldbea.metrics import metric_reference_point
    from pymoo.problems import get_problem

    for n_obj in (5, 10, 15):
        problem = get_problem("dtlz2", n_obj + 9, n_obj)
        reference = metric_reference_point(problem)
        assert reference.shape == (n_obj,)
        assert np.all(np.isfinite(reference))
        assert np.all(reference > 1.0)


def test_many_objective_baseline_reference_directions_are_defined():
    from ibeas.nibea.algo_execution import TestAlgorithms
    from pymoo.problems import get_problem

    for n_obj in (5, 10, 15):
        problem = get_problem("dtlz2", n_obj + 9, n_obj)
        directions = TestAlgorithms.get_ref_dirs(problem)
        assert directions.ndim == 2
        assert directions.shape[1] == n_obj
        npt.assert_allclose(np.sum(directions, axis=1), 1.0)


def test_many_objective_cli_smoke_run_completes_headlessly():
    root = Path(__file__).resolve().parents[1]
    environment = {**os.environ, "MPLBACKEND": "Agg", "PYTHONDONTWRITEBYTECODE": "1"}
    completed = subprocess.run(
        [
            sys.executable,
            "nibea_test.py",
            "--test-single",
            "--pb",
            "DTLZ2",
            "--n_var",
            "14",
            "--n_obj",
            "5",
            "--n_gen",
            "2",
            "--n_pop",
            "10",
        ],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "HLDBEA : Hv" in completed.stdout
