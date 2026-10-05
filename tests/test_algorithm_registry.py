from dataclasses import replace
import json

import numpy as np
import pytest
from pymoo.optimize import minimize

from hldbea.evaluation import EvaluationLedger, LedgerEvaluator
from hldbea.problems import ProblemSpec, build_problem
from hldbea.run_spec import expand_manifest


EXPECTED = {
    "hldbea",
    "nsga2",
    "spea2",
    "nsga3",
    "rvea",
    "moead",
    "sms_emoa",
    "age_moea",
    "maoea_hap",
    "fdsea",
}


def _parameters(algorithm_id):
    common = {
        "crossover_probability": 0.9,
        "crossover_eta": 20,
        "mutation_probability": "inverse_n_var",
        "mutation_eta": 20,
    }
    if algorithm_id == "hldbea":
        return {
            **common,
            "k": 1.0,
            "lambda": 0.1,
            "cone_epsilon": 0.0,
            "neighborhood_mode": "axis",
            "score_aggregation": "sum",
            "selection_mode": "local",
            "objective_policy": "round_robin",
            "tau": 1,
            "local_search_max_iter": 2,
            "restart": False,
            "restart_theta_zero": 0.95,
            "restart_window": 20,
            "restart_delta_hv": 0.0001,
            "restart_fraction": 0.2,
        }
    if algorithm_id == "rvea":
        return {**common, "alpha": 2.0, "adapt_frequency": 0.1}
    if algorithm_id == "moead":
        return {**common, "n_neighbors": 5, "prob_neighbor_mating": 0.9}
    if algorithm_id == "fdsea":
        return {**common, "frequency_terms": 5, "operator": "ga"}
    return common


def _run_spec(algorithm_id, problem_name="dtlz2"):
    n_var = 12 if problem_name == "dtlz2" else 24
    document = {
        "schema_version": 1,
        "manifest_id": "adapter-smoke",
        "stage": "smoke",
        "seeds": [41],
        "problems": [
            {
                "id": f"{problem_name}-m3",
                "name": problem_name,
                "n_obj": 3,
                "n_var": n_var,
                "parameters": {},
                "transform": {"kind": "identity"},
            }
        ],
        "algorithms": [
            {
                "id": algorithm_id,
                "variant": "full" if algorithm_id == "hldbea" else "standard",
                "parameters": _parameters(algorithm_id),
            }
        ],
        "execution": {
            "population_size": 10,
            "evaluation_budget": 20,
            "checkpoints": [10, 20],
            "save_history": False,
        },
    }
    return expand_manifest(document)[0]


def test_registry_advertises_only_described_named_algorithms():
    from hldbea.registry import available_algorithms

    descriptors = available_algorithms()

    assert set(descriptors) == EXPECTED
    for key, descriptor in descriptors.items():
        assert descriptor.id == key
        assert descriptor.display_name
        assert descriptor.source
        assert descriptor.version
        assert descriptor.citation_key
        assert descriptor.exact_budget_support
        assert isinstance(descriptor.available, bool)
        if not descriptor.available:
            assert descriptor.unavailable_reason


def test_hldbea_registry_propagates_declared_neighbourhood_ablation():
    from hldbea.registry import build_algorithm

    spec = _run_spec("hldbea")
    parameters = dict(spec.algorithm_parameters)
    parameters.update(neighborhood_mode="box", score_aggregation="union")
    spec = replace(spec, algorithm_parameters=parameters)
    problem = build_problem(ProblemSpec.from_run_spec(spec))
    ledger = EvaluationLedger(spec.evaluation_budget)

    algorithm = build_algorithm(spec, problem, ledger)

    assert algorithm.neighborhood_mode == "box"
    assert algorithm.score_aggregation == "union"


@pytest.mark.parametrize("problem_name", ["dtlz2", "wfg2", "imop4"])
def test_every_available_adapter_runs_smoke_with_exact_shared_budget(problem_name):
    from hldbea.registry import available_algorithms, build_algorithm

    for algorithm_id, descriptor in available_algorithms().items():
        if not descriptor.available:
            continue
        spec = _run_spec(algorithm_id, problem_name)
        problem = build_problem(ProblemSpec.from_run_spec(spec))
        ledger = EvaluationLedger(spec.evaluation_budget)
        algorithm = build_algorithm(spec, problem, ledger)

        result = minimize(
            problem,
            algorithm,
            ("n_eval", spec.evaluation_budget),
            seed=spec.seed,
            verbose=False,
            copy_algorithm=False,
            save_history=False,
        )

        assert algorithm._declared_seed == spec.seed
        assert algorithm._evaluation_ledger is ledger
        assert isinstance(algorithm.evaluator, LedgerEvaluator)
        assert ledger.used == spec.evaluation_budget, algorithm_id
        assert ledger.solver == (0 if algorithm_id != "hldbea" else ledger.solver)
        assert result.F.ndim == 2
        assert result.F.shape[1] == spec.n_obj
        assert np.all(np.isfinite(result.F))


def test_reference_direction_adapters_match_declared_population():
    from hldbea.registry import build_algorithm

    for algorithm_id in ("nsga3", "rvea", "moead"):
        spec = _run_spec(algorithm_id)
        problem = build_problem(ProblemSpec.from_run_spec(spec))
        algorithm = build_algorithm(
            spec, problem, EvaluationLedger(spec.evaluation_budget)
        )
        directions = algorithm._reference_directions

        assert directions.shape == (spec.population_size, spec.n_obj)
        np.testing.assert_allclose(np.sum(directions, axis=1), 1.0)


def test_spea2_normalization_state_is_isolated_across_objective_dimensions():
    from hldbea.registry import build_algorithm

    three_objectives = _run_spec("spea2")
    two_objectives = replace(
        three_objectives,
        problem_id="dtlz2-m2",
        n_obj=2,
        n_var=11,
    )
    for spec in (three_objectives, two_objectives):
        problem = build_problem(ProblemSpec.from_run_spec(spec))
        ledger = EvaluationLedger(spec.evaluation_budget)
        algorithm = build_algorithm(spec, problem, ledger)
        minimize(
            problem,
            algorithm,
            ("n_eval", spec.evaluation_budget),
            seed=spec.seed,
            verbose=False,
            copy_algorithm=False,
        )

        assert ledger.used == spec.evaluation_budget
        assert algorithm.pop.get("F").shape[1] == spec.n_obj


def test_unknown_or_missing_algorithm_parameters_are_rejected():
    from hldbea.registry import build_algorithm

    spec = _run_spec("nsga2")
    problem = build_problem(ProblemSpec.from_run_spec(spec))
    with pytest.raises(ValueError, match="unknown"):
        build_algorithm(
            replace(spec, algorithm_parameters={**spec.algorithm_parameters, "secret": 1}),
            problem,
            EvaluationLedger(spec.evaluation_budget),
        )
    with pytest.raises(ValueError, match="missing"):
        build_algorithm(
            replace(spec, algorithm_parameters={}),
            problem,
            EvaluationLedger(spec.evaluation_budget),
        )


def test_hldbea_resolves_cone_schedule_once_for_algorithm_and_artifact(tmp_path):
    from hldbea.registry import build_algorithm
    from hldbea.runner import execute_run

    schedule = {
        "kind": "saturating",
        "limit": 0.15,
        "rate": 0.25,
        "origin": 3,
    }
    base = _run_spec("hldbea")
    spec = replace(
        base,
        problem_id="dtlz2-m5",
        n_obj=5,
        n_var=14,
        algorithm_parameters={
            **base.algorithm_parameters,
            "cone_epsilon": schedule,
        },
    )
    problem = build_problem(ProblemSpec.from_run_spec(spec))
    algorithm = build_algorithm(spec, problem, EvaluationLedger(spec.evaluation_budget))

    assert algorithm.cone_epsilon == pytest.approx(0.059020401043104985)
    assert algorithm._resolved_cone_epsilon == pytest.approx(0.059020401043104985)
    assert algorithm._declared_cone_schedule == schedule

    outcome = execute_run(spec, tmp_path)
    envelope = json.loads(
        (outcome.artifact_path / "metadata.json").read_text(encoding="utf-8")
    )
    metadata = envelope["run_metadata"]

    assert outcome.status != "failed"
    assert envelope["spec"]["algorithm"]["parameters"]["cone_epsilon"] == schedule
    assert metadata["declared_cone_schedule"] == schedule
    assert metadata["resolved_cone_epsilon"] == pytest.approx(
        0.059020401043104985
    )
    with np.load(outcome.artifact_path / "arrays.npz", allow_pickle=False) as archive:
        final_objectives = archive["F"]
    from hldbea.metrics import compute_score_diagnostics

    assert metadata["score_diagnostics"] == compute_score_diagnostics(
        final_objectives,
        k=1.0,
        cone_epsilon=0.059020401043104985,
    )


def test_hldbea_zero_local_search_depth_builds_disabled_ablation():
    from hldbea.registry import build_algorithm

    base = _run_spec("hldbea")
    spec = replace(
        base,
        algorithm_parameters={
            **base.algorithm_parameters,
            "local_search_max_iter": 0,
        },
    )
    problem = build_problem(ProblemSpec.from_run_spec(spec))

    algorithm = build_algorithm(spec, problem, EvaluationLedger(spec.evaluation_budget))

    assert algorithm.local_search_max_iter == 0


def test_hldbea_many_objective_hv_uses_fixed_reference_bounds():
    from hldbea.registry import build_algorithm

    base = _run_spec("hldbea")
    spec = replace(
        base,
        problem_id="dtlz2-m10",
        n_obj=10,
        n_var=19,
    )
    problem = build_problem(ProblemSpec.from_run_spec(spec))

    algorithm = build_algorithm(spec, problem, EvaluationLedger(spec.evaluation_budget))

    assert algorithm.metric_ideal_point.shape == (10,)
    assert algorithm.metric_ref_point.shape == (10,)
    assert np.all(np.isfinite(algorithm.metric_ideal_point))
    assert np.all(algorithm.metric_ref_point > algorithm.metric_ideal_point)


def test_optional_dependency_failure_is_machine_readable(monkeypatch):
    import hldbea.registry as registry

    monkeypatch.setattr(registry, "_module_available", lambda name: False)
    descriptor = registry.available_algorithms()["age_moea"]

    assert not descriptor.available
    assert descriptor.optional_dependency == "numba"
    assert "numba" in descriptor.unavailable_reason
    spec = _run_spec("age_moea")
    problem = build_problem(ProblemSpec.from_run_spec(spec))
    with pytest.raises(registry.AlgorithmUnavailable, match="numba"):
        registry.build_algorithm(
            spec, problem, EvaluationLedger(spec.evaluation_budget)
        )
