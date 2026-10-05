"""Construction helpers for algorithms admitted to the experiment registry."""

from __future__ import annotations

from functools import lru_cache
from copy import deepcopy
from typing import Any

import numpy as np
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.util.ref_dirs import get_reference_directions

from .evaluation import EvaluationLedger, LedgerEvaluator
from .cone_schedule import resolve_cone_epsilon
from .metrics import metric_reference_bounds
from .run_spec import RunSpec


COMMON_FIELDS = {
    "crossover_probability",
    "crossover_eta",
    "mutation_probability",
    "mutation_eta",
}
EXTRA_FIELDS = {
    "hldbea": {
        "k",
        "lambda",
        "cone_epsilon",
        "selection_mode",
        "objective_policy",
        "tau",
        "local_search_max_iter",
        "restart",
        "restart_theta_zero",
        "restart_window",
        "restart_delta_hv",
        "restart_fraction",
    },
    "rvea": {"alpha", "adapt_frequency"},
    "moead": {"n_neighbors", "prob_neighbor_mating"},
    "fdsea": {"frequency_terms", "operator"},
}
COMMON_OPTIONAL_FIELDS = {"mutation_scope"}
OPTIONAL_FIELDS = {
    "hldbea": {
        "neighborhood_mode",
        "score_aggregation",
        "local_search_acceptance",
        "candidate_mode",
        "refinement_duplicates",
    },
}
MUTATION_SCOPES = {"individual", "per_variable"}


def _validated_parameters(spec: RunSpec) -> dict[str, Any]:
    required = COMMON_FIELDS | EXTRA_FIELDS.get(spec.algorithm_id, set())
    allowed = (
        required
        | COMMON_OPTIONAL_FIELDS
        | OPTIONAL_FIELDS.get(spec.algorithm_id, set())
    )
    supplied = set(spec.algorithm_parameters)
    unknown = supplied - allowed
    missing = required - supplied
    if unknown:
        raise ValueError(f"unknown algorithm parameters: {sorted(unknown)}")
    if missing:
        raise ValueError(f"missing algorithm parameters: {sorted(missing)}")
    return dict(spec.algorithm_parameters)


def _number(value: Any, name: str, *, low: float | None = None) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not np.isfinite(value):
        raise ValueError(f"{name} must be finite and numeric")
    result = float(value)
    if low is not None and result < low:
        raise ValueError(f"{name} must be >= {low}")
    return result


def _positive_integer(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_integer(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _operators(parameters: dict[str, Any], problem):
    probability = _number(
        parameters["crossover_probability"], "crossover_probability", low=0.0
    )
    if probability > 1:
        raise ValueError("crossover_probability must be <= 1")
    mutation_probability = parameters["mutation_probability"]
    if mutation_probability == "inverse_n_var":
        mutation_probability = 1.0 / int(problem.n_var)
    else:
        mutation_probability = _number(
            mutation_probability, "mutation_probability", low=0.0
        )
        if mutation_probability > 1:
            raise ValueError("mutation_probability must be <= 1")
    crossover = SBX(
        eta=_number(parameters["crossover_eta"], "crossover_eta", low=0.0),
        prob=probability,
    )
    scope = parameters.get("mutation_scope", "individual")
    if scope not in MUTATION_SCOPES:
        raise ValueError(f"mutation_scope must be one of {sorted(MUTATION_SCOPES)}")
    eta = _number(parameters["mutation_eta"], "mutation_eta", low=0.0)
    if scope == "per_variable":
        # Every offspring enters PM and each coordinate mutates with the
        # declared probability, matching the PlatEMO ports.
        mutation = PM(eta=eta, prob=1.0, prob_var=mutation_probability)
    else:
        # Legacy v1/v2 semantics: pymoo gates whole individuals with this
        # probability and then each coordinate with min(0.5, 1/n_var).
        mutation = PM(eta=eta, prob=mutation_probability)
    return crossover, mutation


@lru_cache(maxsize=128)
def _cached_reference_directions(n_obj: int, population_size: int, seed: int):
    directions = get_reference_directions(
        "energy", n_obj, population_size, seed=seed
    )
    return np.asarray(directions, dtype=float)


def reference_directions(spec: RunSpec) -> np.ndarray:
    return _cached_reference_directions(
        spec.n_obj, spec.population_size, spec.seed
    ).copy()


def _common_kwargs(spec: RunSpec, ledger: EvaluationLedger, crossover, mutation):
    return {
        "sampling": FloatRandomSampling(),
        "crossover": crossover,
        "mutation": mutation,
        "evaluator": LedgerEvaluator(ledger),
        "seed": spec.seed,
    }


def build_pymoo_algorithm(spec: RunSpec, problem, ledger: EvaluationLedger):
    parameters = _validated_parameters(spec)
    crossover, mutation = _operators(parameters, problem)
    kwargs = _common_kwargs(spec, ledger, crossover, mutation)

    if spec.algorithm_id == "nsga2":
        from pymoo.algorithms.moo.nsga2 import NSGA2

        return NSGA2(pop_size=spec.population_size, **kwargs), None
    if spec.algorithm_id == "spea2":
        from pymoo.algorithms.moo.spea2 import SPEA2, SPEA2Survival

        # pymoo's default SPEA2Survival instance is created at import time and
        # retains normalization state, contaminating later runs with a
        # different objective dimension. Every immutable run needs its own.
        return SPEA2(
            pop_size=spec.population_size,
            survival=SPEA2Survival(normalize=True),
            **kwargs,
        ), None
    if spec.algorithm_id == "sms_emoa":
        from pymoo.algorithms.moo.sms import SMSEMOA

        return SMSEMOA(pop_size=spec.population_size, **kwargs), None
    if spec.algorithm_id == "age_moea":
        from pymoo.algorithms.moo.age import AGEMOEA

        return AGEMOEA(pop_size=spec.population_size, **kwargs), None

    directions = reference_directions(spec)
    if spec.algorithm_id == "nsga3":
        from pymoo.algorithms.moo.nsga3 import NSGA3

        algorithm = NSGA3(
            ref_dirs=directions,
            pop_size=spec.population_size,
            **kwargs,
        )
    elif spec.algorithm_id == "rvea":
        from pymoo.algorithms.moo.rvea import RVEA
        from pymoo.termination.max_eval import MaximumFunctionCallTermination
        from pymoo.termination.max_gen import MaximumGenerationTermination

        class ExactBudgetRVEA(RVEA):
            """Correct pymoo 0.6.1.3's off-by-one n_eval conversion."""

            def _setup(self, configured_problem, **configured_kwargs):
                if isinstance(self.termination, MaximumFunctionCallTermination):
                    offspring_generations = np.ceil(
                        (self.termination.n_max_evals - self.pop_size)
                        / self.n_offsprings
                    )
                    self.termination = MaximumGenerationTermination(
                        1 + max(0, int(offspring_generations))
                    )
                return super()._setup(configured_problem, **configured_kwargs)

        algorithm = ExactBudgetRVEA(
            ref_dirs=directions,
            pop_size=spec.population_size,
            alpha=_number(parameters["alpha"], "alpha", low=0.0),
            adapt_freq=_number(
                parameters["adapt_frequency"], "adapt_frequency", low=0.0
            ),
            **kwargs,
        )
    elif spec.algorithm_id == "moead":
        from pymoo.algorithms.moo.moead import MOEAD

        n_neighbors = _positive_integer(parameters["n_neighbors"], "n_neighbors")
        if n_neighbors > spec.population_size:
            raise ValueError("n_neighbors cannot exceed population_size")
        probability = _number(
            parameters["prob_neighbor_mating"], "prob_neighbor_mating", low=0.0
        )
        if probability > 1:
            raise ValueError("prob_neighbor_mating must be <= 1")
        algorithm = MOEAD(
            ref_dirs=directions,
            n_neighbors=n_neighbors,
            prob_neighbor_mating=probability,
            **kwargs,
        )
    else:
        raise KeyError(spec.algorithm_id)
    return algorithm, directions


def build_hldbea(spec: RunSpec, problem, ledger: EvaluationLedger):
    parameters = _validated_parameters(spec)
    crossover, mutation = _operators(parameters, problem)
    declared_cone_schedule = deepcopy(parameters["cone_epsilon"])
    resolved_cone_epsilon = resolve_cone_epsilon(
        declared_cone_schedule, spec.n_obj
    )
    metric_ideal_point, metric_ref_point = metric_reference_bounds(problem)

    from ibea import IBEA
    from ibeas.nibea.ibea_advance_4 import Advance

    algorithm = IBEA(
        Advance,
        pop_size=spec.population_size,
        sampling=FloatRandomSampling(),
        selection=Advance.selection(),
        crossover=crossover,
        mutation=mutation,
        K=_number(parameters["k"], "k", low=0.0),
        lmbda=_number(parameters["lambda"], "lambda", low=0.0),
        cone_epsilon=resolved_cone_epsilon,
        neighborhood_mode=parameters.get("neighborhood_mode", "axis"),
        score_aggregation=parameters.get("score_aggregation", "sum"),
        selection_mode=parameters["selection_mode"],
        objective_policy=parameters["objective_policy"],
        evaluation_ledger=ledger,
        metric_ideal_point=metric_ideal_point,
        metric_ref_point=metric_ref_point,
        use_restart=parameters["restart"],
        restart_theta_zero=_number(
            parameters["restart_theta_zero"], "restart_theta_zero", low=0.0
        ),
        restart_window=_positive_integer(
            parameters["restart_window"], "restart_window"
        ),
        restart_delta_hv=_number(
            parameters["restart_delta_hv"], "restart_delta_hv", low=0.0
        ),
        restart_fraction=_number(
            parameters["restart_fraction"], "restart_fraction", low=0.0
        ),
        local_search_max_iter=_nonnegative_integer(
            parameters["local_search_max_iter"], "local_search_max_iter"
        ),
        local_search_acceptance=parameters.get(
            "local_search_acceptance", "converged"
        ),
        candidate_mode=parameters.get("candidate_mode", "fitness"),
        refinement_duplicates=parameters.get("refinement_duplicates", "keep"),
        seed=spec.seed,
        save_history=spec.save_history,
    )
    if not isinstance(parameters["restart"], bool):
        raise ValueError("restart must be boolean")
    algorithm.fitm = False
    algorithm.sc_method = "0"
    algorithm.temp_pop = "0"
    algorithm.env_sel_method = "crowd"
    algorithm.sel_pop = "2"
    algorithm.exact_method = "epsilon"
    algorithm.tau = _positive_integer(parameters["tau"], "tau")
    algorithm._declared_cone_schedule = declared_cone_schedule
    algorithm._resolved_cone_epsilon = resolved_cone_epsilon
    algorithm.inf = "hybrid"
    Advance._infill(algorithm)
    return algorithm, None


def build_recent_algorithm(spec: RunSpec, problem, ledger: EvaluationLedger):
    """Build one audited post-2024 PlatEMO comparator."""

    parameters = _validated_parameters(spec)
    if parameters.get("mutation_scope", "per_variable") != "per_variable":
        raise ValueError("PlatEMO ports always mutate each variable independently")
    mutation_probability = parameters["mutation_probability"]
    if mutation_probability == "inverse_n_var":
        mutation_probability = 1.0 / int(problem.n_var)
    else:
        mutation_probability = _number(
            mutation_probability, "mutation_probability", low=0.0
        )
        if mutation_probability > 1.0:
            raise ValueError("mutation_probability must be <= 1")
    common = {
        "pop_size": spec.population_size,
        "crossover_probability": _number(
            parameters["crossover_probability"],
            "crossover_probability",
            low=0.0,
        ),
        "crossover_eta": _number(parameters["crossover_eta"], "crossover_eta", low=0.0),
        "mutation_probability": mutation_probability,
        "mutation_eta": _number(parameters["mutation_eta"], "mutation_eta", low=0.0),
        "evaluator": LedgerEvaluator(ledger),
        "seed": spec.seed,
    }
    if common["crossover_probability"] > 1.0:
        raise ValueError("crossover_probability must be <= 1")

    from .recent_algorithms import FDSEA, MaOEAHAP

    if spec.algorithm_id == "maoea_hap":
        return MaOEAHAP(**common), None
    if spec.algorithm_id == "fdsea":
        frequency_terms = _positive_integer(
            parameters["frequency_terms"], "frequency_terms"
        )
        operator = parameters["operator"]
        if operator not in {"ga", "de"}:
            raise ValueError("FDSEA operator must be 'ga' or 'de'")
        directions = reference_directions(spec)
        return (
            FDSEA(
                **common,
                ref_dirs=directions,
                frequency_terms=frequency_terms,
                operator=operator,
            ),
            directions,
        )
    raise KeyError(spec.algorithm_id)
