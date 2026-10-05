"""Budget-aware deterministic refinement for HLDBEA."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .evaluation import BudgetExhausted, EvaluationLedger


@dataclass(frozen=True)
class LocalSearchResult:
    x: np.ndarray
    f: np.ndarray
    success: bool
    feasible: bool
    accepted: bool
    evaluations: int
    message: str
    objective_index: int


def select_objective(
    policy: str,
    *,
    n_obj: int,
    generation: int,
    per_axis_counts: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> int:
    """Choose the epsilon-constraint main objective under a declared policy."""

    if (
        not isinstance(n_obj, (int, np.integer))
        or isinstance(n_obj, bool)
        or n_obj < 2
    ):
        raise ValueError("n_obj must be an integer of at least two")
    if (
        not isinstance(generation, (int, np.integer))
        or isinstance(generation, bool)
        or generation < 0
    ):
        raise ValueError("generation must be a non-negative integer")
    if policy not in {"round_robin", "adaptive", "random"}:
        raise ValueError("unknown objective-selection policy")

    if policy == "round_robin":
        return int(generation % n_obj)
    if policy == "adaptive":
        counts = np.asarray(per_axis_counts)
        if (
            counts.ndim != 1
            or len(counts) != n_obj
            or not np.all(np.isfinite(counts))
            or np.any(counts < 0)
        ):
            raise ValueError("per_axis_counts must align with the objectives")
        return int(np.argmax(counts))
    if not isinstance(rng, np.random.Generator):
        raise ValueError("random policy requires a numpy.random.Generator")
    return int(rng.integers(0, n_obj))


def _vector(values: np.ndarray, expected: int, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float).reshape(-1)
    if len(result) != expected or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain {expected} finite values")
    return result


LOCAL_SEARCH_ACCEPTANCE = frozenset({"converged", "feasible_improvement"})


def solve_epsilon_constraint(
    problem,
    x0: np.ndarray,
    *,
    main_obj_index: int,
    max_iter: int,
    ledger: EvaluationLedger,
    initial_f: np.ndarray | None = None,
    tolerance: float = 1e-8,
    acceptance: str = "converged",
) -> LocalSearchResult:
    """Run SLSQP while charging every objective and constraint evaluation."""

    start = np.asarray(x0, dtype=float)
    n_obj = int(problem.n_obj)
    if start.ndim != 1 or not np.all(np.isfinite(start)):
        raise ValueError("x0 must be a finite decision vector")
    if main_obj_index < 0 or main_obj_index >= n_obj:
        raise ValueError("main_obj_index is outside the objective range")
    if not isinstance(max_iter, (int, np.integer)) or max_iter <= 0:
        raise ValueError("max_iter must be a positive integer")
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and non-negative")
    if acceptance not in LOCAL_SEARCH_ACCEPTANCE:
        raise ValueError(f"acceptance must be one of {sorted(LOCAL_SEARCH_ACCEPTANCE)}")

    lower = _vector(problem.xl, len(start), "problem.xl")
    upper = _vector(problem.xu, len(start), "problem.xu")
    if np.any(lower > upper) or np.any(start < lower) or np.any(start > upper):
        raise ValueError("x0 and problem bounds are inconsistent")

    used_before = ledger.used
    if initial_f is None:
        try:
            f0 = _vector(
                ledger.evaluate(
                    problem, start, source="solver", return_values_of=["F"]
                ),
                n_obj,
                "initial objectives",
            )
        except BudgetExhausted as exc:
            return LocalSearchResult(
                x=start.copy(),
                f=np.full(n_obj, np.nan),
                success=False,
                feasible=False,
                accepted=False,
                evaluations=ledger.used - used_before,
                message=f"budget exhausted: {exc}",
                objective_index=main_obj_index,
            )
    else:
        f0 = _vector(initial_f, n_obj, "initial_f")

    other_objectives = np.arange(n_obj) != main_obj_index

    def objective(x: np.ndarray) -> float:
        f = _vector(
            ledger.evaluate(problem, x, source="solver", return_values_of=["F"]),
            n_obj,
            "solver objectives",
        )
        return float(f[main_obj_index])

    def epsilon_constraint(x: np.ndarray) -> np.ndarray:
        f = _vector(
            ledger.evaluate(problem, x, source="solver", return_values_of=["F"]),
            n_obj,
            "epsilon objectives",
        )
        return f0[other_objectives] - f[other_objectives]

    constraints: list[dict[str, object]] = [
        {"type": "ineq", "fun": epsilon_constraint}
    ]
    if int(getattr(problem, "n_ieq_constr", 0)) > 0:

        def original_inequalities(x: np.ndarray) -> np.ndarray:
            values = ledger.evaluate(
                problem, x, source="solver", return_values_of=["G"]
            )
            return -np.asarray(values, dtype=float).reshape(-1)

        constraints.append({"type": "ineq", "fun": original_inequalities})
    if int(getattr(problem, "n_eq_constr", 0)) > 0:

        def original_equalities(x: np.ndarray) -> np.ndarray:
            values = ledger.evaluate(
                problem, x, source="solver", return_values_of=["H"]
            )
            return np.asarray(values, dtype=float).reshape(-1)

        constraints.append({"type": "eq", "fun": original_equalities})

    try:
        from scipy.optimize import minimize

        optimized = minimize(
            objective,
            start,
            method="SLSQP",
            bounds=np.column_stack((lower, upper)),
            constraints=constraints,
            options={"maxiter": int(max_iter), "disp": False},
        )
        candidate = np.asarray(optimized.x, dtype=float)
        candidate_f = _vector(
            ledger.evaluate(
                problem, candidate, source="solver", return_values_of=["F"]
            ),
            n_obj,
            "candidate objectives",
        )
        feasible = bool(
            np.all(candidate >= lower - tolerance)
            and np.all(candidate <= upper + tolerance)
            and np.all(candidate_f[other_objectives] <= f0[other_objectives] + tolerance)
        )
        if int(getattr(problem, "n_ieq_constr", 0)) > 0:
            candidate_g = np.asarray(
                ledger.evaluate(
                    problem, candidate, source="solver", return_values_of=["G"]
                ),
                dtype=float,
            ).reshape(-1)
            feasible = feasible and bool(np.all(candidate_g <= tolerance))
        if int(getattr(problem, "n_eq_constr", 0)) > 0:
            candidate_h = np.asarray(
                ledger.evaluate(
                    problem, candidate, source="solver", return_values_of=["H"]
                ),
                dtype=float,
            ).reshape(-1)
            feasible = feasible and bool(np.all(np.abs(candidate_h) <= tolerance))
        success = bool(optimized.success)
        if acceptance == "converged":
            accepted = bool(
                success
                and feasible
                and candidate_f[main_obj_index] <= f0[main_obj_index] + tolerance
            )
        else:
            # Algorithm 2: keep the k_max-iterate whenever it satisfies the
            # epsilon constraints and strictly improves the main objective,
            # whether or not SLSQP declared convergence.
            accepted = bool(
                feasible
                and np.all(np.isfinite(candidate_f))
                and candidate_f[main_obj_index] < f0[main_obj_index] - tolerance
            )
        return LocalSearchResult(
            x=candidate.copy() if accepted else start.copy(),
            f=candidate_f.copy() if accepted else f0.copy(),
            success=success,
            feasible=feasible,
            accepted=accepted,
            evaluations=ledger.used - used_before,
            message=str(optimized.message),
            objective_index=main_obj_index,
        )
    except BudgetExhausted as exc:
        message = f"budget exhausted: {exc}"
    except Exception as exc:
        message = f"solver failed: {exc}"

    return LocalSearchResult(
        x=start.copy(),
        f=f0.copy(),
        success=False,
        feasible=False,
        accepted=False,
        evaluations=ledger.used - used_before,
        message=message,
        objective_index=main_obj_index,
    )
