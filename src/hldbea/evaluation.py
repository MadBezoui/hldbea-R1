"""Exact, source-aware function-evaluation accounting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from pymoo.core.evaluator import Evaluator


class BudgetExhausted(RuntimeError):
    """Raised before an evaluation that would exceed the declared FE budget."""


@dataclass
class EvaluationLedger:
    max_evaluations: int
    used: int = 0
    evolutionary: int = 0
    solver: int = 0

    def __post_init__(self) -> None:
        if (
            not isinstance(self.max_evaluations, (int, np.integer))
            or isinstance(self.max_evaluations, bool)
            or self.max_evaluations <= 0
        ):
            raise ValueError("max_evaluations must be a positive integer")
        if min(self.used, self.evolutionary, self.solver) < 0:
            raise ValueError("evaluation counters must be non-negative")
        if self.used != self.evolutionary + self.solver:
            raise ValueError("used must equal evolutionary plus solver evaluations")
        if self.used > self.max_evaluations:
            raise ValueError("used evaluations cannot exceed max_evaluations")

    @property
    def remaining(self) -> int:
        return self.max_evaluations - self.used

    def evaluate(
        self,
        problem: Any,
        x: np.ndarray,
        *,
        source: str,
        return_values_of: Sequence[str],
    ) -> Any:
        """Charge the evaluated rows, then dispatch to ``problem.evaluate``."""

        values = np.asarray(x)
        if values.ndim == 1:
            n_evaluations = 1
        elif values.ndim == 2 and len(values) > 0:
            n_evaluations = len(values)
        else:
            raise ValueError("x must be a non-empty vector or matrix")
        self.charge(n_evaluations, source=source)
        return problem.evaluate(values, return_values_of=list(return_values_of))

    def charge(self, n_evaluations: int, *, source: str) -> None:
        """Reserve an auditable number of FEs before dispatching a call."""

        if source not in {"evolutionary", "solver"}:
            raise ValueError("source must be 'evolutionary' or 'solver'")
        if (
            not isinstance(n_evaluations, (int, np.integer))
            or isinstance(n_evaluations, bool)
            or n_evaluations <= 0
        ):
            raise ValueError("n_evaluations must be a positive integer")
        if n_evaluations > self.remaining:
            raise BudgetExhausted(
                f"evaluation requires {n_evaluations} FE with only {self.remaining} remaining"
            )

        self.used += n_evaluations
        if source == "evolutionary":
            self.evolutionary += n_evaluations
        else:
            self.solver += n_evaluations


class LedgerEvaluator(Evaluator):
    """Pymoo evaluator backed by the same ledger used by local search."""

    def __init__(self, ledger: EvaluationLedger, **kwargs: Any) -> None:
        if not isinstance(ledger, EvaluationLedger):
            raise TypeError("ledger must be an EvaluationLedger")
        super().__init__(**kwargs)
        self.ledger = ledger

    def _eval(self, problem: Any, pop: Any, evaluate_values_of: list[str], **kwargs: Any) -> None:
        self.ledger.charge(len(pop), source="evolutionary")
        super()._eval(problem, pop, evaluate_values_of, **kwargs)

    def eval(self, *args: Any, **kwargs: Any) -> Any:
        try:
            return super().eval(*args, **kwargs)
        finally:
            self.sync_from_ledger()

    def sync_from_ledger(self) -> None:
        self.n_eval = self.ledger.used
