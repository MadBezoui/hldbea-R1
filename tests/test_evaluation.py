import numpy as np
import numpy.testing as npt
import pytest


class CountingProblem:
    def __init__(self, fail=False):
        self.evaluations = 0
        self.fail = fail

    def evaluate(self, x, return_values_of=None):
        values = np.asarray(x, dtype=float)
        rows = 1 if values.ndim == 1 else len(values)
        self.evaluations += rows
        if self.fail:
            raise RuntimeError("toy evaluation failed")
        result = np.sum(values**2, axis=-1, keepdims=True)
        return result[0] if values.ndim == 1 else result


def _evaluation_api():
    from hldbea.evaluation import BudgetExhausted, EvaluationLedger

    return BudgetExhausted, EvaluationLedger


def test_evaluation_ledger_counts_vector_and_matrix_rows_by_source():
    """Catches treating a batched objective call as a single function evaluation."""
    _, EvaluationLedger = _evaluation_api()
    ledger = EvaluationLedger(max_evaluations=4)
    problem = CountingProblem()

    vector_result = ledger.evaluate(
        problem, np.array([1.0, 2.0]), source="evolutionary", return_values_of=["F"]
    )
    matrix_result = ledger.evaluate(
        problem,
        np.array([[1.0, 0.0], [0.0, 1.0]]),
        source="solver",
        return_values_of=["F"],
    )

    npt.assert_allclose(vector_result, [5.0])
    npt.assert_allclose(matrix_result, [[1.0], [1.0]])
    assert ledger.used == 3
    assert ledger.evolutionary == 1
    assert ledger.solver == 2
    assert problem.evaluations == 3


def test_evaluation_ledger_stops_before_budget_overspend():
    """Catches dispatching a solver evaluation after the FE budget is exhausted."""
    BudgetExhausted, EvaluationLedger = _evaluation_api()
    ledger = EvaluationLedger(max_evaluations=2)
    problem = CountingProblem()
    ledger.evaluate(
        problem,
        np.array([[1.0, 0.0], [0.0, 1.0]]),
        source="solver",
        return_values_of=["F"],
    )

    with pytest.raises(BudgetExhausted):
        ledger.evaluate(
            problem, np.array([2.0, 2.0]), source="solver", return_values_of=["F"]
        )

    assert ledger.used == 2
    assert ledger.solver == 2
    assert problem.evaluations == 2


def test_evaluation_ledger_charges_a_dispatched_failure():
    """Catches hiding failed black-box calls from the declared FE total."""
    _, EvaluationLedger = _evaluation_api()
    ledger = EvaluationLedger(max_evaluations=2)
    problem = CountingProblem(fail=True)

    with pytest.raises(RuntimeError, match="toy evaluation failed"):
        ledger.evaluate(
            problem, np.array([1.0, 2.0]), source="solver", return_values_of=["F"]
        )

    assert ledger.used == 1
    assert ledger.solver == 1
    assert problem.evaluations == 1


def test_evaluation_ledger_rejects_unknown_source_before_dispatch():
    """Catches unclassified evaluations that cannot be audited later."""
    _, EvaluationLedger = _evaluation_api()
    ledger = EvaluationLedger(max_evaluations=2)
    problem = CountingProblem()

    with pytest.raises(ValueError):
        ledger.evaluate(
            problem, np.array([1.0, 2.0]), source="hidden", return_values_of=["F"]
        )

    assert ledger.used == 0
    assert problem.evaluations == 0


@pytest.mark.parametrize("max_evaluations", [0, -1, 1.5, True])
def test_evaluation_ledger_rejects_invalid_budget(max_evaluations):
    """Catches manifests with a non-positive or fractional FE budget."""
    _, EvaluationLedger = _evaluation_api()
    with pytest.raises(ValueError):
        EvaluationLedger(max_evaluations=max_evaluations)
