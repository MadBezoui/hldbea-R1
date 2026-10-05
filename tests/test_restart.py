import numpy as np
import numpy.testing as npt
import pytest


def _restart_api():
    from hldbea.restart import (
        RestartPolicy,
        RestartState,
        update_restart_state,
    )

    return RestartPolicy, RestartState, update_restart_state


def _step(state, policy, hv, raw_scores=None, fitness=None, generation=0):
    _, _, update_restart_state = _restart_api()
    if raw_scores is None:
        raw_scores = np.zeros(5, dtype=int)
    if fitness is None:
        fitness = np.arange(len(raw_scores), dtype=float)
    return update_restart_state(
        state,
        policy,
        raw_scores=np.asarray(raw_scores),
        fitness=np.asarray(fitness, dtype=float),
        hv=hv,
        generation=generation,
    )


def test_restart_waits_for_a_full_history_window():
    """Catches restart decisions made from too little convergence evidence."""
    RestartPolicy, RestartState, _ = _restart_api()
    policy = RestartPolicy(theta_zero=0.8, window=3, delta_hv=0.01, fraction=0.4)
    state = RestartState()

    decision = _step(state, policy, 1.0, generation=1)

    assert not decision.triggered
    assert decision.hv_delta is None
    assert state.hv_history == [1.0]


def test_restart_requires_the_raw_zero_fraction_threshold():
    """Catches using augmented fitness signs as a proxy for raw-score zero."""
    RestartPolicy, RestartState, _ = _restart_api()
    policy = RestartPolicy(theta_zero=0.75, window=3, delta_hv=0.01, fraction=0.5)
    state = RestartState()

    for generation, hv in enumerate([1.0, 1.0, 1.0], start=1):
        decision = _step(
            state,
            policy,
            hv,
            raw_scores=[0, 1, 1, 1],
            fitness=[-10.0, 1.0, 1.0, 1.0],
            generation=generation,
        )

    assert not decision.triggered
    assert decision.zero_fraction == 0.25
    assert decision.hv_delta == 0.0


def test_restart_does_not_trigger_when_hv_improves():
    """Catches a restart that ignores meaningful hypervolume improvement."""
    RestartPolicy, RestartState, _ = _restart_api()
    policy = RestartPolicy(theta_zero=0.8, window=3, delta_hv=0.05, fraction=0.4)
    state = RestartState()

    for generation, hv in enumerate([1.0, 1.02, 1.1], start=1):
        decision = _step(state, policy, hv, generation=generation)

    assert not decision.triggered
    assert decision.hv_delta == pytest.approx(0.1)


def test_restart_triggers_on_stagnation_and_selects_stable_worst_fitness():
    """Catches reversed replacement ordering or unstable equal-fitness ties."""
    RestartPolicy, RestartState, _ = _restart_api()
    policy = RestartPolicy(theta_zero=0.8, window=3, delta_hv=1e-4, fraction=0.4)
    state = RestartState()
    fitness = [-2.0, -2.0, -1.0, 0.0, 1.0]

    for generation, hv in enumerate([1.0, 1.00001, 1.00002], start=1):
        decision = _step(state, policy, hv, fitness=fitness, generation=generation)

    assert decision.triggered
    assert decision.hv_delta == pytest.approx(0.00002)
    npt.assert_array_equal(decision.replace_indices, [0, 1])
    assert state.hv_history == []


def test_restart_uses_strict_delta_boundary():
    """Catches changing the declared `< delta_hv` condition to `<=`."""
    RestartPolicy, RestartState, _ = _restart_api()
    policy = RestartPolicy(theta_zero=0.8, window=3, delta_hv=0.25, fraction=0.4)
    state = RestartState()

    for generation, hv in enumerate([1.0, 1.125, 1.25], start=1):
        decision = _step(state, policy, hv, generation=generation)

    assert not decision.triggered
    assert decision.hv_delta == 0.25


def test_restart_replaces_at_least_one_individual():
    """Catches floor rounding that turns a positive restart fraction into no action."""
    RestartPolicy, RestartState, _ = _restart_api()
    policy = RestartPolicy(theta_zero=1.0, window=2, delta_hv=0.1, fraction=0.01)
    state = RestartState()

    _step(state, policy, 1.0, raw_scores=[0, 0, 0], fitness=[-3, -2, -1], generation=1)
    decision = _step(
        state,
        policy,
        1.0,
        raw_scores=[0, 0, 0],
        fitness=[-3, -2, -1],
        generation=2,
    )

    npt.assert_array_equal(decision.replace_indices, [0])


def test_restart_rejects_nonfinite_hv_without_mutating_history():
    """Catches NaN stagnation values silently poisoning later decisions."""
    RestartPolicy, RestartState, _ = _restart_api()
    policy = RestartPolicy(window=3)
    state = RestartState(hv_history=[1.0])

    with pytest.raises(ValueError):
        _step(state, policy, np.nan, generation=2)

    assert state.hv_history == [1.0]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"theta_zero": -0.1},
        {"theta_zero": 1.1},
        {"window": 1},
        {"delta_hv": -0.1},
        {"fraction": 0.0},
        {"fraction": 1.1},
    ],
)
def test_restart_policy_rejects_invalid_values(kwargs):
    """Catches experiment manifests with undefined restart semantics."""
    RestartPolicy, _, _ = _restart_api()
    with pytest.raises(ValueError):
        RestartPolicy(**kwargs)


def test_restart_history_retains_only_the_declared_window():
    """Catches unbounded per-run restart memory."""
    RestartPolicy, RestartState, _ = _restart_api()
    policy = RestartPolicy(theta_zero=1.0, window=3, delta_hv=0.0)
    state = RestartState()

    for generation, hv in enumerate([1.0, 2.0, 3.0, 4.0], start=1):
        _step(state, policy, hv, raw_scores=[0, 1, 1, 1, 1], generation=generation)

    assert state.hv_history == [2.0, 3.0, 4.0]
