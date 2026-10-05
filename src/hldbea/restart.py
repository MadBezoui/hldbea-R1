"""Pure stagnation detection based on raw local scores and hypervolume."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class RestartPolicy:
    theta_zero: float = 0.95
    window: int = 20
    delta_hv: float = 1e-4
    fraction: float = 0.2

    def __post_init__(self) -> None:
        if not 0.0 <= self.theta_zero <= 1.0:
            raise ValueError("theta_zero must lie in [0, 1]")
        if not isinstance(self.window, (int, np.integer)) or self.window < 2:
            raise ValueError("window must be an integer of at least two")
        if not np.isfinite(self.delta_hv) or self.delta_hv < 0.0:
            raise ValueError("delta_hv must be finite and non-negative")
        if not np.isfinite(self.fraction) or not 0.0 < self.fraction <= 1.0:
            raise ValueError("fraction must lie in (0, 1]")


@dataclass
class RestartState:
    hv_history: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class RestartDecision:
    triggered: bool
    zero_fraction: float
    hv_delta: float | None
    replace_indices: np.ndarray


def update_restart_state(
    state: RestartState,
    policy: RestartPolicy,
    *,
    raw_scores: np.ndarray,
    fitness: np.ndarray,
    hv: float,
    generation: int,
) -> RestartDecision:
    """Update the bounded HV history and return a deterministic restart decision."""

    scores = np.asarray(raw_scores)
    fit = np.asarray(fitness, dtype=float)
    if scores.ndim != 1 or len(scores) == 0:
        raise ValueError("raw_scores must be a non-empty vector")
    if fit.ndim != 1 or len(fit) != len(scores):
        raise ValueError("fitness must be a vector aligned with raw_scores")
    if not np.all(np.isfinite(scores)) or np.any(scores < 0):
        raise ValueError("raw_scores must contain finite non-negative values")
    if not np.all(np.isfinite(fit)):
        raise ValueError("fitness must contain only finite values")
    if not np.isfinite(hv):
        raise ValueError("hv must be finite")
    if not isinstance(generation, (int, np.integer)) or generation < 0:
        raise ValueError("generation must be a non-negative integer")

    state.hv_history.append(float(hv))
    if len(state.hv_history) > policy.window:
        del state.hv_history[: len(state.hv_history) - policy.window]

    zero_fraction = float(np.mean(scores == 0))
    hv_delta: float | None = None
    if len(state.hv_history) == policy.window:
        hv_delta = abs(state.hv_history[-1] - state.hv_history[0])

    triggered = (
        zero_fraction >= policy.theta_zero
        and hv_delta is not None
        and hv_delta < policy.delta_hv
    )
    replace_indices = np.empty(0, dtype=int)
    if triggered:
        n_replace = max(1, int(np.floor(policy.fraction * len(scores))))
        replace_indices = np.argsort(fit, kind="stable")[:n_replace]
        state.hv_history.clear()

    return RestartDecision(
        triggered=triggered,
        zero_fraction=zero_fraction,
        hv_delta=hv_delta,
        replace_indices=replace_indices,
    )
