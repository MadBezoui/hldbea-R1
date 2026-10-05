"""Deterministic survivor selection for confirmatory HLDBEA experiments."""

from __future__ import annotations

import numpy as np
from pymoo.util.nds.fast_non_dominated_sort import fast_non_dominated_sort


def _validated_objectives(F: np.ndarray) -> np.ndarray:
    values = np.asarray(F, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] < 2:
        raise ValueError("F must be a non-empty two-dimensional objective matrix")
    if not np.all(np.isfinite(values)):
        raise ValueError("F must contain only finite values")
    return values


def _crowding_distance(F: np.ndarray) -> np.ndarray:
    """Return standard NSGA-II crowding distance with stable axis ordering."""

    n_points, n_objectives = F.shape
    distances = np.zeros(n_points, dtype=float)
    if n_points <= 2:
        distances.fill(np.inf)
        return distances

    for axis in range(n_objectives):
        order = np.argsort(F[:, axis], kind="stable")
        ordered = F[order, axis]
        span = ordered[-1] - ordered[0]
        if span <= 0:
            continue
        distances[order[0]] = np.inf
        distances[order[-1]] = np.inf
        interior = order[1:-1]
        distances[interior] += (ordered[2:] - ordered[:-2]) / span
    return distances


def rank_and_crowding_indices(F: np.ndarray, n_survive: int) -> np.ndarray:
    """Select minimization survivors by nondominated rank then crowding."""

    values = _validated_objectives(F)
    if not isinstance(n_survive, (int, np.integer)) or isinstance(n_survive, bool):
        raise ValueError("n_survive must be an integer")
    if n_survive <= 0 or n_survive > len(values):
        raise ValueError("n_survive must lie between one and the population size")

    selected: list[int] = []
    fronts = fast_non_dominated_sort(values)
    for raw_front in fronts:
        front = np.asarray(raw_front, dtype=int)
        remaining = n_survive - len(selected)
        if len(front) <= remaining:
            selected.extend(front.tolist())
        else:
            crowding = _crowding_distance(values[front])
            tie_order = np.lexsort((front, -crowding))
            selected.extend(front[tie_order[:remaining]].tolist())
            break
        if len(selected) == n_survive:
            break
    return np.asarray(selected, dtype=int)


def select_survivor_indices(
    F: np.ndarray,
    raw_scores: np.ndarray,
    n_survive: int,
    *,
    mode: str = "local",
) -> np.ndarray:
    """Select survivors globally or from the raw-score-zero stratum first."""

    values = _validated_objectives(F)
    scores = np.asarray(raw_scores)
    if scores.ndim != 1 or len(scores) != len(values):
        raise ValueError("raw_scores must be a vector aligned with F")
    if not np.all(np.isfinite(scores)) or np.any(scores < 0):
        raise ValueError("raw_scores must contain finite non-negative values")
    if mode not in {"local", "global"}:
        raise ValueError("mode must be 'local' or 'global'")

    if mode == "global":
        return rank_and_crowding_indices(values, n_survive)

    if not isinstance(n_survive, (int, np.integer)) or isinstance(n_survive, bool):
        raise ValueError("n_survive must be an integer")
    if n_survive <= 0 or n_survive > len(values):
        raise ValueError("n_survive must lie between one and the population size")

    zero_indices = np.flatnonzero(scores == 0)
    if len(zero_indices) >= n_survive:
        local = rank_and_crowding_indices(values[zero_indices], n_survive)
        return zero_indices[local]

    rest_indices = np.flatnonzero(scores != 0)
    remaining = n_survive - len(zero_indices)
    rest_selected = rank_and_crowding_indices(values[rest_indices], remaining)
    return np.concatenate((zero_indices, rest_indices[rest_selected])).astype(int)
