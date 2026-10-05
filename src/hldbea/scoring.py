"""Local dominance scores with explicit strict and diagnostic semantics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ScoreResult:
    """All score components used by selection and experiment diagnostics."""

    steps: np.ndarray
    per_axis_counts: np.ndarray
    raw_scores: np.ndarray
    normalized_distances: np.ndarray
    fitness: np.ndarray
    nondominated: np.ndarray
    false_zero: np.ndarray


def _validated_objectives(F: np.ndarray) -> np.ndarray:
    values = np.asarray(F, dtype=float)
    if values.ndim != 2:
        raise ValueError("F must be a two-dimensional objective matrix")
    if values.shape[0] == 0:
        raise ValueError("F must contain at least one individual")
    if values.shape[1] < 2:
        raise ValueError("F must contain at least two objectives")
    if not np.all(np.isfinite(values)):
        raise ValueError("F must contain only finite objective values")
    return values


def pareto_nondominated_mask(F: np.ndarray) -> np.ndarray:
    """Return a mask for minimization Pareto nondominance."""

    values = _validated_objectives(F)
    nondominated = np.ones(values.shape[0], dtype=bool)
    for target in range(values.shape[0]):
        weakly_better = np.all(values <= values[target], axis=1)
        strictly_better = np.any(values < values[target], axis=1)
        nondominated[target] = not np.any(weakly_better & strictly_better)
    return nondominated


def compute_local_scores(
    F: np.ndarray,
    *,
    k: float = 1.0,
    cone_epsilon: float = 0.0,
    lmbda: float = 0.1,
    safe_epsilon: float = 1e-12,
    neighborhood_mode: str = "axis",
    score_aggregation: str = "sum",
) -> ScoreResult:
    """Compute local scores under one explicitly declared neighbourhood.

    ``axis`` is the paper's directional-band relation. ``box`` uses one full
    lower orthotope. ``knn`` uses the nearest ``ceil(k*sqrt(N-1))`` objective
    vectors after per-axis range normalization. Only the axis relation can
    either sum repeated memberships or count their union.
    """

    values = _validated_objectives(F)
    if k < 0:
        raise ValueError("k must be non-negative")
    if cone_epsilon < 0:
        raise ValueError("cone_epsilon must be non-negative")
    if lmbda < 0:
        raise ValueError("lmbda must be non-negative")
    if safe_epsilon <= 0:
        raise ValueError("safe_epsilon must be positive")
    if neighborhood_mode not in {"axis", "box", "knn"}:
        raise ValueError("neighborhood_mode must be 'axis', 'box', or 'knn'")
    if score_aggregation not in {"sum", "union"}:
        raise ValueError("score_aggregation must be 'sum' or 'union'")
    if neighborhood_mode != "axis" and score_aggregation != "union":
        raise ValueError("box and knn neighbourhoods require union aggregation")

    n_individuals, n_objectives = values.shape
    steps = k * np.ptp(values, axis=0) / n_individuals
    per_axis_counts = np.zeros((n_individuals, n_objectives), dtype=int)
    raw_scores = np.zeros(n_individuals, dtype=int)

    if neighborhood_mode == "axis":
        for target in range(n_individuals):
            union = np.zeros(n_individuals, dtype=bool)
            for axis in range(n_objectives):
                in_band = (
                    (values[:, axis] >= values[target, axis] - steps[axis])
                    & (values[:, axis] <= values[target, axis])
                )
                excluded = np.arange(n_objectives) != axis
                within_cone = np.all(
                    values[:, excluded]
                    < values[target, excluded]
                    + cone_epsilon * steps[excluded],
                    axis=1,
                )
                neighbors = in_band & within_cone
                neighbors[target] = False
                union |= neighbors
                per_axis_counts[target, axis] = int(np.count_nonzero(neighbors))
            raw_scores[target] = (
                int(np.sum(per_axis_counts[target]))
                if score_aggregation == "sum"
                else int(np.count_nonzero(union))
            )
    elif neighborhood_mode == "box":
        for target in range(n_individuals):
            lower = values >= values[target] - steps
            upper = values <= values[target] + cone_epsilon * steps
            improving = np.any(values < values[target], axis=1)
            neighbors = np.all(lower & upper, axis=1) & improving
            neighbors[target] = False
            raw_scores[target] = int(np.count_nonzero(neighbors))
            per_axis_counts[target, 0] = raw_scores[target]
    else:
        span = np.maximum(np.ptp(values, axis=0), safe_epsilon)
        normalized = (values - np.min(values, axis=0)) / span
        neighbor_count = (
            0
            if k == 0 or n_individuals == 1
            else min(
                n_individuals - 1,
                max(1, int(np.ceil(k * np.sqrt(n_individuals - 1)))),
            )
        )
        for target in range(n_individuals):
            if neighbor_count == 0:
                continue
            distances = np.linalg.norm(normalized - normalized[target], axis=1)
            distances[target] = np.inf
            nearest = np.argsort(distances, kind="stable")[:neighbor_count]
            relaxed_dominance = np.all(
                values < values[target] + cone_epsilon * steps,
                axis=1,
            ) & np.any(values < values[target], axis=1)
            neighbors = np.zeros(n_individuals, dtype=bool)
            neighbors[nearest] = relaxed_dominance[nearest]
            raw_scores[target] = int(np.count_nonzero(neighbors))
            per_axis_counts[target, 0] = raw_scores[target]

    nondominated = pareto_nondominated_mask(values)
    ideal = np.min(values, axis=0)
    distances = np.linalg.norm(values - ideal, axis=1)
    normalized_distances = distances / max(float(np.max(distances)), safe_epsilon)
    fitness = -raw_scores.astype(float) - lmbda * normalized_distances
    false_zero = (raw_scores == 0) & ~nondominated
    return ScoreResult(
        steps=steps,
        per_axis_counts=per_axis_counts,
        raw_scores=raw_scores,
        normalized_distances=normalized_distances,
        fitness=fitness,
        nondominated=nondominated,
        false_zero=false_zero,
    )
