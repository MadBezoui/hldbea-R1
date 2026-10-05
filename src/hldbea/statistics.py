"""Predeclared, direction-aware statistics for paired experiment outcomes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import friedmanchisquare, rankdata, wilcoxon


METRIC_DIRECTIONS = {
    "hv": "higher",
    "igd_plus": "lower",
    "phi_nz": "higher",
    "r_fz": "lower",
}


def _direction(value: str) -> str:
    if value not in {"higher", "lower"}:
        raise ValueError("direction must be 'higher' or 'lower'")
    return value


def metric_direction(metric: str) -> str:
    try:
        return METRIC_DIRECTIONS[metric]
    except KeyError as exc:
        raise ValueError(f"metric direction is undeclared: {metric}") from exc


def _finite_vector(values: Sequence[float], name: str) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.ndim != 1 or len(result) == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional sample")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


@dataclass(frozen=True)
class DescriptiveSummary:
    n: int
    median: float
    q1: float
    q3: float
    iqr: float
    ci_low: float
    ci_high: float
    confidence: float
    bootstrap_samples: int
    bootstrap_seed: int


def summarize(
    values: Sequence[float],
    *,
    confidence: float = 0.95,
    bootstrap_samples: int = 10_000,
    seed: int = 0,
) -> DescriptiveSummary:
    sample = _finite_vector(values, "values")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie strictly between zero and one")
    if (
        not isinstance(bootstrap_samples, int)
        or isinstance(bootstrap_samples, bool)
        or bootstrap_samples < 1
    ):
        raise ValueError("bootstrap_samples must be a positive integer")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    q1, median, q3 = np.quantile(sample, [0.25, 0.5, 0.75], method="linear")
    generator = np.random.default_rng(seed)
    indices = generator.integers(
        0, len(sample), size=(bootstrap_samples, len(sample))
    )
    bootstrap_medians = np.median(sample[indices], axis=1)
    alpha = 1.0 - confidence
    ci_low, ci_high = np.quantile(
        bootstrap_medians,
        [alpha / 2.0, 1.0 - alpha / 2.0],
        method="linear",
    )
    return DescriptiveSummary(
        n=len(sample),
        median=float(median),
        q1=float(q1),
        q3=float(q3),
        iqr=float(q3 - q1),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        confidence=float(confidence),
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=seed,
    )


@dataclass(frozen=True)
class PairedWilcoxonResult:
    statistic: float
    p_value: float
    n_pairs: int
    n_effective: int
    direction: str
    alternative: str
    scipy_alternative: str
    zero_method: str
    failure_policy: str
    missing_pairs: tuple[str, ...]
    status: str


def _paired_values(
    first: Sequence[float] | Mapping[Any, float],
    second: Sequence[float] | Mapping[Any, float],
    *,
    failure_policy: str,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    if failure_policy not in {"raise", "omit"}:
        raise ValueError("failure_policy must be 'raise' or 'omit'")
    missing: list[str] = []
    if isinstance(first, Mapping) and isinstance(second, Mapping):
        first_keys = set(first)
        second_keys = set(second)
        unmatched = first_keys ^ second_keys
        if unmatched and failure_policy == "raise":
            raise ValueError(f"paired keys differ: {sorted(map(str, unmatched))}")
        missing.extend(sorted(map(str, unmatched)))
        keys = sorted(first_keys & second_keys, key=str)
        left = np.asarray([first[key] for key in keys], dtype=float)
        right = np.asarray([second[key] for key in keys], dtype=float)
        labels = [str(key) for key in keys]
    elif not isinstance(first, Mapping) and not isinstance(second, Mapping):
        left = np.asarray(first, dtype=float)
        right = np.asarray(second, dtype=float)
        if left.ndim != 1 or right.ndim != 1:
            raise ValueError("paired samples must be one-dimensional")
        if len(left) != len(right):
            raise ValueError("paired samples must have equal lengths")
        labels = [str(index) for index in range(len(left))]
    else:
        raise TypeError("paired samples must both be mappings or both be sequences")
    finite = np.isfinite(left) & np.isfinite(right)
    if not np.all(finite):
        failed = [labels[index] for index in np.flatnonzero(~finite)]
        if failure_policy == "raise":
            raise ValueError(f"paired samples contain non-finite values: {failed}")
        missing.extend(failed)
        left = left[finite]
        right = right[finite]
    if len(left) == 0:
        raise ValueError("no complete finite pairs remain")
    return left, right, tuple(sorted(set(missing)))


def paired_wilcoxon(
    first: Sequence[float] | Mapping[Any, float],
    second: Sequence[float] | Mapping[Any, float],
    *,
    direction: str,
    zero_method: str = "wilcox",
    failure_policy: str = "raise",
    alternative: str = "two-sided",
) -> PairedWilcoxonResult:
    direction = _direction(direction)
    if zero_method not in {"wilcox", "pratt", "zsplit"}:
        raise ValueError("zero_method must be 'wilcox', 'pratt', or 'zsplit'")
    alternatives = {
        "two-sided": "two-sided",
        "first_better": "greater" if direction == "higher" else "less",
        "first_worse": "less" if direction == "higher" else "greater",
    }
    if alternative not in alternatives:
        raise ValueError(
            "alternative must be 'two-sided', 'first_better', or 'first_worse'"
        )
    scipy_alternative = alternatives[alternative]
    left, right, missing = _paired_values(
        first, second, failure_policy=failure_policy
    )
    differences = left - right
    n_effective = (
        int(np.count_nonzero(differences))
        if zero_method == "wilcox"
        else len(differences)
    )
    if np.all(differences == 0):
        statistic, p_value, status = 0.0, 1.0, "all_zero"
    else:
        if zero_method == "wilcox":
            # Drop exact zeros explicitly. scipy otherwise forces a normal
            # approximation (and warning) even when the remaining paired
            # sample is small enough for its deterministic exact test.
            tested = differences[differences != 0]
            result = wilcoxon(
                tested,
                zero_method="wilcox",
                alternative=scipy_alternative,
                method="auto",
            )
        else:
            result = wilcoxon(
                left,
                right,
                zero_method=zero_method,
                alternative=scipy_alternative,
                method="auto",
            )
        statistic = float(result.statistic)
        p_value = float(result.pvalue)
        status = "ok"
    return PairedWilcoxonResult(
        statistic=statistic,
        p_value=p_value,
        n_pairs=len(left),
        n_effective=n_effective,
        direction=direction,
        alternative=alternative,
        scipy_alternative=scipy_alternative,
        zero_method=zero_method,
        failure_policy=failure_policy,
        missing_pairs=missing,
        status=status,
    )


@dataclass(frozen=True)
class HolmResult:
    hypothesis: str
    family: str
    p_raw: float
    p_adjusted: float
    reject: bool
    alpha: float


def holm_step_down(
    p_values: Mapping[str, float],
    *,
    family: str,
    alpha: float = 0.05,
) -> dict[str, HolmResult]:
    if not isinstance(family, str) or not family:
        raise ValueError("family must be a non-empty string")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    if not p_values:
        raise ValueError("p_values must not be empty")
    normalized = {}
    for hypothesis, value in p_values.items():
        if not isinstance(hypothesis, str) or not hypothesis:
            raise ValueError("hypothesis names must be non-empty strings")
        if not isinstance(value, (int, float)) or not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("raw p-values must be finite and within [0, 1]")
        normalized[hypothesis] = float(value)
    ordered = sorted(normalized.items(), key=lambda item: (item[1], item[0]))
    adjusted: dict[str, HolmResult] = {}
    running = 0.0
    count = len(ordered)
    for index, (hypothesis, p_raw) in enumerate(ordered):
        running = max(running, min(1.0, (count - index) * p_raw))
        adjusted[hypothesis] = HolmResult(
            hypothesis=hypothesis,
            family=family,
            p_raw=p_raw,
            p_adjusted=running,
            reject=running <= alpha,
            alpha=alpha,
        )
    return adjusted


@dataclass(frozen=True)
class A12Result:
    a12: float
    direction: str
    n_first: int
    n_second: int
    interpretation: str


def vargha_delaney_a12(
    first: Sequence[float], second: Sequence[float], *, direction: str
) -> A12Result:
    direction = _direction(direction)
    left = _finite_vector(first, "first")
    right = _finite_vector(second, "second")
    if direction == "higher":
        wins = left[:, None] > right[None, :]
    else:
        wins = left[:, None] < right[None, :]
    ties = left[:, None] == right[None, :]
    estimate = float((np.count_nonzero(wins) + 0.5 * np.count_nonzero(ties)) / wins.size)
    if estimate > 0.5:
        interpretation = "first_better"
    elif estimate < 0.5:
        interpretation = "second_better"
    else:
        interpretation = "tie"
    return A12Result(
        a12=estimate,
        direction=direction,
        n_first=len(left),
        n_second=len(right),
        interpretation=interpretation,
    )


@dataclass(frozen=True)
class FriedmanResult:
    statistic: float
    p_value: float
    n_blocks: int
    algorithms: tuple[str, ...]
    mean_ranks: dict[str, float]
    direction: str


def friedman_complete_blocks(
    blocks: Mapping[Any, Mapping[str, float]], *, direction: str
) -> FriedmanResult:
    direction = _direction(direction)
    if len(blocks) < 2:
        raise ValueError("Friedman test requires at least two complete blocks")
    ordered_blocks = sorted(blocks, key=str)
    first_algorithms = set(blocks[ordered_blocks[0]])
    if len(first_algorithms) < 3:
        raise ValueError("Friedman test requires at least three algorithms")
    if any(set(blocks[key]) != first_algorithms for key in ordered_blocks):
        raise ValueError("Friedman test requires complete matched blocks")
    algorithms = tuple(sorted(first_algorithms))
    matrix = np.asarray(
        [[blocks[block][algorithm] for algorithm in algorithms] for block in ordered_blocks],
        dtype=float,
    )
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Friedman blocks must contain only finite values")
    if np.all(matrix == matrix[:, :1]):
        # Every block is fully tied: no evidence of any difference.
        statistic, p_value = 0.0, 1.0
    else:
        result = friedmanchisquare(
            *(matrix[:, index] for index in range(len(algorithms)))
        )
        statistic, p_value = float(result.statistic), float(result.pvalue)
    oriented = -matrix if direction == "higher" else matrix
    ranks = np.row_stack([rankdata(row, method="average") for row in oriented])
    return FriedmanResult(
        statistic=statistic,
        p_value=p_value,
        n_blocks=len(ordered_blocks),
        algorithms=algorithms,
        mean_ranks={
            algorithm: float(value)
            for algorithm, value in zip(algorithms, np.mean(ranks, axis=0))
        },
        direction=direction,
    )


def observations_digest(records) -> str:
    """Canonical SHA-256 of the observations a statistical report analysed.

    ``records`` is an iterable of ``(run_id, problem_id, label, seed, values)``
    tuples, where ``values`` maps metric names to floats. Order does not matter.
    """

    import hashlib
    import json

    rows = sorted(
        [
            str(run_id),
            str(problem_id),
            str(label),
            int(seed),
            {str(k): repr(float(v)) for k, v in sorted(dict(values).items())},
        ]
        for run_id, problem_id, label, seed, values in records
    )
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def matched_pairs_rank_biserial(first, second, *, direction: str) -> float:
    """Paired effect size in [-1, 1]; positive means ``first`` is better.

    Zero differences are discarded, as in the Wilcoxon ``wilcox`` method.
    """

    direction = _direction(direction)
    a = np.asarray(first, dtype=float)
    b = np.asarray(second, dtype=float)
    if a.shape != b.shape:
        raise ValueError("paired samples must have the same shape")
    diff = a - b if direction == "higher" else b - a
    diff = diff[diff != 0]
    if diff.size == 0:
        return 0.0
    ranks = rankdata(np.abs(diff))
    positive = float(np.sum(ranks[diff > 0]))
    negative = float(np.sum(ranks[diff < 0]))
    return (positive - negative) / (positive + negative)
