"""Deterministic reference geometry for benchmark metrics."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
from numba import njit
from pymoo.indicators.hv import HV
from pymoo.indicators.igd_plus import IGDPlus
from pymoo.util.ref_dirs import get_reference_directions
from scipy.stats import qmc

from .run_spec import canonical_sha256
from .scoring import compute_local_scores, pareto_nondominated_mask


_EXACT_HV_DIMENSION_LIMIT = 5
_SOBOL_HV_POWER = 16
_SOBOL_HV_SEED = 20261003


@contextmanager
def isolated_numpy_seed(seed: int):
    """Temporarily seed legacy NumPy RNG users without leaking global state."""

    state = np.random.get_state()
    try:
        np.random.seed(seed)
        yield
    finally:
        np.random.set_state(state)


def metric_reference_directions(n_obj: int) -> np.ndarray:
    """Return a tractable, deterministic Das--Dennis design through 15 objectives."""

    if not isinstance(n_obj, (int, np.integer)) or isinstance(n_obj, bool) or n_obj < 2:
        raise ValueError("n_obj must be an integer of at least two")
    if n_obj <= 2:
        partitions = 99
    elif n_obj <= 3:
        partitions = 12
    elif n_obj <= 5:
        partitions = 6
    elif n_obj <= 10:
        partitions = 3
    else:
        partitions = 2
    return np.asarray(
        get_reference_directions("das-dennis", n_obj, n_partitions=partitions),
        dtype=float,
    )


def pareto_front_for_metrics(problem) -> np.ndarray:
    """Resolve a deterministic analytical/reference Pareto front when available."""

    directions = metric_reference_directions(int(problem.n_obj))
    with isolated_numpy_seed(getattr(problem, "_reference_seed", 0)):
        try:
            front = problem.pareto_front(ref_dirs=directions)
        except TypeError:
            front = problem.pareto_front()
    values = np.asarray(front, dtype=float)
    if values.ndim != 2 or values.shape[1] != int(problem.n_obj):
        raise ValueError("problem did not provide a dimensionally valid Pareto front")
    if not np.all(np.isfinite(values)):
        raise ValueError("Pareto front contains non-finite values")
    return values


def metric_reference_point(problem, *, margin: float = 0.1) -> np.ndarray:
    """Build an explicit point worse than a deterministic reference front."""

    if not np.isfinite(margin) or margin <= 0:
        raise ValueError("margin must be finite and positive")
    front = pareto_front_for_metrics(problem)
    ideal = np.min(front, axis=0)
    nadir = np.max(front, axis=0)
    scale = np.maximum(nadir - ideal, np.maximum(np.abs(nadir), 1.0))
    reference = nadir + margin * scale
    if reference.shape != (int(problem.n_obj),) or not np.all(np.isfinite(reference)):
        raise ValueError("failed to construct a valid metric reference point")
    return reference


def metric_reference_bounds(
    problem, *, margin: float = 0.1
) -> tuple[np.ndarray, np.ndarray]:
    """Return a fixed ideal/reference box for algorithm-internal HV diagnostics."""

    if not np.isfinite(margin) or margin <= 0:
        raise ValueError("margin must be finite and positive")
    front = pareto_front_for_metrics(problem)
    ideal = np.min(front, axis=0)
    nadir = np.max(front, axis=0)
    scale = np.maximum(nadir - ideal, np.maximum(np.abs(nadir), 1.0))
    reference = nadir + margin * scale
    if (
        ideal.shape != (int(problem.n_obj),)
        or reference.shape != ideal.shape
        or not np.all(np.isfinite(ideal))
        or not np.all(np.isfinite(reference))
        or np.any(reference <= ideal)
    ):
        raise ValueError("failed to construct valid metric reference bounds")
    return ideal, reference


def hypervolume_estimator_metadata(n_obj: int) -> dict[str, Any]:
    """Describe the dimension-aware HV implementation without hidden defaults."""

    if not isinstance(n_obj, (int, np.integer)) or isinstance(n_obj, bool) or n_obj < 2:
        raise ValueError("n_obj must be an integer of at least two")
    if n_obj <= _EXACT_HV_DIMENSION_LIMIT:
        return {"kind": "exact"}
    return {
        "kind": "sobol_qmc",
        "exact_dimension_limit": _EXACT_HV_DIMENSION_LIMIT,
        "samples": 2**_SOBOL_HV_POWER,
        "scramble": True,
        "seed": _SOBOL_HV_SEED,
        "lower_bound": "reference_ideal",
    }


@lru_cache(maxsize=None)
def _sobol_unit_samples(n_obj: int) -> np.ndarray:
    sampler = qmc.Sobol(d=n_obj, scramble=True, seed=_SOBOL_HV_SEED)
    samples = np.asarray(sampler.random_base2(_SOBOL_HV_POWER), dtype=float)
    samples.setflags(write=False)
    return samples


@njit(cache=True)
def _dominated_sample_count(samples: np.ndarray, points: np.ndarray) -> int:
    """Count dominated QMC samples with early exits and one CPU thread."""

    count = 0
    for sample_index in range(samples.shape[0]):
        for point_index in range(points.shape[0]):
            is_dominated = True
            for objective_index in range(samples.shape[1]):
                if points[point_index, objective_index] > samples[
                    sample_index, objective_index
                ]:
                    is_dominated = False
                    break
            if is_dominated:
                count += 1
                break
    return count


def compute_hypervolume(
    F: np.ndarray,
    reference_point: np.ndarray,
    *,
    ideal_point: np.ndarray | None = None,
) -> float:
    """Compute exact low-dimensional HV or fixed-box deterministic Sobol HV.

    Exact HV becomes exponential on nearly nondominated many-objective WFG
    populations. Above five objectives, all calls reuse the same scrambled
    Sobol design and fixed ideal/reference box. Differences therefore use
    common deterministic samples rather than independent Monte Carlo noise.
    """

    values = np.asarray(F, dtype=float)
    reference = np.asarray(reference_point, dtype=float)
    if values.ndim != 2 or len(values) == 0:
        raise ValueError("objective matrix must be non-empty and two-dimensional")
    if not np.all(np.isfinite(values)):
        raise ValueError("objective matrix must be finite")
    if reference.shape != (values.shape[1],) or not np.all(np.isfinite(reference)):
        raise ValueError("reference point has invalid geometry")
    if values.shape[1] <= _EXACT_HV_DIMENSION_LIMIT:
        return float(HV(ref_point=reference)(values))

    if ideal_point is None:
        raise ValueError("many-objective hypervolume requires a fixed ideal point")
    ideal = np.asarray(ideal_point, dtype=float)
    if (
        ideal.shape != reference.shape
        or not np.all(np.isfinite(ideal))
        or np.any(reference <= ideal)
    ):
        raise ValueError("ideal/reference hypervolume box is invalid")

    eligible = values[np.all(values < reference, axis=1)]
    if len(eligible) == 0:
        return 0.0
    eligible = eligible[pareto_nondominated_mask(eligible)]
    unit = _sobol_unit_samples(values.shape[1])
    samples = np.ascontiguousarray(ideal + unit * (reference - ideal))
    eligible = np.ascontiguousarray(eligible)
    dominated_count = _dominated_sample_count(samples, eligible)
    box_volume = float(np.prod(reference - ideal))
    return box_volume * dominated_count / len(samples)


@dataclass(frozen=True)
class ReferenceGeometry:
    problem_id: str
    n_obj: int
    reference_directions: np.ndarray
    reference_set: np.ndarray
    ideal: np.ndarray
    nadir: np.ndarray
    hv_reference_point: np.ndarray
    transform_hash: str
    geometry_hash: str

    def metadata(self) -> dict[str, Any]:
        metadata = {
            "problem_id": self.problem_id,
            "n_obj": self.n_obj,
            "ideal": self.ideal.tolist(),
            "nadir": self.nadir.tolist(),
            "hv_reference_point": self.hv_reference_point.tolist(),
            "transform_hash": self.transform_hash,
            "geometry_hash": self.geometry_hash,
            "reference_set_rows": len(self.reference_set),
        }
        if self.n_obj > _EXACT_HV_DIMENSION_LIMIT:
            metadata["hv_estimator"] = hypervolume_estimator_metadata(self.n_obj)
        return metadata


def build_reference_geometry(problem_spec) -> ReferenceGeometry:
    from .problems import ProblemSpec, build_problem

    if not isinstance(problem_spec, ProblemSpec):
        try:
            problem_spec = ProblemSpec.from_run_spec(problem_spec)
        except (AttributeError, TypeError) as exc:
            raise TypeError("problem_spec must be a ProblemSpec or RunSpec") from exc
    problem = build_problem(problem_spec)
    directions = metric_reference_directions(problem_spec.n_obj)
    with isolated_numpy_seed(problem._reference_seed):
        try:
            reference_set = problem.pareto_front(ref_dirs=directions)
        except TypeError:
            reference_set = problem.pareto_front()
    reference_set = np.asarray(reference_set, dtype=float)
    if (
        reference_set.ndim != 2
        or reference_set.shape[1] != problem_spec.n_obj
        or not np.all(np.isfinite(reference_set))
    ):
        raise ValueError("reference set is invalid")
    ideal = np.min(reference_set, axis=0)
    nadir = np.max(reference_set, axis=0)
    span = nadir - ideal
    if np.any(span <= 0) or not np.all(np.isfinite(span)):
        raise ValueError("reference geometry has a degenerate objective span")
    hv_reference = np.full(problem_spec.n_obj, 1.1, dtype=float)
    transform_hash = problem.objective_transform["hash"]
    payload = {
        "problem": problem_spec.canonical_dict(),
        "reference_directions": directions.tolist(),
        "reference_set": reference_set.tolist(),
        "ideal": ideal.tolist(),
        "nadir": nadir.tolist(),
        "hv_reference_point": hv_reference.tolist(),
        "transform_hash": transform_hash,
    }
    if problem_spec.n_obj > _EXACT_HV_DIMENSION_LIMIT:
        payload["hv_estimator"] = hypervolume_estimator_metadata(problem_spec.n_obj)
    return ReferenceGeometry(
        problem_id=problem_spec.id,
        n_obj=problem_spec.n_obj,
        reference_directions=directions,
        reference_set=reference_set,
        ideal=ideal,
        nadir=nadir,
        hv_reference_point=hv_reference,
        transform_hash=transform_hash,
        geometry_hash=canonical_sha256(payload),
    )


def normalize_objectives(F: np.ndarray, geometry: ReferenceGeometry) -> np.ndarray:
    values = np.asarray(F, dtype=float)
    if values.ndim != 2 or values.shape[1] != geometry.n_obj:
        raise ValueError("objective dimension does not match reference geometry")
    if len(values) == 0 or not np.all(np.isfinite(values)):
        raise ValueError("objective matrix must be non-empty and finite")
    span = geometry.nadir - geometry.ideal
    if np.any(span <= 0):
        raise ValueError("reference geometry has a degenerate dimension")
    return (values - geometry.ideal) / span


def compute_metrics(F: np.ndarray, geometry: ReferenceGeometry) -> dict[str, float]:
    approximation = normalize_objectives(F, geometry)
    reference = normalize_objectives(geometry.reference_set, geometry)
    hv = compute_hypervolume(
        approximation,
        geometry.hv_reference_point,
        ideal_point=np.zeros(geometry.n_obj, dtype=float),
    )
    igd_plus = float(IGDPlus(reference)(approximation))
    if not np.isfinite(hv) or not np.isfinite(igd_plus):
        raise ValueError("metric computation produced non-finite values")
    return {"hv": hv, "igd_plus": igd_plus}


def compute_score_diagnostics(
    F: np.ndarray,
    *,
    k: float,
    cone_epsilon: float,
    neighborhood_mode: str = "axis",
    score_aggregation: str = "sum",
) -> dict[str, float]:
    result = compute_local_scores(
        F,
        k=k,
        cone_epsilon=cone_epsilon,
        lmbda=0.0,
        neighborhood_mode=neighborhood_mode,
        score_aggregation=score_aggregation,
    )
    zero_count = int(np.count_nonzero(result.raw_scores == 0))
    return {
        "phi_nz": float(np.mean(result.raw_scores > 0)),
        "r_fz": float(np.count_nonzero(result.false_zero) / max(1, zero_count)),
    }
