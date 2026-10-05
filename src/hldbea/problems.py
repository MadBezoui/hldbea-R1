"""Benchmark construction with explicit, reproducible objective transforms."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
from pymoo.core.problem import Problem
from pymoo.problems import get_problem
from scipy.stats import qmc

from .run_spec import RunSpec, canonical_sha256
from .transforms import orthonormal_rotation


@dataclass(frozen=True)
class ProblemSpec:
    id: str
    name: str
    n_obj: int
    n_var: int
    parameters: dict[str, Any]
    transform: dict[str, Any]

    @classmethod
    def from_run_spec(cls, spec: RunSpec) -> "ProblemSpec":
        return cls(
            id=spec.problem_id,
            name=spec.problem_name,
            n_obj=spec.n_obj,
            n_var=spec.n_var,
            parameters=deepcopy(spec.problem_parameters),
            transform=deepcopy(spec.transform),
        )

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "n_obj": self.n_obj,
            "n_var": self.n_var,
            "parameters": deepcopy(self.parameters),
            "transform": deepcopy(self.transform),
        }


def _base_problem(spec: ProblemSpec):
    name = spec.name.lower()
    if name == "rdtlz2":
        from .rotated_dtlz import build_rotated_front_dtlz2

        return build_rotated_front_dtlz2(
            n_var=spec.n_var,
            n_obj=spec.n_obj,
            parameters=deepcopy(spec.parameters),
        )
    if name == "rlinear":
        from .rotated_dtlz import build_rotated_linear_front

        return build_rotated_linear_front(
            n_var=spec.n_var,
            n_obj=spec.n_obj,
            parameters=deepcopy(spec.parameters),
        )
    if name.startswith("imop"):
        from .imop import build_imop_problem

        return build_imop_problem(
            name,
            n_var=spec.n_var,
            parameters=deepcopy(spec.parameters),
        )
    if not (name.startswith("zdt") or name.startswith("dtlz") or name.startswith("wfg")):
        raise ValueError(f"unsupported benchmark family: {spec.name}")
    if name.startswith("zdt"):
        if spec.n_obj != 2:
            raise ValueError("ZDT problems require exactly two objectives")
        return get_problem(name, n_var=spec.n_var, **deepcopy(spec.parameters))
    return get_problem(
        name,
        n_var=spec.n_var,
        n_obj=spec.n_obj,
        **deepcopy(spec.parameters),
    )


def reference_geometry_seed(spec: ProblemSpec) -> int:
    """Derive a problem-only seed shared by every algorithm and run seed."""

    digest = canonical_sha256(
        {"purpose": "reference-geometry", "problem": spec.canonical_dict()}
    )
    return int(digest[:8], 16)


def _wfg_reference_decisions(problem, ref_dirs: np.ndarray) -> np.ndarray:
    """Build a bounded deterministic sample of pymoo's exact WFG Pareto set.

    pymoo's generic WFG front starts by enumerating all ``2**k`` binary
    position vectors. That is tractable at five objectives but explodes to
    268,435,456 vectors for the standard 15-objective instance. WFG's own
    ``_positional_to_optimal`` mapping defines Pareto-set decisions directly,
    so a Sobol interior plus axis anchors provides valid Pareto points without
    the exponential enumeration.
    """

    directions = np.asarray(ref_dirs, dtype=float)
    if directions.ndim != 2 or directions.shape[1] != int(problem.n_obj):
        raise ValueError("WFG reference directions have invalid dimensions")
    k = int(problem.k)
    requested = max(512, 4 * len(directions))
    power = int(np.ceil(np.log2(requested)))
    seed = 20261003 + 1000 * int(problem.n_obj) + int(problem.n_var)
    positions = qmc.Sobol(d=k, scramble=True, seed=seed).random_base2(power)
    if problem.__class__.__name__.lower() == "wfg1":
        positions = np.power(positions, 50.0)
    identity = np.eye(k, dtype=float)
    anchors = np.row_stack(
        (
            np.zeros((1, k), dtype=float),
            np.ones((1, k), dtype=float),
            identity,
            1.0 - identity,
        )
    )
    positions = np.row_stack((anchors, positions))
    return np.asarray(problem._positional_to_optimal(positions), dtype=float)


def _nondominated_mask(values: np.ndarray, chunk: int = 512) -> np.ndarray:
    keep = np.ones(len(values), dtype=bool)
    for start in range(0, len(values), chunk):
        block = values[start : start + chunk]
        weakly = np.all(values[None, :, :] <= block[:, None, :], axis=2)
        strictly = np.any(values[None, :, :] < block[:, None, :], axis=2)
        keep[start : start + chunk] = ~np.any(weakly & strictly, axis=1)
    return keep


_DTLZ7_FRONTS: dict[tuple[int, int], np.ndarray] = {}


def _dtlz7_front(problem, n_points: int = 20_000) -> np.ndarray:
    """Exact DTLZ7 front: nondominated g=1 surface on a deterministic grid."""

    n_obj = int(problem.n_obj)
    key = (n_obj, int(problem.n_var))
    if key in _DTLZ7_FRONTS:
        return _DTLZ7_FRONTS[key].copy()
    per_axis = max(2, int(np.ceil(n_points ** (1.0 / (n_obj - 1)))))
    axis = np.linspace(0.0, 1.0, per_axis)
    grid = np.stack(np.meshgrid(*([axis] * (n_obj - 1)), indexing="ij"), axis=-1)
    decisions = np.zeros((grid.size // (n_obj - 1), int(problem.n_var)), dtype=float)
    decisions[:, : n_obj - 1] = grid.reshape(-1, n_obj - 1)
    values = np.asarray(problem.evaluate(decisions, return_values_of=["F"]), dtype=float)
    _DTLZ7_FRONTS[key] = values[_nondominated_mask(values)]
    return _DTLZ7_FRONTS[key].copy()


def _pareto_front(problem, *args, **kwargs) -> np.ndarray:
    """Return pymoo's front or the exact DTLZ5/6/7 fronts it omits or downloads."""

    name = problem.__class__.__name__.lower()
    if name.startswith("wfg") and int(problem.n_obj) > 5:
        ref_dirs = kwargs.get("ref_dirs")
        if ref_dirs is None and args:
            ref_dirs = args[0]
        if ref_dirs is None:
            from .metrics import metric_reference_directions

            ref_dirs = metric_reference_directions(int(problem.n_obj))
        decisions = _wfg_reference_decisions(problem, ref_dirs)
        return np.asarray(
            problem.evaluate(decisions, return_values_of=["F"]), dtype=float
        )
    if name in {"dtlz5", "dtlz6"}:
        ref_dirs = kwargs.get("ref_dirs")
        if ref_dirs is None and args:
            ref_dirs = args[0]
        n_points = 512 if ref_dirs is None else max(2, len(ref_dirs))
        decisions = np.full((n_points, int(problem.n_var)), 0.5, dtype=float)
        decisions[:, 0] = np.linspace(0.0, 1.0, n_points)
        if name == "dtlz6":
            decisions[:, int(problem.n_obj) - 1 :] = 0.0
        return np.asarray(
            problem.evaluate(decisions, return_values_of=["F"]), dtype=float
        )
    if name == "dtlz7":
        return _dtlz7_front(problem)
    return np.asarray(problem.pareto_front(*args, **kwargs), dtype=float)


def _reference_front(problem, *, seed: int) -> np.ndarray:
    from .metrics import isolated_numpy_seed, metric_reference_directions

    directions = metric_reference_directions(int(problem.n_obj))
    with isolated_numpy_seed(seed):
        try:
            front = _pareto_front(problem, ref_dirs=directions)
        except TypeError:
            front = _pareto_front(problem)
    values = np.asarray(front, dtype=float)
    if values.ndim != 2 or values.shape[1] != int(problem.n_obj):
        raise ValueError("benchmark did not provide a valid reference front")
    if not np.all(np.isfinite(values)):
        raise ValueError("benchmark reference front contains non-finite values")
    return values


def _rotation_matrix(transform: dict[str, Any], n_obj: int) -> np.ndarray:
    keys = {key for key in transform if key != "kind"}
    if len(keys) != 1 or not keys <= {"seed", "angle_degrees", "matrix"}:
        raise ValueError("rotation requires exactly one of seed, angle_degrees, or matrix")
    if "seed" in transform:
        seed = transform["seed"]
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("rotation seed must be an integer")
        matrix = orthonormal_rotation(n_obj, seed)
    elif "angle_degrees" in transform:
        angle = transform["angle_degrees"]
        if not isinstance(angle, (int, float)) or isinstance(angle, bool) or not np.isfinite(angle):
            raise ValueError("rotation angle must be finite")
        radians = np.deg2rad(float(angle))
        matrix = np.eye(n_obj)
        matrix[:2, :2] = [
            [np.cos(radians), -np.sin(radians)],
            [np.sin(radians), np.cos(radians)],
        ]
    else:
        matrix = np.asarray(transform["matrix"], dtype=float)
    if matrix.shape != (n_obj, n_obj) or not np.all(np.isfinite(matrix)):
        raise ValueError("rotation matrix has invalid dimensions or values")
    if not np.allclose(matrix @ matrix.T, np.eye(n_obj), atol=1e-12, rtol=1e-12):
        raise ValueError("rotation matrix must be orthonormal")
    return matrix


class ObjectiveTransformProblem(Problem):
    def __init__(
        self,
        base_problem,
        *,
        kind: str,
        matrix: np.ndarray,
        center: np.ndarray,
        offset: np.ndarray,
        reference_seed: int,
    ) -> None:
        super().__init__(
            n_var=base_problem.n_var,
            n_obj=base_problem.n_obj,
            n_ieq_constr=base_problem.n_ieq_constr,
            n_eq_constr=base_problem.n_eq_constr,
            xl=np.asarray(base_problem.xl, dtype=float),
            xu=np.asarray(base_problem.xu, dtype=float),
            vtype=base_problem.vtype,
        )
        self.base_problem = base_problem
        self._transform_matrix = np.asarray(matrix, dtype=float)
        self._transform_center = np.asarray(center, dtype=float)
        self._transform_offset = np.asarray(offset, dtype=float)
        self._reference_seed = int(reference_seed)
        payload = {
            "kind": kind,
            "matrix": self._transform_matrix.tolist(),
            "center": self._transform_center.tolist(),
            "offset": self._transform_offset.tolist(),
        }
        self.objective_transform = {
            **payload,
            "matrix": self._transform_matrix.copy(),
            "center": self._transform_center.copy(),
            "offset": self._transform_offset.copy(),
            "hash": canonical_sha256(payload),
        }

    def name(self):
        return self.base_problem.name()

    def transform_objectives(self, objectives: np.ndarray) -> np.ndarray:
        values = np.asarray(objectives, dtype=float)
        if values.ndim != 2 or values.shape[1] != self.n_obj:
            raise ValueError("objective matrix has the wrong dimension")
        if not np.all(np.isfinite(values)):
            raise ValueError("objective matrix must be finite")
        return (
            (values - self._transform_center) @ self._transform_matrix.T
            + self._transform_center
            + self._transform_offset
        )

    def inverse_transform_objectives(self, objectives: np.ndarray) -> np.ndarray:
        values = np.asarray(objectives, dtype=float)
        if values.ndim != 2 or values.shape[1] != self.n_obj:
            raise ValueError("objective matrix has the wrong dimension")
        if not np.all(np.isfinite(values)):
            raise ValueError("objective matrix must be finite")
        return (
            (values - self._transform_center - self._transform_offset)
            @ self._transform_matrix
            + self._transform_center
        )

    def _evaluate(self, x, out, *args, **kwargs):
        evaluated = self.base_problem.evaluate(x, return_as_dictionary=True)
        out.update(evaluated)
        out["F"] = self.transform_objectives(evaluated["F"])

    def _calc_pareto_front(self, *args, **kwargs):
        front = _pareto_front(self.base_problem, *args, **kwargs)
        return self.transform_objectives(front)


def build_problem(spec: ProblemSpec | RunSpec) -> ObjectiveTransformProblem:
    problem_spec = ProblemSpec.from_run_spec(spec) if isinstance(spec, RunSpec) else spec
    if not isinstance(problem_spec, ProblemSpec):
        raise TypeError("spec must be a ProblemSpec or RunSpec")
    if problem_spec.n_obj < 2 or problem_spec.n_var < 1:
        raise ValueError("problem dimensions are invalid")
    transform = deepcopy(problem_spec.transform)
    if not isinstance(transform, dict) or transform.get("kind") not in {
        "identity",
        "rotation",
    }:
        raise ValueError("unknown objective transform")

    base = _base_problem(problem_spec)
    reference_seed = reference_geometry_seed(problem_spec)
    if base.n_obj != problem_spec.n_obj or base.n_var != problem_spec.n_var:
        raise ValueError("constructed benchmark dimensions do not match the spec")
    if transform["kind"] == "identity":
        if set(transform) != {"kind"}:
            raise ValueError("identity transform accepts only its kind")
        matrix = np.eye(problem_spec.n_obj)
        center = np.zeros(problem_spec.n_obj)
        offset = np.zeros(problem_spec.n_obj)
    else:
        matrix = _rotation_matrix(transform, problem_spec.n_obj)
        reference = _reference_front(base, seed=reference_seed)
        ideal = np.min(reference, axis=0)
        nadir = np.max(reference, axis=0)
        center = (ideal + nadir) / 2.0
        unshifted = (reference - center) @ matrix.T + center
        offset = np.maximum(0.0, -np.min(unshifted, axis=0))
    return ObjectiveTransformProblem(
        base,
        kind=transform["kind"],
        matrix=matrix,
        center=center,
        offset=offset,
        reference_seed=reference_seed,
    )
