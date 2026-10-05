"""Dominance-valid DTLZ2 benchmark with a rotated Pareto-front geometry."""

from __future__ import annotations

from typing import Any

import numpy as np
from pymoo.core.problem import Problem

from .scoring import pareto_nondominated_mask


class RotatedFrontDTLZ2(Problem):
    """Rotate the DTLZ2 unit front while keeping distance dominance explicit.

    Let ``h(x)`` be the positive-orthant DTLZ2 unit direction and let ``g(x)``
    be the usual squared distance from the Pareto set.  The objectives are

    ``R(theta) h(x) + offset(theta) + g(x) * 1``.

    Consequently, every decision with ``g > 0`` is dominated by the decision
    with identical position variables and ``g = 0``.  This is deliberately
    different from rotating ``(1 + g) h`` as a whole, which need not preserve
    dominance after an arbitrary objective-space rotation.
    """

    def __init__(self, *, n_var: int, n_obj: int, angle_degrees: float) -> None:
        if not isinstance(n_obj, (int, np.integer)) or isinstance(n_obj, bool) or n_obj < 2:
            raise ValueError("n_obj must be an integer of at least two")
        if not isinstance(n_var, (int, np.integer)) or isinstance(n_var, bool) or n_var < n_obj:
            raise ValueError("n_var must be an integer no smaller than n_obj")
        if (
            not isinstance(angle_degrees, (int, float, np.integer, np.floating))
            or isinstance(angle_degrees, (bool, np.bool_))
            or not np.isfinite(angle_degrees)
            or not 0.0 <= float(angle_degrees) <= 90.0
        ):
            raise ValueError("angle_degrees must be finite and lie in [0, 90]")
        super().__init__(
            n_var=int(n_var),
            n_obj=int(n_obj),
            xl=np.zeros(int(n_var)),
            xu=np.ones(int(n_var)),
        )
        self.angle_degrees = float(angle_degrees)
        radians = np.deg2rad(self.angle_degrees)
        self.rotation = np.eye(int(n_obj), dtype=float)
        self.rotation[:2, :2] = [
            [np.cos(radians), -np.sin(radians)],
            [np.sin(radians), np.cos(radians)],
        ]
        # Minimum of r.h over the positive unit sphere is -||min(r, 0)||.
        self.offset = np.linalg.norm(np.minimum(self.rotation, 0.0), axis=1)

    def name(self) -> str:
        return "RotatedFrontDTLZ2"

    def _directions(self, x: np.ndarray) -> np.ndarray:
        values = np.asarray(x, dtype=float)
        h = np.ones((len(values), self.n_obj), dtype=float)
        for objective in range(self.n_obj):
            cosine_count = self.n_obj - objective - 1
            if cosine_count:
                h[:, objective] *= np.prod(
                    np.cos(values[:, :cosine_count] * np.pi / 2.0), axis=1
                )
            if objective:
                h[:, objective] *= np.sin(
                    values[:, self.n_obj - objective - 1] * np.pi / 2.0
                )
        return h

    def _rotate_front(self, directions: np.ndarray) -> np.ndarray:
        return np.asarray(directions, dtype=float) @ self.rotation.T + self.offset

    def _evaluate(self, x, out, *args, **kwargs) -> None:
        values = np.asarray(x, dtype=float)
        directions = self._directions(values)
        distance = np.sum(
            np.square(values[:, self.n_obj - 1 :] - 0.5), axis=1
        )
        out["F"] = self._rotate_front(directions) + distance[:, None]

    def _calc_pareto_front(self, ref_dirs=None, *args: Any, **kwargs: Any) -> np.ndarray:
        if ref_dirs is None:
            from .metrics import metric_reference_directions

            ref_dirs = metric_reference_directions(self.n_obj)
        directions = np.asarray(ref_dirs, dtype=float)
        if directions.ndim != 2 or directions.shape[1] != self.n_obj:
            raise ValueError("reference directions have invalid dimensions")
        norms = np.linalg.norm(directions, axis=1)
        if np.any(norms <= 0.0) or not np.all(np.isfinite(norms)):
            raise ValueError("reference directions must be finite and nonzero")
        front = self._rotate_front(directions / norms[:, None])
        return front[pareto_nondominated_mask(front)]


def build_rotated_front_dtlz2(
    *, n_var: int, n_obj: int, parameters: dict[str, Any]
) -> RotatedFrontDTLZ2:
    if set(parameters) != {"angle_degrees"}:
        raise ValueError("rdtlz2 parameters must contain only angle_degrees")
    return RotatedFrontDTLZ2(
        n_var=n_var,
        n_obj=n_obj,
        angle_degrees=parameters["angle_degrees"],
    )


def rotation_about_diagonal(n_obj: int, angle_degrees: float) -> np.ndarray:
    """Rotation by ``angle_degrees`` about the diagonal ``(1, ..., 1)``.

    Only three objectives are supported, where this is the unique rotation
    that maps the plane ``sum(f) = 1`` onto itself and fixes its centroid.
    """

    if n_obj != 3:
        raise ValueError("the diagonal rotation is defined for three objectives")
    axis = np.ones(3) / np.sqrt(3.0)
    theta = np.deg2rad(float(angle_degrees))
    cross = np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
    )
    return (
        np.cos(theta) * np.eye(3)
        + np.sin(theta) * cross
        + (1.0 - np.cos(theta)) * np.outer(axis, axis)
    )


class RotatedLinearFront(Problem):
    """Linear three-objective front rigidly rotated inside its own plane.

    With simplex weights ``h(x)`` (``sum(h) = 1``), the DTLZ2 distance
    ``g(x) = sum_{i >= m} (x_i - 0.5)^2`` and the rotation ``R`` about the
    diagonal, the objectives are ``c + R (h(x) - c) + g(x) 1`` with ``c`` the
    centroid ``(1/3, 1/3, 1/3)``.  Since ``R`` fixes the diagonal, every
    objective vector satisfies ``sum(f) = 1 + 3 g``.  Points with ``g = 0`` lie
    on one plane with a positive normal and are therefore mutually
    nondominated, and a point with ``g > 0`` is dominated by its counterpart
    with ``g = 0``.  The Pareto set is ``g = 0`` for every angle and the Pareto
    front is the same equilateral triangle, rotated by ``angle_degrees`` in its
    plane.  The triangle's symmetry makes 60 degrees the largest distinct
    misalignment with the coordinate axes.
    """

    def __init__(self, *, n_var: int, n_obj: int, angle_degrees: float) -> None:
        if n_obj != 3:
            raise ValueError("rlinear is defined for three objectives")
        if not isinstance(n_var, (int, np.integer)) or isinstance(n_var, bool) or n_var < n_obj:
            raise ValueError("n_var must be an integer no smaller than n_obj")
        if (
            not isinstance(angle_degrees, (int, float, np.integer, np.floating))
            or isinstance(angle_degrees, (bool, np.bool_))
            or not np.isfinite(angle_degrees)
            or not 0.0 <= float(angle_degrees) <= 60.0
        ):
            raise ValueError("angle_degrees must be finite and lie in [0, 60]")
        super().__init__(
            n_var=int(n_var), n_obj=3, xl=np.zeros(int(n_var)), xu=np.ones(int(n_var))
        )
        self.angle_degrees = float(angle_degrees)
        self.rotation = rotation_about_diagonal(3, self.angle_degrees)
        self.centroid = np.full(3, 1.0 / 3.0)

    def name(self) -> str:
        return "RotatedLinearFront"

    @staticmethod
    def _weights(x: np.ndarray) -> np.ndarray:
        values = np.asarray(x, dtype=float)
        return np.column_stack(
            (
                values[:, 0] * values[:, 1],
                values[:, 0] * (1.0 - values[:, 1]),
                1.0 - values[:, 0],
            )
        )

    def _place(self, weights: np.ndarray) -> np.ndarray:
        return self.centroid + (np.asarray(weights, dtype=float) - self.centroid) @ self.rotation.T

    def _evaluate(self, x, out, *args, **kwargs) -> None:
        values = np.asarray(x, dtype=float)
        distance = np.sum(np.square(values[:, 2:] - 0.5), axis=1)
        out["F"] = self._place(self._weights(values)) + distance[:, None]

    def _calc_pareto_front(self, ref_dirs=None, *args: Any, **kwargs: Any) -> np.ndarray:
        if ref_dirs is None:
            from .metrics import metric_reference_directions

            ref_dirs = metric_reference_directions(3)
        weights = np.asarray(ref_dirs, dtype=float)
        if weights.ndim != 2 or weights.shape[1] != 3 or np.any(weights < 0):
            raise ValueError("reference directions must be nonnegative 3-vectors")
        weights = weights / np.sum(weights, axis=1, keepdims=True)
        return self._place(weights)


def build_rotated_linear_front(
    *, n_var: int, n_obj: int, parameters: dict[str, Any]
) -> RotatedLinearFront:
    if set(parameters) != {"angle_degrees"}:
        raise ValueError("rlinear parameters must contain only angle_degrees")
    return RotatedLinearFront(
        n_var=n_var, n_obj=n_obj, angle_degrees=parameters["angle_degrees"]
    )
