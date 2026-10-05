"""Audited IMOP benchmark subset for the reviewer validation suite.

The formulas are ports of IMOP3, IMOP4, and IMOP7 from PlatEMO commit
``d25e65d1ffba58dbf4d7e1b5259786187d12968a``. These three unconstrained
problems expose disconnected, degenerate, and irregular spherical fronts.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pymoo.core.problem import Problem
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


PLATEMO_COMMIT = "d25e65d1ffba58dbf4d7e1b5259786187d12968a"
SUPPORTED_IMOP = {"imop3": 2, "imop4": 3, "imop7": 3}


class IMOPProblem(Problem):
    def __init__(
        self,
        variant: str,
        *,
        n_var: int = 10,
        a1: float = 0.05,
        a2: float = 10.0,
        k: int = 5,
    ) -> None:
        if variant not in SUPPORTED_IMOP:
            raise ValueError(f"unsupported IMOP variant: {variant}")
        if not isinstance(k, int) or isinstance(k, bool) or k < 2 or k >= n_var:
            raise ValueError("IMOP k must be an integer in [2, n_var)")
        if not np.isfinite(a1) or a1 <= 0:
            raise ValueError("IMOP a1 must be finite and positive")
        if not np.isfinite(a2) or a2 <= 0:
            raise ValueError("IMOP a2 must be finite and positive")
        super().__init__(
            n_var=n_var,
            n_obj=SUPPORTED_IMOP[variant],
            xl=np.zeros(n_var),
            xu=np.ones(n_var),
        )
        self.variant = variant
        self.a1 = float(a1)
        self.a2 = float(a2)
        self.k = int(k)

    def name(self):
        return self.variant.upper()

    def _coordinates(self, decisions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        y1 = np.mean(decisions[:, 0 : self.k : 2], axis=1) ** self.a1
        y2 = np.mean(decisions[:, 1 : self.k : 2], axis=1) ** self.a2
        return y1, y2

    def _evaluate(self, x, out, *args, **kwargs):
        decisions = np.asarray(x, dtype=float)
        g = np.sum((decisions[:, self.k :] - 0.5) ** 2, axis=1)
        if self.variant == "imop3":
            y1 = np.mean(decisions[:, : self.k], axis=1) ** self.a1
            out["F"] = np.column_stack(
                (g + 1.0 + np.cos(10.0 * np.pi * y1) / 5.0 - y1, g + y1)
            )
            return

        y1, y2 = self._coordinates(decisions)
        if self.variant == "imop4":
            out["F"] = np.column_stack(
                (
                    (1.0 + g) * y1,
                    (1.0 + g) * (y1 + np.sin(10.0 * np.pi * y1) / 10.0),
                    (1.0 + g) * (1.0 - y1),
                )
            )
            return

        objectives = np.column_stack(
            (
                (1.0 + g) * np.cos(y1 * np.pi / 2.0) * np.cos(y2 * np.pi / 2.0),
                (1.0 + g) * np.cos(y1 * np.pi / 2.0) * np.sin(y2 * np.pi / 2.0),
                (1.0 + g) * np.sin(y1 * np.pi / 2.0),
            )
        )
        separation = np.minimum.reduce(
            (
                np.abs(objectives[:, 0] - objectives[:, 1]),
                np.abs(objectives[:, 1] - objectives[:, 2]),
                np.abs(objectives[:, 2] - objectives[:, 0]),
            )
        )
        out["F"] = objectives + (10.0 * np.maximum(0.0, separation - 0.1))[:, None]

    def _calc_pareto_front(self, *args, **kwargs):
        if self.variant == "imop3":
            y1 = np.linspace(0.0, 1.0, 4096)
            front = np.column_stack(
                (1.0 + np.cos(10.0 * np.pi * y1) / 5.0 - y1, y1)
            )
            indices = NonDominatedSorting().do(
                front, only_non_dominated_front=True
            )
            return front[indices]
        if self.variant == "imop4":
            y1 = np.linspace(0.0, 1.0, 2048)
            return np.column_stack(
                (y1, y1 + np.sin(10.0 * np.pi * y1) / 10.0, 1.0 - y1)
            )

        angles = np.linspace(0.0, np.pi / 2.0, 96)
        alpha, beta = np.meshgrid(angles, angles, indexing="ij")
        front = np.column_stack(
            (
                (np.sin(alpha) * np.cos(beta)).ravel(),
                (np.sin(alpha) * np.sin(beta)).ravel(),
                (np.cos(alpha) * np.ones_like(beta)).ravel(),
            )
        )
        separation = np.minimum.reduce(
            (
                np.abs(front[:, 0] - front[:, 1]),
                np.abs(front[:, 1] - front[:, 2]),
                np.abs(front[:, 2] - front[:, 0]),
            )
        )
        return front[separation <= 0.1]


def build_imop_problem(
    name: str, *, n_var: int, parameters: dict[str, Any]
) -> IMOPProblem:
    allowed = {"a1", "a2", "k"}
    unknown = set(parameters) - allowed
    if unknown:
        raise ValueError(f"unknown IMOP parameters: {sorted(unknown)}")
    return IMOPProblem(name, n_var=n_var, **parameters)
