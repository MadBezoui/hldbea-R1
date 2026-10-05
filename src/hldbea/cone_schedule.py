"""Validated dimension-aware cone-relaxation schedules."""

from __future__ import annotations

import math
from typing import Any


def _finite_nonnegative(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite and numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _origin(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("cone schedule origin must be a non-negative integer")
    return value


def _require_fields(schedule: dict[str, object], expected: set[str]) -> None:
    if set(schedule) != expected:
        raise ValueError(
            "cone schedule fields must be exactly " + ", ".join(sorted(expected))
        )


def resolve_cone_epsilon(spec: float | dict[str, object], n_obj: int) -> float:
    """Resolve a declared cone schedule to one finite non-negative scalar."""

    if isinstance(n_obj, bool) or not isinstance(n_obj, int) or n_obj <= 0:
        raise ValueError("n_obj must be a positive integer")
    if isinstance(spec, bool):
        raise ValueError("cone schedule must be numeric or a mapping")
    if isinstance(spec, (int, float)):
        return _finite_nonnegative(spec, "cone schedule")
    if not isinstance(spec, dict):
        raise ValueError("cone schedule must be numeric or a mapping")

    kind = spec.get("kind")
    if kind == "fixed":
        _require_fields(spec, {"kind", "value"})
        return _finite_nonnegative(spec["value"], "cone schedule value")
    if kind == "linear":
        _require_fields(spec, {"kind", "slope", "origin", "cap"})
        slope = _finite_nonnegative(spec["slope"], "cone schedule slope")
        origin = _origin(spec["origin"])
        cap_value = spec["cap"]
        cap = (
            math.inf
            if cap_value is None
            else _finite_nonnegative(cap_value, "cone schedule cap")
        )
        return float(min(cap, slope * max(0, n_obj - origin)))
    if kind == "saturating":
        _require_fields(spec, {"kind", "limit", "rate", "origin"})
        limit = _finite_nonnegative(spec["limit"], "cone schedule limit")
        rate = _finite_nonnegative(spec["rate"], "cone schedule rate")
        if rate <= 0.0:
            raise ValueError("cone schedule rate must be positive")
        origin = _origin(spec["origin"])
        return float(limit * (1.0 - math.exp(-rate * max(0, n_obj - origin))))
    raise ValueError("unknown cone schedule kind")
