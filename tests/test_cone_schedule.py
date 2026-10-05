import math

import pytest


@pytest.mark.parametrize(
    ("schedule", "n_obj", "expected"),
    [
        (0.1, 10, 0.1),
        ({"kind": "fixed", "value": 0.0}, 15, 0.0),
        (
            {"kind": "linear", "slope": 0.02, "origin": 3, "cap": 0.1},
            5,
            0.04,
        ),
        (
            {"kind": "linear", "slope": 0.02, "origin": 3, "cap": 0.1},
            10,
            0.1,
        ),
        (
            {"kind": "saturating", "limit": 0.15, "rate": 0.25, "origin": 3},
            10,
            0.12393390848243321,
        ),
    ],
)
def test_cone_schedule_resolves_declared_contract(schedule, n_obj, expected):
    from hldbea.cone_schedule import resolve_cone_epsilon

    assert resolve_cone_epsilon(schedule, n_obj) == pytest.approx(expected)


@pytest.mark.parametrize("n_obj", [3, 5, 8, 10, 15])
def test_cone_schedules_are_finite_and_nonnegative_at_required_dimensions(n_obj):
    from hldbea.cone_schedule import resolve_cone_epsilon

    schedules = [
        {"kind": "fixed", "value": 0.0},
        {"kind": "linear", "slope": 0.02, "origin": 3, "cap": 0.1},
        {"kind": "saturating", "limit": 0.15, "rate": 0.25, "origin": 3},
    ]

    for schedule in schedules:
        resolved = resolve_cone_epsilon(schedule, n_obj)
        assert math.isfinite(resolved)
        assert resolved >= 0.0


@pytest.mark.parametrize(
    ("schedule", "n_obj", "message"),
    [
        (True, 3, "schedule"),
        (-0.1, 3, "non-negative"),
        (float("inf"), 3, "finite"),
        ({"kind": "mystery", "value": 0.1}, 3, "kind"),
        ({"kind": "fixed"}, 3, "fields"),
        ({"kind": "fixed", "value": 0.1, "extra": 1}, 3, "fields"),
        ({"kind": "linear", "slope": -0.1, "origin": 3, "cap": None}, 3, "slope"),
        ({"kind": "linear", "slope": 0.1, "origin": 3, "cap": -1}, 3, "cap"),
        (
            {"kind": "saturating", "limit": 0.1, "rate": 0.0, "origin": 3},
            3,
            "rate",
        ),
        ({"kind": "fixed", "value": 0.1}, 0, "n_obj"),
    ],
)
def test_invalid_cone_schedule_fails_before_execution(schedule, n_obj, message):
    from hldbea.cone_schedule import resolve_cone_epsilon

    with pytest.raises(ValueError, match=message):
        resolve_cone_epsilon(schedule, n_obj)
