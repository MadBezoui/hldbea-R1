"""Fail-closed audit of the prespecified calibration design."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Any

from .run_spec import RunSpec, expand_manifest, load_manifest


CALIBRATION_SEEDS = (41001, 41002, 41003)
VALIDATION_SEEDS = frozenset(range(51001, 51031))
VALIDATION_PROBLEMS = frozenset(
    {
        "dtlz2",
        "dtlz3",
        "dtlz4",
        "dtlz7",
        "wfg2",
        "wfg3",
        "wfg9",
        "maf",
        "imop3",
        "imop4",
        "imop7",
        "rdtlz2",
        "rlinear",
    }
)
ROLE_BY_MANIFEST = {
    "calibration-core-screen-v1": "core",
    "calibration-restart-screen-v1": "restart",
    "calibration-cone-screen-v1": "cone",
    "calibration-core-screen-v3": "core",
    "calibration-restart-screen-v3": "restart",
    "calibration-cone-screen-v3": "cone",
}
EXPECTED_PROBLEMS = {
    "core": {
        "calibration-dtlz1-m3",
        "calibration-dtlz5-m5",
        "calibration-wfg1-m5",
    },
    "restart": {
        "calibration-zdt3-m2",
        "calibration-dtlz6-m5",
        "calibration-wfg4-m5",
    },
    "cone": {
        f"calibration-{family}-m{n_obj}"
        for family in ("dtlz1", "wfg1")
        for n_obj in (5, 8, 10, 15)
    },
}
FACTOR_GRIDS = {
    "core": {
        "lambda": (0.0, 0.01, 0.05, 0.1, 0.2, 0.5),
        "k": (0.5, 1.0, 2.0),
        "selection_mode": ("local", "global"),
        "local_search_max_iter": (0, 2),
        "objective_policy": ("round_robin", "adaptive", "random"),
    },
    "restart": {
        "restart": (False, True),
        "restart_theta_zero": (0.8, 0.9, 0.95, 0.99),
        "restart_window": (10, 20, 30, 50),
        "restart_delta_hv": (0.00001, 0.0001, 0.001),
        "restart_fraction": (0.1, 0.2, 0.3, 0.5),
    },
    "cone": {
        "schedule_kinds": ("fixed", "linear", "saturating"),
    },
}


V3_PARAMETERS = {
    "local_search_acceptance": "feasible_improvement",
    "mutation_scope": "per_variable",
}


def _baseline(*, cone: Any = 0.0, v3: bool = False) -> dict[str, Any]:
    extra = dict(V3_PARAMETERS) if v3 else {}
    return extra | {
        "k": 1.0,
        "lambda": 0.1,
        "cone_epsilon": deepcopy(cone),
        "selection_mode": "local",
        "objective_policy": "round_robin",
        "tau": 1,
        "local_search_max_iter": 2,
        "restart": True,
        "restart_theta_zero": 0.95,
        "restart_window": 20,
        "restart_delta_hv": 0.0001,
        "restart_fraction": 0.2,
        "crossover_probability": 0.9,
        "crossover_eta": 20,
        "mutation_probability": "inverse_n_var",
        "mutation_eta": 20,
    }


def _changed(base: dict[str, Any], **changes: Any) -> dict[str, Any]:
    result = deepcopy(base)
    result.update(deepcopy(changes))
    return result


def _expected_variants(role: str, v3: bool = False) -> dict[str, dict[str, Any]]:
    if role == "core":
        base = _baseline(v3=v3)
        return {
            "baseline": base,
            "lambda-0": _changed(base, **{"lambda": 0.0}),
            "lambda-0p01": _changed(base, **{"lambda": 0.01}),
            "lambda-0p05": _changed(base, **{"lambda": 0.05}),
            "lambda-0p2": _changed(base, **{"lambda": 0.2}),
            "lambda-0p5": _changed(base, **{"lambda": 0.5}),
            "k-0p5": _changed(base, k=0.5),
            "k-2": _changed(base, k=2.0),
            "selection-global": _changed(base, selection_mode="global"),
            "no-local-search": _changed(base, local_search_max_iter=0),
            "objective-adaptive": _changed(base, objective_policy="adaptive"),
            "objective-random": _changed(base, objective_policy="random"),
        }
    if role == "restart":
        base = _baseline(v3=v3)
        return {
            "baseline": base,
            "no-restart": _changed(base, restart=False),
            "theta-0p8": _changed(base, restart_theta_zero=0.8),
            "theta-0p9": _changed(base, restart_theta_zero=0.9),
            "theta-0p99": _changed(base, restart_theta_zero=0.99),
            "window-10": _changed(base, restart_window=10),
            "window-30": _changed(base, restart_window=30),
            "window-50": _changed(base, restart_window=50),
            "delta-1e-5": _changed(base, restart_delta_hv=0.00001),
            "delta-1e-3": _changed(base, restart_delta_hv=0.001),
            "rho-0p1": _changed(base, restart_fraction=0.1),
            "rho-0p3": _changed(base, restart_fraction=0.3),
            "rho-0p5": _changed(base, restart_fraction=0.5),
        }
    base = _baseline(cone={"kind": "fixed", "value": 0.0}, v3=v3)
    return {
        "cone-strict": base,
        "cone-fixed-0p1": _changed(
            base, cone_epsilon={"kind": "fixed", "value": 0.1}
        ),
        "cone-linear": _changed(
            base,
            cone_epsilon={
                "kind": "linear",
                "slope": 0.02,
                "origin": 3,
                "cap": 0.15,
            },
        ),
        "cone-saturating": _changed(
            base,
            cone_epsilon={
                "kind": "saturating",
                "limit": 0.15,
                "rate": 0.25,
                "origin": 3,
            },
        ),
    }


def estimate_run_bytes(spec: RunSpec) -> int:
    """Conservative compact-artifact bound shared with the runner."""

    final_arrays = spec.population_size * (spec.n_var + spec.n_obj + 1) * 8
    checkpoint_payload = len(spec.checkpoints) * 2_048
    base_metadata = 32_768
    event_payload = spec.evaluation_budget * (
        2_048 if spec.algorithm_id == "hldbea" else 128
    )
    return final_arrays + checkpoint_payload + base_metadata + event_payload


@dataclass(frozen=True)
class DesignAuditReport:
    passes: bool
    violations: tuple[str, ...]
    manifest_hashes: dict[str, str]
    per_manifest: dict[str, dict[str, int]]
    run_count: int
    total_evaluations: int
    estimated_max_bytes: int
    seed_set: tuple[int, ...]
    objective_counts: tuple[int, ...]
    problem_ids: tuple[str, ...]
    factor_grids: dict[str, dict[str, tuple[Any, ...]]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def audit_calibration_design(manifests: list[Path]) -> DesignAuditReport:
    violations: list[str] = []
    manifest_hashes: dict[str, str] = {}
    per_manifest: dict[str, dict[str, int]] = {}
    all_specs: list[RunSpec] = []
    roles: list[str] = []
    seed_set: set[int] = set()
    objective_counts: set[int] = set()
    problem_ids: set[str] = set()

    for raw_path in manifests:
        path = Path(raw_path)
        try:
            manifest_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
            document = load_manifest(path)
            specs = expand_manifest(document)
        except (FileNotFoundError, OSError, TypeError, ValueError) as exc:
            violations.append(f"{path.name}: invalid manifest: {exc}")
            continue
        role = ROLE_BY_MANIFEST.get(document["manifest_id"])
        if role is None:
            violations.append(f"{path.name}: unexpected calibration manifest id")
            continue
        if role in roles:
            violations.append(f"{path.name}: duplicate {role} design role")
        roles.append(role)
        if document["stage"] != "calibration":
            violations.append(f"{path.name}: stage must be calibration")
        seeds = tuple(document["seeds"])
        if seeds != CALIBRATION_SEEDS:
            violations.append(f"{path.name}: calibration seed grid is incomplete")
        overlap = set(seeds) & VALIDATION_SEEDS
        if overlap:
            violations.append(
                f"{path.name}: validation seed overlap: {sorted(overlap)}"
            )
        execution = document["execution"]
        if execution["population_size"] != 100:
            violations.append(f"{path.name}: population size must be 100")
        if execution["evaluation_budget"] != 2000:
            violations.append(f"{path.name}: evaluation budget must be 2000")
        if execution["checkpoints"] != [100, 500, 1000, 2000]:
            violations.append(f"{path.name}: checkpoint grid is inconsistent")
        if execution["save_history"] is not False:
            violations.append(f"{path.name}: full history must be disabled")

        actual_problem_ids = {problem["id"] for problem in document["problems"]}
        if actual_problem_ids != EXPECTED_PROBLEMS[role]:
            violations.append(f"{path.name}: problem grid mismatch")
        invalid_problem_names = {
            problem["name"]
            for problem in document["problems"]
            if problem["name"] in VALIDATION_PROBLEMS
        }
        if invalid_problem_names:
            violations.append(
                f"{path.name}: validation problem overlap: {sorted(invalid_problem_names)}"
            )

        algorithms = document["algorithms"]
        if any(algorithm["id"] != "hldbea" for algorithm in algorithms):
            violations.append(f"{path.name}: unexpected algorithm in calibration")
        actual_variants = {
            algorithm["variant"]: algorithm["parameters"] for algorithm in algorithms
        }
        expected_variants = _expected_variants(
            role, v3=document["manifest_id"].endswith("-v3")
        )
        if set(actual_variants) != set(expected_variants):
            violations.append(f"{path.name}: factor grid variants mismatch")
        baseline_parameters = next(iter(expected_variants.values()))
        for variant, parameters in actual_variants.items():
            changes = {
                key
                for key in set(baseline_parameters) | set(parameters)
                if baseline_parameters.get(key) != parameters.get(key)
            }
            if len(changes) > 1:
                violations.append(
                    f"{path.name}: variant {variant} violates OFAT: {sorted(changes)}"
                )
            if expected_variants.get(variant) != parameters:
                violations.append(
                    f"{path.name}: variant {variant} has an incorrect grid definition"
                )

        run_count = len(specs)
        estimated = sum(estimate_run_bytes(spec) for spec in specs)
        per_manifest[path.name] = {
            "run_count": run_count,
            "total_evaluations": sum(spec.evaluation_budget for spec in specs),
            "estimated_max_bytes": estimated,
        }
        all_specs.extend(specs)
        seed_set.update(spec.seed for spec in specs)
        objective_counts.update(spec.n_obj for spec in specs)
        problem_ids.update(spec.problem_id for spec in specs)

    if set(roles) != set(ROLE_BY_MANIFEST.values()) or len(roles) != 3:
        violations.append("the core, restart, and cone calibration roles are required once")
    run_ids = [spec.run_id for spec in all_specs]
    if len(run_ids) != len(set(run_ids)):
        violations.append("duplicate run ids across calibration manifests")

    run_count = len(all_specs)
    total_evaluations = sum(spec.evaluation_budget for spec in all_specs)
    estimated_max_bytes = sum(estimate_run_bytes(spec) for spec in all_specs)
    return DesignAuditReport(
        passes=not violations,
        violations=tuple(violations),
        manifest_hashes=manifest_hashes,
        per_manifest=per_manifest,
        run_count=run_count,
        total_evaluations=total_evaluations,
        estimated_max_bytes=estimated_max_bytes,
        seed_set=tuple(sorted(seed_set)),
        objective_counts=tuple(sorted(objective_counts)),
        problem_ids=tuple(sorted(problem_ids)),
        factor_grids=deepcopy(FACTOR_GRIDS),
    )
