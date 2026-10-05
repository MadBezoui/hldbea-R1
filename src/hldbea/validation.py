"""Dataset-level validation gates for immutable experiment artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping

from .artifacts import artifact_path, artifact_status, validate_run_artifact
from .metrics import build_reference_geometry
from .problems import ProblemSpec
from .run_spec import RunSpec, expand_manifest, load_manifest


ISSUE_CATEGORIES = (
    "missing_runs",
    "failed_runs",
    "duplicate_runs",
    "unexpected_runs",
    "invalid_runs",
    "seed_pairing",
    "fe_violations",
    "geometry_mismatches",
    "nonfinite_metrics",
    "event_inconsistencies",
    "dependency_gates",
)
REQUIRED_ENVIRONMENT = (
    "python",
    "numpy",
    "scipy",
    "pandas",
    "pymoo",
    "matplotlib",
    "pyyaml",
    "numba",
    "llvmlite",
)
EXACT_DEPENDENCY_VERSIONS = {
    "pymoo": "0.6.1.3",
    "scipy": "1.14.1",
    "numba": "0.61.2",
}
RESERVED_VALIDATION_SEEDS = frozenset(range(51001, 51031))
RESERVED_VALIDATION_PROBLEMS = frozenset(
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


@dataclass(frozen=True)
class DatasetReport:
    manifest_id: str
    stage: str
    artifact_root: str
    planned: int
    completed: int
    valid: int
    invalid: int
    failed: int
    missing: int
    duplicate: int
    issues: dict[str, tuple[str, ...]]
    gates: dict[str, bool]
    evidence: dict[str, Any]

    def passes(self, gate: str = "analysis") -> bool:
        if gate not in self.gates:
            raise ValueError(f"unknown validation gate: {gate}")
        return self.gates[gate]

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_id": self.manifest_id,
            "stage": self.stage,
            "artifact_root": self.artifact_root,
            "counts": {
                "planned": self.planned,
                "completed": self.completed,
                "valid": self.valid,
                "invalid": self.invalid,
                "failed": self.failed,
                "missing": self.missing,
                "duplicate": self.duplicate,
            },
            "issues": {
                category: list(self.issues[category]) for category in ISSUE_CATEGORIES
            },
            "gates": dict(self.gates),
            "evidence": self.evidence,
        }

    def to_markdown(self) -> str:
        status = lambda value: "PASS" if value else "FAIL"
        lines = [
            "# Dataset validation",
            "",
            f"- Manifest: `{self.manifest_id}`",
            f"- Stage: `{self.stage}`",
            f"- Artifact root: `{self.artifact_root}`",
            f"- Artifact gate: **{status(self.gates['artifacts'])}**",
            f"- Analysis gate: **{status(self.gates['analysis'])}**",
            "",
            "| Planned | Completed | Valid | Invalid | Failed | Missing | Duplicate |",
            "|---:|---:|---:|---:|---:|---:|---:|",
            (
                f"| {self.planned} | {self.completed} | {self.valid} | "
                f"{self.invalid} | {self.failed} | {self.missing} | "
                f"{self.duplicate} |"
            ),
            "",
            "## Findings",
            "",
        ]
        for category in ISSUE_CATEGORIES:
            findings = self.issues[category]
            lines.append(f"### {category.replace('_', ' ').title()}")
            lines.append("")
            if findings:
                lines.extend(f"- {finding}" for finding in findings)
            else:
                lines.append("- None")
            lines.append("")
        lines.extend(
            [
                "## Evidence index",
                "",
                f"- Stable geometry hashes: {len(self.evidence['geometry_hashes'])}",
                f"- Paired seed groups: {len(self.evidence['paired_seeds'])}",
                f"- Valid run IDs: {len(self.evidence['valid_run_ids'])}",
                "",
            ]
        )
        return "\n".join(lines).rstrip() + "\n"


def _load_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _nonnegative_integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _append(issues: dict[str, list[str]], category: str, message: str) -> None:
    issues[category].append(message)


def validate_calibration_boundary(document: Mapping[str, Any]) -> tuple[str, ...]:
    """Return tuning/validation boundary violations without reading artifacts."""

    violations: list[str] = []
    if document.get("stage") != "calibration":
        violations.append("manifest stage must be calibration")
    seeds = document.get("seeds")
    if not isinstance(seeds, list):
        violations.append("calibration seeds must be a list")
    else:
        overlap = {
            seed for seed in seeds if isinstance(seed, int)
        } & RESERVED_VALIDATION_SEEDS
        if overlap:
            violations.append(f"validation seed overlap: {sorted(overlap)}")
    problems = document.get("problems")
    if not isinstance(problems, list):
        violations.append("calibration problems must be a list")
    else:
        overlap = sorted(
            {
                problem.get("name")
                for problem in problems
                if isinstance(problem, Mapping)
                and problem.get("name") in RESERVED_VALIDATION_PROBLEMS
            }
        )
        if overlap:
            violations.append(f"validation problem overlap: {overlap}")
    return tuple(violations)


def _active_occurrences(
    manifest_root: Path,
) -> tuple[dict[str, list[Path]], list[str]]:
    occurrences: dict[str, list[Path]] = {}
    unexpected_unreadable: list[str] = []
    if not manifest_root.is_dir():
        return occurrences, unexpected_unreadable
    reserved = {"failures", "invalid"}
    for candidate in sorted(manifest_root.iterdir(), key=lambda path: path.name):
        if not candidate.is_dir() or candidate.name in reserved or candidate.name.startswith("."):
            continue
        metadata = _load_json(candidate / "metadata.json")
        if not isinstance(metadata, dict) or not isinstance(metadata.get("run_id"), str):
            unexpected_unreadable.append(str(candidate))
            continue
        occurrences.setdefault(metadata["run_id"], []).append(candidate)
    return occurrences, unexpected_unreadable


def _check_function_evaluations(
    spec: RunSpec,
    run: Mapping[str, Any],
    checkpoints: Any,
    issues: dict[str, list[str]],
) -> None:
    evaluations = run.get("evaluations")
    if not isinstance(evaluations, Mapping):
        _append(issues, "fe_violations", f"{spec.run_id}: evaluations are missing")
        return
    total = evaluations.get("total")
    evolutionary = evaluations.get("evolutionary")
    solver = evaluations.get("solver")
    if not all(_nonnegative_integer(value) for value in (total, evolutionary, solver)):
        _append(issues, "fe_violations", f"{spec.run_id}: FE counters are invalid")
        return
    if total != evolutionary + solver:
        _append(
            issues,
            "fe_violations",
            f"{spec.run_id}: total FE does not equal evolutionary plus solver FE",
        )
    if total > spec.evaluation_budget:
        _append(
            issues,
            "fe_violations",
            f"{spec.run_id}: {total} FE exceeds budget {spec.evaluation_budget}",
        )
    if not isinstance(checkpoints, list):
        _append(issues, "fe_violations", f"{spec.run_id}: checkpoints are unreadable")
        return
    previous_target = -1
    previous_observation = -1
    for index, checkpoint in enumerate(checkpoints):
        if not isinstance(checkpoint, Mapping):
            _append(
                issues,
                "fe_violations",
                f"{spec.run_id}: checkpoint {index} is not a mapping",
            )
            continue
        observation = checkpoint.get("evaluation")
        target = checkpoint.get("target_evaluation", observation)
        if not _nonnegative_integer(observation) or not _nonnegative_integer(target):
            _append(
                issues,
                "fe_violations",
                f"{spec.run_id}: checkpoint {index} has invalid FE values",
            )
            continue
        if (
            observation < previous_observation
            or observation > total
            or target <= previous_target
            or target > observation
            or target not in spec.checkpoints
        ):
            _append(
                issues,
                "fe_violations",
                f"{spec.run_id}: checkpoint {index} violates FE ordering or bounds",
            )
        previous_target = target
        previous_observation = observation


def _check_metrics(
    spec: RunSpec,
    run: Mapping[str, Any],
    checkpoints: Any,
    issues: dict[str, list[str]],
) -> None:
    collections = [
        ("final metrics", run.get("metrics"), ("hv", "igd_plus")),
        (
            "score diagnostics",
            run.get("score_diagnostics"),
            ("phi_nz", "r_fz"),
        ),
    ]
    for label, values, keys in collections:
        if not isinstance(values, Mapping) or any(
            not _finite_number(values.get(key)) for key in keys
        ):
            _append(
                issues,
                "nonfinite_metrics",
                f"{spec.run_id}: {label} are missing or non-finite",
            )
    if isinstance(checkpoints, list):
        for index, checkpoint in enumerate(checkpoints):
            if not isinstance(checkpoint, Mapping) or any(
                not _finite_number(checkpoint.get(key))
                for key in ("hv", "igd_plus", "phi_nz", "r_fz")
            ):
                _append(
                    issues,
                    "nonfinite_metrics",
                    f"{spec.run_id}: checkpoint {index} metrics are missing or non-finite",
                )


def _check_events(
    spec: RunSpec,
    run: Mapping[str, Any],
    events: Any,
    issues: dict[str, list[str]],
) -> None:
    if not isinstance(events, Mapping):
        _append(issues, "event_inconsistencies", f"{spec.run_id}: events are unreadable")
        return
    local = events.get("local_search")
    restart = events.get("restart")
    if not isinstance(local, list) or not isinstance(restart, list):
        _append(
            issues,
            "event_inconsistencies",
            f"{spec.run_id}: event collections must be lists",
        )
        return
    total = (run.get("evaluations") or {}).get("total")
    invalid_charge = False
    previous_after = -1
    local_gain = 0.0
    for event in local:
        if not isinstance(event, Mapping):
            invalid_charge = True
            continue
        evaluations = event.get("evaluations")
        before = event.get("evaluation_before")
        after = event.get("evaluation_after")
        gain = event.get("hv_gain")
        efficiency = event.get("gain_per_evaluation")
        valid_numbers = (
            _nonnegative_integer(evaluations)
            and _nonnegative_integer(before)
            and _nonnegative_integer(after)
            and _finite_number(gain)
            and float(gain) >= 0.0
            and _finite_number(efficiency)
            and float(efficiency) >= 0.0
        )
        expected_efficiency = (
            float(gain) / evaluations
            if valid_numbers and evaluations
            else 0.0
        )
        if (
            not valid_numbers
            or after - before != evaluations
            or before < previous_after
            or not _nonnegative_integer(total)
            or after > total
            or (not event.get("accepted") and float(gain) > 0.0)
            or not math.isclose(
                float(efficiency), expected_efficiency, rel_tol=1e-12, abs_tol=1e-15
            )
        ):
            invalid_charge = True
        else:
            local_gain += float(gain)
            previous_after = after
    charged = (
        -1
        if invalid_charge
        else sum(int(event["evaluations"]) for event in local)
    )
    solver = (run.get("evaluations") or {}).get("solver")
    summary = run.get("solver_summary")
    if (
        invalid_charge
        or charged != solver
        or not isinstance(summary, Mapping)
        or summary.get("calls") != len(local)
        or summary.get("charged_evaluations") != charged
        or not _finite_number(summary.get("accepted_hv_gain"))
        or not math.isclose(
            float(summary.get("accepted_hv_gain", -1.0)),
            local_gain,
            rel_tol=1e-12,
            abs_tol=1e-15,
        )
        or not _finite_number(summary.get("gain_per_evaluation"))
        or not math.isclose(
            float(summary.get("gain_per_evaluation", -1.0)),
            local_gain / charged if charged else 0.0,
            rel_tol=1e-12,
            abs_tol=1e-15,
        )
    ):
        _append(
            issues,
            "event_inconsistencies",
            f"{spec.run_id}: local-search events disagree with FE counters or summary",
        )
    if any(
        event.get("accepted") and not event.get("feasible")
        for event in local
        if isinstance(event, Mapping)
    ):
        _append(
            issues,
            "event_inconsistencies",
            f"{spec.run_id}: an infeasible local-search event is marked accepted",
        )
    restart_summary = run.get("restart_summary")
    triggered = sum(
        event.get("status") == "triggered"
        for event in restart
        if isinstance(event, Mapping)
    )
    invalid_restart = False
    replacements = 0
    for event in restart:
        if not isinstance(event, Mapping):
            invalid_restart = True
            continue
        evaluation = event.get("evaluation")
        evaluation_after = event.get("evaluation_after")
        replacement_count = event.get("replacement_count")
        status = event.get("status")
        if (
            not _nonnegative_integer(evaluation)
            or not _nonnegative_integer(evaluation_after)
            or evaluation_after < evaluation
            or not _nonnegative_integer(total)
            or evaluation_after > total
            or not _nonnegative_integer(event.get("generation"))
            or not _nonnegative_integer(replacement_count)
        ):
            invalid_restart = True
        if status in {"observed", "triggered", "skipped_budget"} and (
            not _finite_number(event.get("hv"))
            or not _finite_number(event.get("zero_fraction"))
        ):
            invalid_restart = True
        replacements += replacement_count if _nonnegative_integer(replacement_count) else 0
    if (
        invalid_restart
        or not isinstance(restart_summary, Mapping)
        or restart_summary.get("checks") != len(restart)
        or restart_summary.get("triggered") != triggered
        or restart_summary.get("replacements") != replacements
    ):
        _append(
            issues,
            "event_inconsistencies",
            f"{spec.run_id}: restart events disagree with their summary",
        )


def _check_dependencies(
    spec: RunSpec,
    run: Mapping[str, Any],
    environments: dict[str, dict[str, str]],
    issues: dict[str, list[str]],
) -> None:
    environment = run.get("environment")
    if not isinstance(environment, Mapping):
        _append(
            issues,
            "dependency_gates",
            f"{spec.run_id}: environment metadata is missing",
        )
        return
    normalized = {key: str(environment.get(key, "missing")) for key in REQUIRED_ENVIRONMENT}
    environments[spec.run_id] = normalized
    for dependency, version in normalized.items():
        if version in {"missing", "not-installed", "unavailable", ""}:
            _append(
                issues,
                "dependency_gates",
                f"{spec.run_id}: dependency {dependency} is unavailable",
            )
    for dependency, expected in EXACT_DEPENDENCY_VERSIONS.items():
        if normalized[dependency] != expected:
            _append(
                issues,
                "dependency_gates",
                f"{spec.run_id}: {dependency}={normalized[dependency]} but {expected} is pinned",
            )
    algorithm = run.get("algorithm")
    if not isinstance(algorithm, Mapping) or algorithm.get("available") is not True:
        _append(
            issues,
            "dependency_gates",
            f"{spec.run_id}: algorithm availability was not established",
        )


def validate_dataset(
    manifest: str | Path | Mapping[str, Any], artifact_root: str | Path
) -> DatasetReport:
    document = load_manifest(manifest) if isinstance(manifest, (str, Path)) else dict(manifest)
    specs = sorted(expand_manifest(document), key=lambda spec: spec.run_id)
    root = Path(artifact_root)
    issues: dict[str, list[str]] = {category: [] for category in ISSUE_CATEGORIES}
    statuses = {spec.run_id: artifact_status(root, spec) for spec in specs}
    valid_ids = {run_id for run_id, status in statuses.items() if status == "valid"}

    for run_id, status in statuses.items():
        if status == "missing":
            _append(issues, "missing_runs", run_id)
        elif status == "failed":
            _append(issues, "failed_runs", run_id)

    occurrences, unreadable = _active_occurrences(root / document["manifest_id"])
    expected_ids = set(statuses)
    duplicate = 0
    for run_id, paths in occurrences.items():
        if run_id not in expected_ids:
            _append(
                issues,
                "unexpected_runs",
                f"{run_id}: {', '.join(str(path) for path in paths)}",
            )
        if len(paths) > 1:
            duplicate += len(paths) - 1
            _append(
                issues,
                "duplicate_runs",
                f"{run_id}: {len(paths)} active artifact directories",
            )
    for path in unreadable:
        _append(issues, "unexpected_runs", f"unreadable artifact directory: {path}")

    expected_seeds = {spec.seed for spec in specs}
    pairing: dict[tuple[str, str, str], set[int]] = {}
    for spec in specs:
        key = (spec.problem_id, spec.algorithm_id, spec.variant)
        pairing.setdefault(key, set())
        if spec.run_id in valid_ids:
            pairing[key].add(spec.seed)
    for key, observed in sorted(pairing.items()):
        if observed != expected_seeds:
            _append(
                issues,
                "seed_pairing",
                f"{'/'.join(key)}: valid seeds {sorted(observed)}, expected {sorted(expected_seeds)}",
            )

    geometry_cache: dict[str, dict[str, Any]] = {}
    environments: dict[str, dict[str, str]] = {}
    evaluation_totals: dict[str, int] = {}
    for spec in specs:
        path = artifact_path(root, spec)
        if not path.is_dir():
            continue
        validation = validate_run_artifact(path, spec)
        if not validation.valid:
            _append(
                issues,
                "invalid_runs",
                f"{spec.run_id}: {'; '.join(validation.errors)}",
            )
        envelope = _load_json(path / "metadata.json")
        events = _load_json(path / "events.json")
        checkpoints = _load_json(path / "checkpoints.json")
        if not isinstance(envelope, Mapping) or not isinstance(
            envelope.get("run_metadata"), Mapping
        ):
            continue
        run = envelope["run_metadata"]
        total = (run.get("evaluations") or {}).get("total")
        if _nonnegative_integer(total):
            evaluation_totals[spec.run_id] = total
        _check_function_evaluations(spec, run, checkpoints, issues)
        _check_metrics(spec, run, checkpoints, issues)
        _check_events(spec, run, events, issues)
        _check_dependencies(spec, run, environments, issues)

        if spec.problem_id not in geometry_cache:
            geometry_cache[spec.problem_id] = build_reference_geometry(
                ProblemSpec.from_run_spec(spec)
            ).metadata()
        expected_geometry = geometry_cache[spec.problem_id]
        actual_geometry = run.get("geometry")
        if not isinstance(actual_geometry, Mapping) or any(
            actual_geometry.get(key) != value
            for key, value in expected_geometry.items()
        ):
            _append(
                issues,
                "geometry_mismatches",
                f"{spec.run_id}: stored reference geometry does not match the manifest",
            )

    if environments:
        baseline_id = sorted(environments)[0]
        baseline = environments[baseline_id]
        for run_id in sorted(environments):
            if environments[run_id] != baseline:
                _append(
                    issues,
                    "dependency_gates",
                    f"{run_id}: dependency versions differ from {baseline_id}",
                )

    frozen_issues = {
        category: tuple(sorted(set(issues[category])))
        for category in ISSUE_CATEGORIES
    }
    valid = sum(status == "valid" for status in statuses.values())
    invalid = sum(status == "invalid" for status in statuses.values())
    failed = sum(status == "failed" for status in statuses.values())
    missing = sum(status == "missing" for status in statuses.values())
    artifact_gate = (
        valid == len(specs)
        and invalid == 0
        and failed == 0
        and missing == 0
        and duplicate == 0
        and not frozen_issues["unexpected_runs"]
    )
    analysis_gate = artifact_gate and all(
        not findings for findings in frozen_issues.values()
    )
    evidence = {
        "valid_run_ids": sorted(valid_ids),
        "geometry_hashes": {
            problem_id: metadata["geometry_hash"]
            for problem_id, metadata in sorted(geometry_cache.items())
        },
        "paired_seeds": {
            "/".join(key): sorted(seeds) for key, seeds in sorted(pairing.items())
        },
        "evaluation_totals": dict(sorted(evaluation_totals.items())),
        "dependency_versions": (
            environments[sorted(environments)[0]] if environments else {}
        ),
    }
    return DatasetReport(
        manifest_id=document["manifest_id"],
        stage=document["stage"],
        artifact_root=str(root),
        planned=len(specs),
        completed=valid + invalid,
        valid=valid,
        invalid=invalid,
        failed=failed,
        missing=missing,
        duplicate=duplicate,
        issues=frozen_issues,
        gates={"artifacts": artifact_gate, "analysis": analysis_gate},
        evidence=evidence,
    )
