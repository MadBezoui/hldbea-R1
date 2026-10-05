"""Deterministic, leakage-resistant calibration selection."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import rankdata
import yaml

from .artifacts import artifact_path
from .design_audit import ROLE_BY_MANIFEST, audit_calibration_design
from .run_spec import canonical_sha256, expand_manifest, load_manifest
from .statistics import metric_direction
from .validation import (
    RESERVED_VALIDATION_PROBLEMS,
    RESERVED_VALIDATION_SEEDS,
    validate_calibration_boundary,
    validate_dataset,
)


SELECTION_RULE_VERSION = "percentile-rank-v1"


@dataclass(frozen=True)
class CalibrationRank:
    variant: str
    median_rank: float
    worst_quartile_rank: float
    median_runtime_seconds: float
    block_count: int
    metric_observations: int


@dataclass(frozen=True)
class CandidateConfiguration:
    variant: str
    parameters: dict[str, Any]
    sources: tuple[str, ...]


@dataclass(frozen=True)
class FrozenConfiguration:
    selection_rule_version: str
    variant: str
    parameters: dict[str, Any]
    rank: CalibrationRank


def _is_validation_problem(problem_id: str) -> bool:
    lowered = problem_id.lower()
    return any(
        lowered == family
        or lowered.startswith(f"{family}-")
        or f"-{family}-" in lowered
        for family in RESERVED_VALIDATION_PROBLEMS
    )


def rank_complete_blocks(
    records: Sequence[Mapping[str, Any]],
    metrics: tuple[str, ...] = ("hv", "igd_plus"),
) -> list[CalibrationRank]:
    """Rank variants only after converting each matched block to percentiles."""

    if not records:
        raise ValueError("calibration records must not be empty")
    if not metrics or len(set(metrics)) != len(metrics):
        raise ValueError("metrics must be a non-empty unique tuple")
    directions = {metric: metric_direction(metric) for metric in metrics}
    blocks: dict[tuple[str, int], dict[str, Mapping[str, Any]]] = {}
    runtimes: dict[str, list[float]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("each calibration record must be a mapping")
        problem_id = record.get("problem_id")
        variant = record.get("variant")
        seed = record.get("seed")
        values = record.get("metrics")
        runtime = record.get("runtime_seconds")
        if not isinstance(problem_id, str) or not problem_id:
            raise ValueError("problem_id must be a non-empty string")
        if _is_validation_problem(problem_id):
            raise ValueError(f"validation problem leaked into calibration: {problem_id}")
        if not isinstance(variant, str) or not variant:
            raise ValueError("variant must be a non-empty string")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("seed must be an integer")
        if seed in RESERVED_VALIDATION_SEEDS:
            raise ValueError(f"validation seed leaked into calibration: {seed}")
        if not isinstance(values, Mapping):
            raise ValueError("record metrics must be a mapping")
        for metric in metrics:
            value = values.get(metric)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"metric {metric} must be finite")
        if (
            isinstance(runtime, bool)
            or not isinstance(runtime, (int, float))
            or not math.isfinite(float(runtime))
            or float(runtime) < 0.0
        ):
            raise ValueError("runtime_seconds must be finite and non-negative")
        block = blocks.setdefault((problem_id, seed), {})
        if variant in block:
            raise ValueError(
                f"duplicate calibration record for {problem_id}/{seed}/{variant}"
            )
        block[variant] = record
        runtimes.setdefault(variant, []).append(float(runtime))

    expected_variants = set(next(iter(blocks.values())))
    if not expected_variants:
        raise ValueError("calibration blocks must include variants")
    if any(set(block) != expected_variants for block in blocks.values()):
        raise ValueError("calibration requires complete matched variant blocks")

    quality: dict[str, list[float]] = {variant: [] for variant in expected_variants}
    variants = sorted(expected_variants)
    for block in blocks.values():
        for metric, direction in directions.items():
            raw = np.asarray(
                [float(block[variant]["metrics"][metric]) for variant in variants]
            )
            oriented = raw if direction == "higher" else -raw
            ranks = rankdata(oriented, method="average")
            percentiles = (
                np.ones(len(variants), dtype=float)
                if len(variants) == 1
                else (ranks - 1.0) / (len(variants) - 1.0)
            )
            for variant, percentile in zip(variants, percentiles):
                quality[variant].append(float(percentile))

    ranking = []
    for variant in variants:
        scores = np.asarray(quality[variant], dtype=float)
        ranking.append(
            CalibrationRank(
                variant=variant,
                median_rank=float(np.median(scores)),
                worst_quartile_rank=float(
                    np.quantile(scores, 0.25, method="linear")
                ),
                median_runtime_seconds=float(np.median(runtimes[variant])),
                block_count=len(blocks),
                metric_observations=len(scores),
            )
        )
    return sorted(
        ranking,
        key=lambda item: (
            -item.median_rank,
            -item.worst_quartile_rank,
            item.median_runtime_seconds,
            item.variant,
        ),
    )


def select_top(ranking: Sequence[CalibrationRank], count: int) -> list[CalibrationRank]:
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValueError("top count must be a positive integer")
    if count > len(ranking):
        raise ValueError("top count exceeds available variants")
    return list(ranking[:count])


def canonical_json_text(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
    ) + "\n"


def _confirmation_document(
    candidates: Sequence[CandidateConfiguration],
) -> dict[str, Any]:
    if not candidates:
        raise ValueError("confirmation candidates must not be empty")
    variants = [candidate.variant for candidate in candidates]
    if len(variants) != len(set(variants)):
        raise ValueError("confirmation candidate variants must be unique")
    return {
        "schema_version": 1,
        "manifest_id": "calibration-confirm-v1",
        "stage": "calibration",
        "seeds": [41001, 41002, 41003, 41004, 41005],
        "problems": [
            {
                "id": "calibration-dtlz1-m3",
                "name": "dtlz1",
                "n_obj": 3,
                "n_var": 7,
                "parameters": {},
                "transform": {"kind": "identity"},
            },
            {
                "id": "calibration-dtlz5-m5",
                "name": "dtlz5",
                "n_obj": 5,
                "n_var": 14,
                "parameters": {},
                "transform": {"kind": "identity"},
            },
            {
                "id": "calibration-dtlz6-m10",
                "name": "dtlz6",
                "n_obj": 10,
                "n_var": 19,
                "parameters": {},
                "transform": {"kind": "identity"},
            },
            {
                "id": "calibration-wfg1-m5",
                "name": "wfg1",
                "n_obj": 5,
                "n_var": 24,
                "parameters": {},
                "transform": {"kind": "identity"},
            },
            {
                "id": "calibration-wfg4-m10",
                "name": "wfg4",
                "n_obj": 10,
                "n_var": 28,
                "parameters": {},
                "transform": {"kind": "identity"},
            },
        ],
        "algorithms": [
            {
                "id": "hldbea",
                "variant": candidate.variant,
                "parameters": deepcopy(candidate.parameters),
            }
            for candidate in candidates
        ],
        "execution": {
            "population_size": 100,
            "evaluation_budget": 10000,
            "checkpoints": [100, 500, 1000, 2000, 5000, 10000],
            "save_history": False,
        },
    }


def confirmation_manifest_text(
    candidates: Sequence[CandidateConfiguration],
) -> str:
    return yaml.safe_dump(
        _confirmation_document(candidates),
        sort_keys=False,
        default_flow_style=False,
    )


def _directory_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for child in sorted(path.iterdir(), key=lambda item: item.name):
        if child.is_file():
            digest.update(child.name.encode("utf-8"))
            digest.update(b"\0")
            digest.update(child.read_bytes())
            digest.update(b"\0")
    return digest.hexdigest()


def load_validated_records(
    manifest: str | Path, artifact_root: str | Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = Path(manifest)
    document = load_manifest(path)
    boundary = validate_calibration_boundary(document)
    if boundary:
        raise ValueError("; ".join(boundary))
    report = validate_dataset(path, artifact_root)
    if not report.passes("analysis"):
        raise RuntimeError("calibration dataset failed the analysis validation gate")
    records: list[dict[str, Any]] = []
    artifact_hashes: dict[str, str] = {}
    for spec in sorted(expand_manifest(document), key=lambda item: item.run_id):
        location = artifact_path(artifact_root, spec)
        envelope = json.loads((location / "metadata.json").read_text(encoding="utf-8"))
        run = envelope["run_metadata"]
        records.append(
            {
                "problem_id": spec.problem_id,
                "variant": spec.variant,
                "seed": spec.seed,
                "metrics": {
                    "hv": float(run["metrics"]["hv"]),
                    "igd_plus": float(run["metrics"]["igd_plus"]),
                },
                "runtime_seconds": float(run["runtime"]["wall_seconds"]),
            }
        )
        artifact_hashes[spec.run_id] = _directory_sha256(location)
    return records, {
        "manifest_name": path.name,
        "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "artifact_hashes": artifact_hashes,
    }


def freeze_configuration(
    records: Sequence[Mapping[str, Any]],
    candidate_parameters: Mapping[str, Mapping[str, Any]],
) -> FrozenConfiguration:
    ranking = rank_complete_blocks(records)
    winner = ranking[0]
    if set(candidate_parameters) != {item.variant for item in ranking}:
        raise ValueError("candidate parameter labels must match ranked variants")
    return FrozenConfiguration(
        selection_rule_version=SELECTION_RULE_VERSION,
        variant=winner.variant,
        parameters=deepcopy(dict(candidate_parameters[winner.variant])),
        rank=winner,
    )


def _parameter_delta(base: Mapping[str, Any], selected: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: deepcopy(value)
        for key, value in selected.items()
        if base.get(key) != value
    }


def compose_candidates(
    documents: Mapping[str, Mapping[str, Any]],
    rankings: Mapping[str, Sequence[CalibrationRank]],
    *,
    top: int = 5,
) -> list[CandidateConfiguration]:
    if not isinstance(top, int) or isinstance(top, bool) or not 1 <= top <= 5:
        raise ValueError("top must lie within 1..5")
    baselines = {"core": "baseline", "restart": "baseline", "cone": "cone-strict"}
    variants = {
        role: {
            item["variant"]: item["parameters"]
            for item in documents[role]["algorithms"]
        }
        for role in ("core", "restart", "cone")
    }
    shared = deepcopy(variants["core"]["baseline"])
    winners = {role: rankings[role][0].variant for role in rankings}
    core = deepcopy(variants["core"][winners["core"]])
    restart = deepcopy(shared)
    restart.update(
        _parameter_delta(
            variants["restart"][baselines["restart"]],
            variants["restart"][winners["restart"]],
        )
    )
    cone = deepcopy(shared)
    cone.update(
        _parameter_delta(
            variants["cone"][baselines["cone"]],
            variants["cone"][winners["cone"]],
        )
    )
    combined = deepcopy(shared)
    for role in ("core", "restart", "cone"):
        combined.update(
            _parameter_delta(
                variants[role][baselines[role]], variants[role][winners[role]]
            )
        )
    proposed = [
        CandidateConfiguration("baseline", shared, ("core:baseline",)),
        CandidateConfiguration("best-core", core, (f"core:{winners['core']}",)),
        CandidateConfiguration(
            "best-restart", restart, (f"restart:{winners['restart']}",)
        ),
        CandidateConfiguration("best-cone", cone, (f"cone:{winners['cone']}",)),
        CandidateConfiguration(
            "combined-best",
            combined,
            tuple(f"{role}:{winners[role]}" for role in ("core", "restart", "cone")),
        ),
    ]
    result = []
    seen: set[str] = set()
    for candidate in proposed:
        digest = canonical_sha256(candidate.parameters)
        if digest not in seen:
            seen.add(digest)
            result.append(candidate)
    return result[:top]


def build_shortlist(
    manifests: Sequence[Path], artifact_root: Path, *, top: int = 5
) -> tuple[dict[str, Any], str]:
    audit = audit_calibration_design(list(manifests))
    if not audit.passes:
        raise ValueError("calibration design audit failed: " + "; ".join(audit.violations))
    documents: dict[str, dict[str, Any]] = {}
    rankings: dict[str, list[CalibrationRank]] = {}
    source_artifacts: dict[str, str] = {}
    for path in manifests:
        document = load_manifest(path)
        role = ROLE_BY_MANIFEST[document["manifest_id"]]
        records, provenance = load_validated_records(path, artifact_root)
        documents[role] = document
        rankings[role] = rank_complete_blocks(records)
        source_artifacts.update(provenance["artifact_hashes"])
    candidates = compose_candidates(documents, rankings, top=top)
    manifest_text = confirmation_manifest_text(candidates)
    report = {
        "schema_version": 1,
        "mode": "shortlist",
        "selection_rule_version": SELECTION_RULE_VERSION,
        "source_manifest_hashes": dict(sorted(audit.manifest_hashes.items())),
        "source_artifact_hashes": dict(sorted(source_artifacts.items())),
        "rankings": {
            role: [asdict(item) for item in rankings[role]]
            for role in sorted(rankings)
        },
        "family_winners": {
            role: rankings[role][0].variant for role in sorted(rankings)
        },
        "candidates": [asdict(candidate) for candidate in candidates],
        "confirmation_manifest_sha256": hashlib.sha256(
            manifest_text.encode("utf-8")
        ).hexdigest(),
    }
    return report, manifest_text
