"""Strict experiment manifests and deterministic run identities."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import yaml


ROOT_FIELDS = {
    "schema_version",
    "manifest_id",
    "stage",
    "seeds",
    "problems",
    "algorithms",
    "execution",
}
PROBLEM_FIELDS = {
    "id",
    "name",
    "n_obj",
    "n_var",
    "parameters",
    "transform",
}
ALGORITHM_FIELDS = {"id", "variant", "parameters"}
EXECUTION_FIELDS = {
    "population_size",
    "evaluation_budget",
    "checkpoints",
    "save_history",
}
TRANSFORM_FIELDS = {"kind", "seed", "angle_degrees", "matrix"}


def canonical_sha256(value: Any) -> str:
    """Hash a finite JSON value with stable separators and mapping order."""

    try:
        payload = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("value must contain canonical finite JSON values") from exc
    return hashlib.sha256(payload).hexdigest()


def _strict_integer(value: Any, name: str, *, minimum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    if not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} keys must be strings")
    return value


def _known_fields(value: dict[str, Any], allowed: set[str], name: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"unknown {name} fields: {sorted(unknown)}")
    missing = allowed - set(value)
    if missing:
        raise ValueError(f"missing {name} fields: {sorted(missing)}")


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[a-z0-9][a-z0-9._-]*", value):
        raise ValueError(f"{name} must be a non-empty lowercase identifier")
    return value


def _json_copy(value: Any, name: str) -> Any:
    copied = deepcopy(value)
    try:
        json.dumps(copied, allow_nan=False, sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain canonical finite JSON values") from exc
    return copied


@dataclass(frozen=True)
class RunSpec:
    schema_version: int
    manifest_id: str
    stage: str
    problem_id: str
    problem_name: str
    n_obj: int
    n_var: int
    problem_parameters: dict[str, Any]
    transform: dict[str, Any]
    algorithm_id: str
    variant: str
    algorithm_parameters: dict[str, Any]
    population_size: int
    evaluation_budget: int
    checkpoints: tuple[int, ...]
    save_history: bool
    seed: int

    def canonical_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "manifest_id": self.manifest_id,
            "stage": self.stage,
            "problem": {
                "id": self.problem_id,
                "name": self.problem_name,
                "n_obj": self.n_obj,
                "n_var": self.n_var,
                "parameters": deepcopy(self.problem_parameters),
                "transform": deepcopy(self.transform),
            },
            "algorithm": {
                "id": self.algorithm_id,
                "variant": self.variant,
                "parameters": deepcopy(self.algorithm_parameters),
            },
            "execution": {
                "population_size": self.population_size,
                "evaluation_budget": self.evaluation_budget,
                "checkpoints": list(self.checkpoints),
                "save_history": self.save_history,
            },
            "seed": self.seed,
        }

    @property
    def config_hash(self) -> str:
        return canonical_sha256(self.canonical_dict())

    @property
    def run_id(self) -> str:
        return (
            f"{self.manifest_id}__{self.problem_id}__{self.algorithm_id}-"
            f"{self.variant}__s{self.seed}__{self.config_hash[:12]}"
        )


def load_manifest(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"manifest does not exist: {source}")
    with source.open(encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    return _mapping(document, "manifest")


def _validate_transform(value: Any) -> dict[str, Any]:
    transform = _mapping(value, "problem transform")
    unknown = set(transform) - TRANSFORM_FIELDS
    if unknown:
        raise ValueError(f"unknown problem transform fields: {sorted(unknown)}")
    kind = transform.get("kind")
    if kind not in {"identity", "rotation"}:
        raise ValueError("transform kind must be 'identity' or 'rotation'")
    if kind == "identity" and set(transform) != {"kind"}:
        raise ValueError("identity transform accepts only its kind")
    if kind == "rotation" and not ({"seed", "angle_degrees", "matrix"} & set(transform)):
        raise ValueError("rotation transform requires seed, angle_degrees, or matrix")
    return _json_copy(transform, "problem transform")


def expand_manifest(document: dict[str, Any]) -> list[RunSpec]:
    root = _mapping(document, "manifest")
    _known_fields(root, ROOT_FIELDS, "manifest")
    if root["schema_version"] != 1:
        raise ValueError("schema_version must equal 1")
    manifest_id = _identifier(root["manifest_id"], "manifest_id")
    stage = _identifier(root["stage"], "stage")

    seeds = root["seeds"]
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("seeds must be a non-empty list")
    seeds = [_strict_integer(seed, "seed", minimum=0) for seed in seeds]
    if len(seeds) != len(set(seeds)):
        raise ValueError("seeds must be unique")

    execution = _mapping(root["execution"], "execution")
    _known_fields(execution, EXECUTION_FIELDS, "execution")
    population_size = _strict_integer(
        execution["population_size"], "population_size", minimum=2
    )
    evaluation_budget = _strict_integer(
        execution["evaluation_budget"], "evaluation_budget", minimum=1
    )
    if evaluation_budget < population_size:
        raise ValueError("evaluation_budget must cover the initial population")
    checkpoints = execution["checkpoints"]
    if not isinstance(checkpoints, list) or not checkpoints:
        raise ValueError("checkpoints must be a non-empty list")
    checkpoints = tuple(
        _strict_integer(value, "checkpoint", minimum=1) for value in checkpoints
    )
    if tuple(sorted(set(checkpoints))) != checkpoints:
        raise ValueError("checkpoints must be strictly increasing")
    if checkpoints[-1] != evaluation_budget:
        raise ValueError("the final checkpoint must equal evaluation_budget")
    if not isinstance(execution["save_history"], bool):
        raise ValueError("save_history must be boolean")

    raw_problems = root["problems"]
    if not isinstance(raw_problems, list) or not raw_problems:
        raise ValueError("problems must be a non-empty list")
    problems = []
    for index, raw_problem in enumerate(raw_problems):
        problem = _mapping(raw_problem, f"problem {index}")
        _known_fields(problem, PROBLEM_FIELDS, f"problem {index}")
        problems.append(
            {
                "id": _identifier(problem["id"], "problem id"),
                "name": _identifier(problem["name"], "problem name"),
                "n_obj": _strict_integer(problem["n_obj"], "n_obj", minimum=2),
                "n_var": _strict_integer(problem["n_var"], "n_var", minimum=1),
                "parameters": _json_copy(
                    _mapping(problem["parameters"], "problem parameters"),
                    "problem parameters",
                ),
                "transform": _validate_transform(problem["transform"]),
            }
        )
    problem_ids = [problem["id"] for problem in problems]
    if len(problem_ids) != len(set(problem_ids)):
        raise ValueError("problem ids must be unique")

    raw_algorithms = root["algorithms"]
    if not isinstance(raw_algorithms, list) or not raw_algorithms:
        raise ValueError("algorithms must be a non-empty list")
    algorithms = []
    for index, raw_algorithm in enumerate(raw_algorithms):
        algorithm = _mapping(raw_algorithm, f"algorithm {index}")
        _known_fields(algorithm, ALGORITHM_FIELDS, f"algorithm {index}")
        algorithms.append(
            {
                "id": _identifier(algorithm["id"], "algorithm id"),
                "variant": _identifier(algorithm["variant"], "algorithm variant"),
                "parameters": _json_copy(
                    _mapping(algorithm["parameters"], "algorithm parameters"),
                    "algorithm parameters",
                ),
            }
        )
    algorithm_keys = [(item["id"], item["variant"]) for item in algorithms]
    if len(algorithm_keys) != len(set(algorithm_keys)):
        raise ValueError("algorithm id/variant pairs must be unique")

    specs = []
    for problem in problems:
        for algorithm in algorithms:
            for seed in seeds:
                specs.append(
                    RunSpec(
                        schema_version=1,
                        manifest_id=manifest_id,
                        stage=stage,
                        problem_id=problem["id"],
                        problem_name=problem["name"],
                        n_obj=problem["n_obj"],
                        n_var=problem["n_var"],
                        problem_parameters=deepcopy(problem["parameters"]),
                        transform=deepcopy(problem["transform"]),
                        algorithm_id=algorithm["id"],
                        variant=algorithm["variant"],
                        algorithm_parameters=deepcopy(algorithm["parameters"]),
                        population_size=population_size,
                        evaluation_budget=evaluation_budget,
                        checkpoints=checkpoints,
                        save_history=execution["save_history"],
                        seed=seed,
                    )
                )
    run_ids = [spec.run_id for spec in specs]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("expanded run ids must be unique")
    return specs
