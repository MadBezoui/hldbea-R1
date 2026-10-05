"""Atomic, immutable and checksum-validated experiment artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import traceback
from typing import Any, Mapping, Sequence
import uuid

import numpy as np

from .run_spec import RunSpec, canonical_sha256


ARTIFACT_SCHEMA_VERSION = 1
PAYLOAD_FILES = ("arrays.npz", "events.json", "checkpoints.json")


@dataclass(frozen=True)
class RunArtifact:
    path: Path
    run_id: str
    config_hash: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ValidationResult:
    path: Path
    valid: bool
    errors: tuple[str, ...]
    run_id: str | None = None
    config_hash: str | None = None


def artifact_path(root: str | Path, spec: RunSpec) -> Path:
    return Path(root) / spec.manifest_id / spec.run_id


def arrays_digest(run_dir: str | Path) -> str:
    """SHA-256 of the final decision and objective arrays of a run."""

    with np.load(Path(run_dir) / "arrays.npz", allow_pickle=False) as archive:
        digest = hashlib.sha256()
        for name in ("X", "F"):
            values = np.ascontiguousarray(archive[name], dtype=np.float64)
            digest.update(name.encode("ascii"))
            digest.update(str(values.shape).encode("ascii"))
            digest.update(values.tobytes())
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_text(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
    ) + "\n"


def _strict_nonnegative_integer(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _validate_payload(
    spec: Any,
    metadata: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    events: Mapping[str, Any],
    checkpoints: Sequence[Mapping[str, Any]],
) -> None:
    if not isinstance(metadata, Mapping):
        raise ValueError("metadata must be a mapping")
    if metadata.get("termination_status") not in {
        "completed",
        "budget_exhausted",
        "terminated",
    }:
        raise ValueError("metadata has an invalid termination_status")
    evaluations = metadata.get("evaluations")
    if not isinstance(evaluations, Mapping):
        raise ValueError("metadata evaluations must be a mapping")
    total = _strict_nonnegative_integer(evaluations.get("total"), "total FE")
    evolutionary = _strict_nonnegative_integer(
        evaluations.get("evolutionary"), "evolutionary FE"
    )
    solver = _strict_nonnegative_integer(evaluations.get("solver"), "solver FE")
    if total != evolutionary + solver:
        raise ValueError("total FE must equal evolutionary plus solver FE")
    if total > spec.evaluation_budget:
        raise ValueError("run exceeds its declared evaluation budget")

    if not isinstance(arrays, Mapping) or set(arrays) < {"X", "F"}:
        raise ValueError("arrays must contain X and F")
    checked = {}
    for name, raw in arrays.items():
        value = np.asarray(raw)
        if value.dtype.hasobject or not np.issubdtype(value.dtype, np.number):
            raise ValueError(f"array {name} must be numeric and pickle-free")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"array {name} must contain only finite values")
        checked[name] = value
    if checked["X"].ndim != 2 or checked["X"].shape[1] != spec.n_var:
        raise ValueError("X has an invalid decision-space shape")
    if checked["F"].ndim != 2 or checked["F"].shape[1] != spec.n_obj:
        raise ValueError("F has an invalid objective-space shape")
    if len(checked["X"]) != len(checked["F"]):
        raise ValueError("X and F population rows must align")

    if not isinstance(events, Mapping):
        raise ValueError("events must be a mapping")
    if not isinstance(checkpoints, Sequence) or isinstance(checkpoints, (str, bytes)):
        raise ValueError("checkpoints must be a sequence")
    previous_evaluation = -1
    previous_target = -1
    for checkpoint in checkpoints:
        if not isinstance(checkpoint, Mapping):
            raise ValueError("each checkpoint must be a mapping")
        evaluation = _strict_nonnegative_integer(
            checkpoint.get("evaluation"), "checkpoint evaluation"
        )
        target = _strict_nonnegative_integer(
            checkpoint.get("target_evaluation", evaluation),
            "checkpoint target evaluation",
        )
        if evaluation < previous_evaluation or evaluation > spec.evaluation_budget:
            raise ValueError(
                "checkpoint evaluations must be non-decreasing within the budget"
            )
        if target <= previous_target or target > evaluation:
            raise ValueError(
                "checkpoint targets must increase and not exceed their observation"
            )
        if target not in spec.checkpoints:
            raise ValueError("checkpoint target is not declared in the run spec")
        previous_evaluation = evaluation
        previous_target = target

    _json_text(dict(metadata))
    _json_text(dict(events))
    _json_text([dict(checkpoint) for checkpoint in checkpoints])


def write_run_artifact(
    root: str | Path,
    spec: RunSpec,
    metadata: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    events: Mapping[str, Any],
    checkpoints: Sequence[Mapping[str, Any]],
) -> Path:
    """Validate, write to a sibling temporary directory, then atomically publish."""

    _validate_payload(spec, metadata, arrays, events, checkpoints)
    target = artifact_path(root, spec)
    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"run artifact already exists: {target}")

    temporary = parent / f".{spec.run_id}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir()
    try:
        np.savez_compressed(
            temporary / "arrays.npz",
            **{name: np.asarray(value) for name, value in arrays.items()},
        )
        (temporary / "events.json").write_text(
            _json_text(dict(events)), encoding="utf-8"
        )
        (temporary / "checkpoints.json").write_text(
            _json_text([dict(checkpoint) for checkpoint in checkpoints]),
            encoding="utf-8",
        )
        checksums = {
            filename: _sha256_file(temporary / filename) for filename in PAYLOAD_FILES
        }
        envelope = {
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "run_id": spec.run_id,
            "config_hash": spec.config_hash,
            "spec": spec.canonical_dict(),
            "run_metadata": dict(metadata),
            "files": checksums,
        }
        (temporary / "metadata.json").write_text(
            _json_text(envelope), encoding="utf-8"
        )
        os.rename(temporary, target)
    except Exception:
        # Retain the temporary directory for post-mortem inspection.
        raise
    return target


def _read_json(path: Path, errors: list[str]) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        errors.append(f"cannot read {path.name}: {type(exc).__name__}: {exc}")
        return None


def validate_run_artifact(
    path: str | Path, expected_spec: RunSpec | None = None
) -> ValidationResult:
    location = Path(path)
    errors: list[str] = []
    if not location.is_dir():
        return ValidationResult(location, False, ("artifact directory is missing",))

    envelope = _read_json(location / "metadata.json", errors)
    if not isinstance(envelope, dict):
        return ValidationResult(location, False, tuple(errors or ["metadata is invalid"]))
    run_id = envelope.get("run_id")
    config_hash = envelope.get("config_hash")
    if envelope.get("artifact_schema_version") != ARTIFACT_SCHEMA_VERSION:
        errors.append("unsupported artifact schema version")
    embedded_spec = envelope.get("spec")
    if not isinstance(embedded_spec, dict):
        errors.append("embedded spec is missing or invalid")
    else:
        try:
            if canonical_sha256(embedded_spec) != config_hash:
                errors.append("embedded spec config hash mismatch")
        except ValueError as exc:
            errors.append(f"embedded spec is not canonical: {exc}")
    if expected_spec is not None:
        if config_hash != expected_spec.config_hash:
            errors.append("expected config hash does not match artifact")
        if run_id != expected_spec.run_id:
            errors.append("expected run id does not match artifact")
    if location.name != run_id:
        errors.append("directory name does not match run id")

    files = envelope.get("files")
    if not isinstance(files, dict) or set(files) != set(PAYLOAD_FILES):
        errors.append("payload file manifest is invalid")
    else:
        for filename in PAYLOAD_FILES:
            payload = location / filename
            if not payload.is_file():
                errors.append(f"payload file is missing: {filename}")
            elif _sha256_file(payload) != files[filename]:
                errors.append(f"checksum mismatch for {filename}")

    events = _read_json(location / "events.json", errors)
    checkpoints = _read_json(location / "checkpoints.json", errors)
    arrays: dict[str, np.ndarray] = {}
    try:
        with np.load(location / "arrays.npz", allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
    except (OSError, ValueError, KeyError) as exc:
        errors.append(f"cannot read arrays.npz: {type(exc).__name__}: {exc}")

    if isinstance(embedded_spec, dict):
        try:
            problem = embedded_spec["problem"]
            execution = embedded_spec["execution"]

            @dataclass(frozen=True)
            class _ShapeBudget:
                n_var: int
                n_obj: int
                evaluation_budget: int
                checkpoints: tuple[int, ...]

            validation_shape = _ShapeBudget(
                n_var=problem["n_var"],
                n_obj=problem["n_obj"],
                evaluation_budget=execution["evaluation_budget"],
                checkpoints=tuple(execution["checkpoints"]),
            )
            _validate_payload(
                validation_shape,
                envelope.get("run_metadata"),
                arrays,
                events,
                checkpoints,
            )
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"payload validation failed: {exc}")

    return ValidationResult(
        path=location,
        valid=not errors,
        errors=tuple(errors),
        run_id=run_id if isinstance(run_id, str) else None,
        config_hash=config_hash if isinstance(config_hash, str) else None,
    )


def write_failure_artifact(
    root: str | Path,
    spec: RunSpec,
    error: BaseException,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> Path:
    failure_root = Path(root) / spec.manifest_id / "failures" / spec.run_id
    failure_root.mkdir(parents=True, exist_ok=True)
    attempt_id = uuid.uuid4().hex
    target = failure_root / f"attempt-{attempt_id}.json"
    temporary = failure_root / f".attempt-{attempt_id}.tmp"
    document = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "run_id": spec.run_id,
        "config_hash": spec.config_hash,
        "metadata": dict(metadata or {}),
        "error": {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": "".join(
                traceback.format_exception(type(error), error, error.__traceback__)
            ),
        },
    }
    temporary.write_text(_json_text(document), encoding="utf-8")
    os.replace(temporary, target)
    return target


def artifact_status(root: str | Path, spec: RunSpec) -> str:
    target = artifact_path(root, spec)
    if target.exists():
        return "valid" if validate_run_artifact(target, spec).valid else "invalid"
    failure_root = Path(root) / spec.manifest_id / "failures" / spec.run_id
    if failure_root.is_dir() and any(failure_root.glob("attempt-*.json")):
        return "failed"
    return "missing"
