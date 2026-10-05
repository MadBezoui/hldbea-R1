from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from hldbea.run_spec import expand_manifest


def _spec(seed=7):
    document = {
        "schema_version": 1,
        "manifest_id": "artifact-unit",
        "stage": "smoke",
        "seeds": [seed],
        "problems": [
            {
                "id": "dtlz2-m2",
                "name": "dtlz2",
                "n_obj": 2,
                "n_var": 3,
                "parameters": {},
                "transform": {"kind": "identity"},
            }
        ],
        "algorithms": [
            {"id": "nsga2", "variant": "standard", "parameters": {}}
        ],
        "execution": {
            "population_size": 3,
            "evaluation_budget": 12,
            "checkpoints": [3, 6, 12],
            "save_history": False,
        },
    }
    return expand_manifest(document)[0]


def _payload(total=12):
    metadata = {
        "termination_status": "budget_exhausted",
        "evaluations": {"total": total, "evolutionary": total - 2, "solver": 2},
        "runtime": {"wall_seconds": 0.25, "cpu_seconds": 0.2},
    }
    arrays = {
        "X": np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
        "F": np.array([[0.2, 0.8], [0.4, 0.6]]),
    }
    events = {"restart": [], "local_search": [{"accepted": True}]}
    checkpoints = [
        {"evaluation": 3, "hv": 0.1, "igd_plus": 0.4},
        {"evaluation": 12, "hv": 0.2, "igd_plus": 0.3},
    ]
    return metadata, arrays, events, checkpoints


def test_atomic_artifact_round_trip_and_valid_resume_status(tmp_path):
    from hldbea.artifacts import artifact_status, validate_run_artifact, write_run_artifact

    spec = _spec()
    path = write_run_artifact(tmp_path, spec, *_payload())
    validation = validate_run_artifact(path, expected_spec=spec)

    assert path.name == spec.run_id
    assert validation.valid
    assert validation.errors == ()
    assert artifact_status(tmp_path, spec) == "valid"
    assert not list(path.parent.glob(f".{spec.run_id}.tmp-*"))


def test_valid_artifact_is_immutable(tmp_path):
    from hldbea.artifacts import write_run_artifact

    spec = _spec()
    write_run_artifact(tmp_path, spec, *_payload())

    with pytest.raises(FileExistsError):
        write_run_artifact(tmp_path, spec, *_payload())


def test_checksum_corruption_is_invalid_not_resumable(tmp_path):
    from hldbea.artifacts import artifact_status, validate_run_artifact, write_run_artifact

    spec = _spec()
    path = write_run_artifact(tmp_path, spec, *_payload())
    (path / "events.json").write_text("{}\n", encoding="utf-8")

    validation = validate_run_artifact(path, expected_spec=spec)

    assert not validation.valid
    assert any("checksum" in error for error in validation.errors)
    assert artifact_status(tmp_path, spec) == "invalid"


def test_expected_spec_hash_mismatch_is_rejected(tmp_path):
    from hldbea.artifacts import validate_run_artifact, write_run_artifact

    spec = _spec()
    path = write_run_artifact(tmp_path, spec, *_payload())
    other = replace(spec, seed=13)

    validation = validate_run_artifact(path, expected_spec=other)

    assert not validation.valid
    assert any("expected config hash" in error for error in validation.errors)


def test_over_budget_metadata_is_never_finalized(tmp_path):
    from hldbea.artifacts import write_run_artifact

    spec = _spec()
    payload = _payload(total=13)

    with pytest.raises(ValueError, match="budget"):
        write_run_artifact(tmp_path, spec, *payload)
    assert not (tmp_path / spec.manifest_id / spec.run_id).exists()


def test_nonfinite_population_is_never_finalized(tmp_path):
    from hldbea.artifacts import write_run_artifact

    spec = _spec()
    metadata, arrays, events, checkpoints = _payload()
    arrays["F"][0, 0] = np.nan

    with pytest.raises(ValueError, match="finite"):
        write_run_artifact(tmp_path, spec, metadata, arrays, events, checkpoints)
    assert not (tmp_path / spec.manifest_id / spec.run_id).exists()


def test_crossed_checkpoint_targets_may_share_the_same_observed_evaluation(tmp_path):
    from hldbea.artifacts import validate_run_artifact, write_run_artifact

    spec = _spec()
    metadata, arrays, events, _ = _payload()
    checkpoints = [
        {"target_evaluation": 3, "evaluation": 7, "hv": 0.1},
        {"target_evaluation": 6, "evaluation": 7, "hv": 0.1},
        {"target_evaluation": 12, "evaluation": 12, "hv": 0.2},
    ]

    path = write_run_artifact(
        tmp_path, spec, metadata, arrays, events, checkpoints
    )

    assert validate_run_artifact(path, expected_spec=spec).valid


def test_interrupted_temp_directory_stays_inspectable_and_run_is_missing(tmp_path):
    from hldbea.artifacts import artifact_status

    spec = _spec()
    parent = tmp_path / spec.manifest_id
    interrupted = parent / f".{spec.run_id}.tmp-interrupted"
    interrupted.mkdir(parents=True)
    (interrupted / "partial.json").write_text('{"state":"partial"}', encoding="utf-8")

    assert artifact_status(tmp_path, spec) == "missing"
    assert interrupted.is_dir()


def test_failure_artifact_does_not_mark_run_complete(tmp_path):
    from hldbea.artifacts import artifact_status, write_failure_artifact

    spec = _spec()
    failure = write_failure_artifact(
        tmp_path,
        spec,
        RuntimeError("synthetic failure"),
        metadata={"worker": "unit"},
    )

    assert artifact_status(tmp_path, spec) == "failed"
    document = json.loads(failure.read_text(encoding="utf-8"))
    assert document["run_id"] == spec.run_id
    assert document["error"]["type"] == "RuntimeError"
    assert "synthetic failure" in document["error"]["message"]
    assert not (tmp_path / spec.manifest_id / spec.run_id).exists()
