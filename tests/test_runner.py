from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from hldbea.artifacts import artifact_status, validate_run_artifact
from hldbea.run_spec import expand_manifest, load_manifest


MANIFEST = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "manifests"
    / "integration-smoke.yaml"
)


def _spec(algorithm="nsga2", problem="dtlz2-m3", seed=31001):
    specs = expand_manifest(load_manifest(MANIFEST))
    return next(
        spec
        for spec in specs
        if spec.algorithm_id == algorithm
        and spec.problem_id == problem
        and spec.seed == seed
    )


def _load(path):
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    events = json.loads((path / "events.json").read_text(encoding="utf-8"))
    checkpoints = json.loads(
        (path / "checkpoints.json").read_text(encoding="utf-8")
    )
    with np.load(path / "arrays.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    return metadata, events, checkpoints, arrays


def test_single_run_writes_complete_valid_artifact(tmp_path):
    from hldbea.runner import execute_run

    spec = replace(
        _spec(), evaluation_budget=20, checkpoints=(10, 20)
    )
    outcome = execute_run(spec, tmp_path)
    validation = validate_run_artifact(outcome.artifact_path, spec)
    envelope, events, checkpoints, arrays = _load(outcome.artifact_path)
    run = envelope["run_metadata"]

    assert outcome.status == "budget_exhausted"
    assert validation.valid, validation.errors
    assert run["evaluations"] == {"total": 20, "evolutionary": 20, "solver": 0}
    assert run["environment"]["python"]
    assert run["environment"]["pymoo"] == "0.6.1.3"
    assert len(run["git"]["source_sha256"]) == 64
    if run["git"]["commit"] == "unavailable":
        # Source archive without Git metadata: still a valid, fingerprinted run.
        assert run["git"]["dirty"] is None and run["git"]["error"]
    else:
        assert len(run["git"]["commit"]) == 40
        assert isinstance(run["git"]["dirty"], bool)
    assert run["history_saved"] is False
    assert run["history_length"] == 0
    assert np.all(np.isfinite(arrays["X"]))
    assert np.all(np.isfinite(arrays["F"]))
    assert events == {"local_search": [], "restart": []}
    assert [item["target_evaluation"] for item in checkpoints] == [10, 20]
    assert all(item["evaluation"] <= 20 for item in checkpoints)


def test_same_seed_replays_same_population_metrics_and_checkpoints(tmp_path):
    from hldbea.runner import execute_run

    spec = replace(_spec(), evaluation_budget=20, checkpoints=(10, 20))
    first = execute_run(spec, tmp_path / "first")
    second = execute_run(spec, tmp_path / "second")
    meta1, events1, checkpoints1, arrays1 = _load(first.artifact_path)
    meta2, events2, checkpoints2, arrays2 = _load(second.artifact_path)

    np.testing.assert_array_equal(arrays1["X"], arrays2["X"])
    np.testing.assert_array_equal(arrays1["F"], arrays2["F"])
    assert meta1["run_metadata"]["metrics"] == meta2["run_metadata"]["metrics"]
    assert checkpoints1 == checkpoints2
    assert events1 == events2


def test_different_seed_changes_stochastic_population(tmp_path):
    from hldbea.runner import execute_run

    first_spec = replace(_spec(), evaluation_budget=20, checkpoints=(10, 20))
    second_spec = replace(first_spec, seed=31002)
    first = execute_run(first_spec, tmp_path / "first")
    second = execute_run(second_spec, tmp_path / "second")
    _, _, _, arrays1 = _load(first.artifact_path)
    _, _, _, arrays2 = _load(second.artifact_path)

    assert not np.array_equal(arrays1["X"], arrays2["X"])


def test_run_stops_cleanly_before_an_evolutionary_batch_would_overspend(tmp_path):
    from hldbea.runner import execute_run

    spec = replace(_spec(), evaluation_budget=15, checkpoints=(10, 15))
    outcome = execute_run(spec, tmp_path)
    envelope, _, checkpoints, _ = _load(outcome.artifact_path)
    evaluations = envelope["run_metadata"]["evaluations"]

    assert outcome.status == "budget_exhausted"
    assert evaluations["total"] == 10
    assert evaluations["total"] <= spec.evaluation_budget
    assert [item["target_evaluation"] for item in checkpoints] == [10]
    assert validate_run_artifact(outcome.artifact_path, spec).valid


def test_hldbea_run_unifies_solver_and_evolutionary_fe_and_records_events(tmp_path):
    from hldbea.runner import execute_run

    spec = replace(
        _spec("hldbea"), evaluation_budget=30, checkpoints=(10, 20, 30)
    )
    outcome = execute_run(spec, tmp_path)
    envelope, events, _, _ = _load(outcome.artifact_path)
    run = envelope["run_metadata"]

    assert validate_run_artifact(outcome.artifact_path, spec).valid
    assert run["evaluations"]["total"] == (
        run["evaluations"]["evolutionary"] + run["evaluations"]["solver"]
    )
    assert run["evaluations"]["total"] <= spec.evaluation_budget
    assert run["evaluations"]["solver"] > 0
    assert len(events["local_search"]) > 0
    assert all(
        {
            "evaluations",
            "evaluation_before",
            "evaluation_after",
            "hv_gain",
            "gain_per_evaluation",
        }
        <= set(event)
        for event in events["local_search"]
    )
    assert all(
        event["evaluation_after"] - event["evaluation_before"]
        == event["evaluations"]
        for event in events["local_search"]
    )
    assert run["solver_summary"]["accepted_hv_gain"] == sum(
        event["hv_gain"] for event in events["local_search"]
    )
    assert run["solver_summary"]["gain_per_evaluation"] == (
        run["solver_summary"]["accepted_hv_gain"]
        / run["solver_summary"]["charged_evaluations"]
    )


def test_failed_run_is_captured_without_complete_artifact(tmp_path):
    from hldbea.runner import execute_run

    spec = replace(_spec(), algorithm_id="unknown")
    outcome = execute_run(spec, tmp_path)

    assert outcome.status == "failed"
    assert outcome.error_type == "ValueError"
    assert artifact_status(tmp_path, spec) == "failed"
    assert not (tmp_path / spec.manifest_id / spec.run_id).exists()


def test_git_metadata_keeps_source_digest_without_git(monkeypatch):
    import subprocess

    from hldbea import runner

    def no_git(*args, **kwargs):
        raise OSError("git unavailable")

    monkeypatch.setattr(subprocess, "run", no_git)
    meta = runner._git_metadata()
    assert meta["commit"] == "unavailable"
    assert meta["source_sha256"] == runner._source_digest()


@pytest.mark.skipif(
    __import__("shutil").which("git") is None
    or not (Path(__file__).resolve().parents[1] / ".git").exists(),
    reason="checkout-specific Git metadata",
)
def test_git_metadata_in_a_checkout_records_commit_and_diff():
    from hldbea import runner

    meta = runner._git_metadata()
    assert len(meta["commit"]) == 40
    assert isinstance(meta["tracked_sources_modified"], bool)
    assert len(meta["source_diff_sha256"]) == 64
