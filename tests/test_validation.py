import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import yaml

from hldbea.run_spec import expand_manifest


ROOT = Path(__file__).resolve().parents[1]


def _document(algorithms=("nsga2", "spea2"), seeds=(7, 19)):
    common = {
        "crossover_probability": 0.9,
        "crossover_eta": 20,
        "mutation_probability": "inverse_n_var",
        "mutation_eta": 20,
    }
    hldbea = {
        **common,
        "k": 1.0,
        "lambda": 0.1,
        "cone_epsilon": 0.0,
        "selection_mode": "local",
        "objective_policy": "round_robin",
        "tau": 1,
        "local_search_max_iter": 2,
        "restart": False,
        "restart_theta_zero": 0.95,
        "restart_window": 20,
        "restart_delta_hv": 0.0001,
        "restart_fraction": 0.2,
    }
    return {
        "schema_version": 1,
        "manifest_id": "validation-unit",
        "stage": "smoke",
        "seeds": list(seeds),
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
            {
                "id": algorithm,
                "variant": "standard",
                "parameters": hldbea if algorithm == "hldbea" else common,
            }
            for algorithm in algorithms
        ],
        "execution": {
            "population_size": 4,
            "evaluation_budget": 8,
            "checkpoints": [4, 8],
            "save_history": False,
        },
    }


def _materialize(document, artifact_root, *, omit=()):
    from hldbea.runner import execute_run

    omitted = set(omit)
    for spec in expand_manifest(document):
        if spec.run_id not in omitted:
            outcome = execute_run(spec, artifact_root)
            assert outcome.status == "budget_exhausted", outcome


def _artifact(artifact_root, spec):
    return artifact_root / spec.manifest_id / spec.run_id


def _mutate_metadata(path, mutation):
    metadata_path = path / "metadata.json"
    document = json.loads(metadata_path.read_text(encoding="utf-8"))
    mutation(document["run_metadata"])
    metadata_path.write_text(
        json.dumps(document, allow_nan=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _mutate_events(path, mutation):
    events_path = path / "events.json"
    events = json.loads(events_path.read_text(encoding="utf-8"))
    mutation(events)
    events_path.write_text(
        json.dumps(events, allow_nan=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    metadata_path = path / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["files"]["events.json"] = hashlib.sha256(
        events_path.read_bytes()
    ).hexdigest()
    metadata_path.write_text(
        json.dumps(metadata, allow_nan=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_clean_two_seed_matrix_passes_artifact_and_analysis_gates(tmp_path):
    from hldbea.validation import validate_dataset

    document = _document()
    _materialize(document, tmp_path)
    report = validate_dataset(document, tmp_path)

    assert report.planned == 4
    assert report.completed == 4
    assert report.valid == 4
    assert report.failed == report.missing == report.invalid == 0
    assert report.duplicate == 0
    assert report.gates == {"artifacts": True, "analysis": True}
    assert all(not values for values in report.issues.values())


def test_missing_run_is_reported_as_seed_pairing_failure(tmp_path):
    from hldbea.validation import validate_dataset

    document = _document()
    specs = expand_manifest(document)
    _materialize(document, tmp_path, omit=(specs[-1].run_id,))
    report = validate_dataset(document, tmp_path)

    assert report.missing == 1
    assert specs[-1].run_id in report.issues["missing_runs"]
    assert report.issues["seed_pairing"]
    assert not report.gates["artifacts"]


def test_failed_attempt_is_distinct_from_missing_run(tmp_path):
    from hldbea.artifacts import write_failure_artifact
    from hldbea.validation import validate_dataset

    document = _document(algorithms=("nsga2",), seeds=(7,))
    spec = expand_manifest(document)[0]
    write_failure_artifact(tmp_path, spec, RuntimeError("synthetic"))
    report = validate_dataset(document, tmp_path)

    assert report.failed == 1
    assert report.missing == 0
    assert spec.run_id in report.issues["failed_runs"]


def test_duplicate_active_run_is_detected(tmp_path):
    from hldbea.validation import validate_dataset

    document = _document(algorithms=("nsga2",), seeds=(7,))
    _materialize(document, tmp_path)
    spec = expand_manifest(document)[0]
    duplicate = tmp_path / spec.manifest_id / "duplicate-copy"
    shutil.copytree(_artifact(tmp_path, spec), duplicate)

    report = validate_dataset(document, tmp_path)

    assert report.duplicate == 1
    assert any(spec.run_id in issue for issue in report.issues["duplicate_runs"])
    assert not report.gates["artifacts"]


def test_fe_violation_is_reported_even_when_artifact_is_invalid(tmp_path):
    from hldbea.validation import validate_dataset

    document = _document(algorithms=("nsga2",), seeds=(7,))
    _materialize(document, tmp_path)
    spec = expand_manifest(document)[0]
    _mutate_metadata(
        _artifact(tmp_path, spec),
        lambda run: run["evaluations"].update(total=9, evolutionary=9),
    )

    report = validate_dataset(document, tmp_path)

    assert report.invalid == 1
    assert report.issues["fe_violations"]


def test_geometry_mismatch_fails_analysis_gate(tmp_path):
    from hldbea.validation import validate_dataset

    document = _document(algorithms=("nsga2",), seeds=(7,))
    _materialize(document, tmp_path)
    spec = expand_manifest(document)[0]
    _mutate_metadata(
        _artifact(tmp_path, spec),
        lambda run: run["geometry"].update(geometry_hash="0" * 64),
    )

    report = validate_dataset(document, tmp_path)

    assert report.valid == 1
    assert report.issues["geometry_mismatches"]
    assert report.gates["artifacts"]
    assert not report.gates["analysis"]


def test_nonfinite_metric_is_reported(tmp_path):
    from hldbea.validation import validate_dataset

    document = _document(algorithms=("nsga2",), seeds=(7,))
    _materialize(document, tmp_path)
    spec = expand_manifest(document)[0]
    _mutate_metadata(
        _artifact(tmp_path, spec),
        lambda run: run["metrics"].update(hv=float("nan")),
    )

    report = validate_dataset(document, tmp_path)

    assert report.issues["nonfinite_metrics"]
    assert not report.gates["analysis"]


def test_event_summary_inconsistency_is_reported(tmp_path):
    from hldbea.validation import validate_dataset

    document = _document(algorithms=("nsga2",), seeds=(7,))
    _materialize(document, tmp_path)
    spec = expand_manifest(document)[0]
    _mutate_metadata(
        _artifact(tmp_path, spec),
        lambda run: run["solver_summary"].update(calls=3),
    )

    report = validate_dataset(document, tmp_path)

    assert report.issues["event_inconsistencies"]
    assert not report.gates["analysis"]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda event: event.update(hv_gain=-0.1),
        lambda event: event.update(hv_gain=float("nan")),
        lambda event: event.update(accepted=False, hv_gain=0.1),
        lambda event: event.update(evaluation_before=5, evaluation_after=4),
    ],
)
def test_invalid_local_gain_or_timing_fails_analysis_gate(tmp_path, mutation):
    from hldbea.validation import validate_dataset

    document = _document(algorithms=("hldbea",), seeds=(7,))
    document["execution"].update(evaluation_budget=20, checkpoints=[4, 20])
    _materialize(document, tmp_path)
    spec = expand_manifest(document)[0]
    path = _artifact(tmp_path, spec)
    _mutate_events(path, lambda events: mutation(events["local_search"][0]))

    report = validate_dataset(document, tmp_path)

    assert report.issues["event_inconsistencies"]
    assert not report.gates["analysis"]


def test_restart_event_beyond_budget_fails_analysis_gate(tmp_path):
    from hldbea.validation import validate_dataset

    document = _document(algorithms=("hldbea",), seeds=(7,))
    document["execution"].update(evaluation_budget=20, checkpoints=[4, 20])
    _materialize(document, tmp_path)
    spec = expand_manifest(document)[0]
    path = _artifact(tmp_path, spec)
    _mutate_events(
        path,
        lambda events: events["restart"].append(
            {
                "evaluation": 21,
                "evaluation_after": 21,
                "generation": 1,
                "hv": 0.5,
                "zero_fraction": 0.5,
                "replacement_count": 0,
                "replace_indices": [],
                "status": "observed",
            }
        ),
    )
    _mutate_metadata(path, lambda run: run["restart_summary"].update(checks=1))

    report = validate_dataset(document, tmp_path)

    assert report.issues["event_inconsistencies"]
    assert not report.gates["analysis"]


def test_dependency_gate_detects_missing_or_inconsistent_runtime(tmp_path):
    from hldbea.validation import validate_dataset

    document = _document(algorithms=("nsga2",), seeds=(7,))
    _materialize(document, tmp_path)
    spec = expand_manifest(document)[0]
    _mutate_metadata(
        _artifact(tmp_path, spec),
        lambda run: run["environment"].update(pymoo="not-installed"),
    )

    report = validate_dataset(document, tmp_path)

    assert report.issues["dependency_gates"]
    assert not report.gates["analysis"]


def test_cli_writes_json_and_markdown_and_uses_selected_gate(tmp_path):
    document = _document(algorithms=("nsga2",), seeds=(7,))
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    json_output = tmp_path / "report.json"
    markdown_output = tmp_path / "report.md"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "analysis" / "validate_runs.py"),
            "--manifest",
            str(manifest),
            "--artifact-root",
            str(tmp_path / "missing-artifacts"),
            "--json-output",
            str(json_output),
            "--markdown-output",
            str(markdown_output),
            "--gate",
            "analysis",
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 1
    assert json.loads(json_output.read_text())["gates"]["analysis"] is False
    assert "# Dataset validation" in markdown_output.read_text()
