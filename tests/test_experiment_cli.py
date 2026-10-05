import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "experiments" / "run.py"


def _write_manifest(tmp_path, *, algorithm="nsga2", seeds=(19, 7)):
    manifest = tmp_path / f"{algorithm}.yaml"
    manifest.write_text(
        f"""schema_version: 1
manifest_id: cli-{algorithm}
stage: smoke
seeds: {list(seeds)}
problems:
  - id: dtlz2-m2
    name: dtlz2
    n_obj: 2
    n_var: 3
    parameters: {{}}
    transform:
      kind: identity
algorithms:
  - id: {algorithm}
    variant: standard
    parameters:
      crossover_probability: 0.9
      crossover_eta: 20
      mutation_probability: inverse_n_var
      mutation_eta: 20
execution:
  population_size: 4
  evaluation_budget: 8
  checkpoints: [4, 8]
  save_history: false
""",
        encoding="utf-8",
    )
    return manifest


def _run_cli(manifest, artifact_root, *arguments):
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    environment["MPLBACKEND"] = "Agg"
    return subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--manifest",
            str(manifest),
            "--artifact-root",
            str(artifact_root),
            *arguments,
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )


def _records(stdout, prefix):
    return [
        json.loads(line.removeprefix(prefix))
        for line in stdout.splitlines()
        if line.startswith(prefix)
    ]


def test_dry_run_reports_exact_inventory_and_disk_bound(tmp_path):
    manifest = _write_manifest(tmp_path)
    result = _run_cli(manifest, tmp_path / "artifacts", "--dry-run")

    assert result.returncode == 0, result.stderr
    plan = _records(result.stdout, "PLAN ")[0]
    assert plan["planned"] == 2
    assert plan["missing"] == 2
    assert plan["valid"] == 0
    assert plan["invalid"] == 0
    assert plan["failed"] == 0
    assert plan["estimated_max_bytes"] > 0
    assert plan["workers"] == min(4, os.cpu_count() or 1)


def test_only_run_id_filters_to_one_exact_run(tmp_path):
    from hldbea.run_spec import expand_manifest, load_manifest

    manifest = _write_manifest(tmp_path)
    selected = sorted(
        expand_manifest(load_manifest(manifest)), key=lambda spec: spec.run_id
    )[0]
    result = _run_cli(
        manifest,
        tmp_path / "artifacts",
        "--dry-run",
        "--only-run-id",
        selected.run_id,
    )

    assert result.returncode == 0, result.stderr
    plan = _records(result.stdout, "PLAN ")[0]
    assert plan["planned"] == 1
    assert plan["run_ids"] == [selected.run_id]


def test_parallel_output_is_sorted_and_resume_skips_valid_runs(tmp_path):
    manifest = _write_manifest(tmp_path)
    artifact_root = tmp_path / "artifacts"
    first = _run_cli(manifest, artifact_root, "--workers", "2")

    assert first.returncode == 0, first.stderr
    runs = _records(first.stdout, "RUN ")
    assert [record["run_id"] for record in runs] == sorted(
        record["run_id"] for record in runs
    )
    assert {record["status"] for record in runs} == {"budget_exhausted"}

    resumed = _run_cli(manifest, artifact_root, "--workers", "2", "--resume")
    assert resumed.returncode == 0, resumed.stderr
    summary = _records(resumed.stdout, "SUMMARY ")[0]
    assert summary["scheduled"] == 0
    assert summary["skipped_valid"] == 2


def test_resume_quarantines_invalid_artifact_then_recomputes_it(tmp_path):
    from hldbea.run_spec import expand_manifest, load_manifest

    manifest = _write_manifest(tmp_path, seeds=(7,))
    artifact_root = tmp_path / "artifacts"
    first = _run_cli(manifest, artifact_root, "--workers", "1")
    assert first.returncode == 0, first.stderr
    spec = expand_manifest(load_manifest(manifest))[0]
    artifact = artifact_root / spec.manifest_id / spec.run_id
    (artifact / "events.json").write_text("{}\n", encoding="utf-8")

    resumed = _run_cli(manifest, artifact_root, "--workers", "1", "--resume")

    assert resumed.returncode == 0, resumed.stderr
    assert (artifact / "metadata.json").is_file()
    quarantined = list(
        (artifact_root / spec.manifest_id / "invalid" / spec.run_id).iterdir()
    )
    assert len(quarantined) == 1
    assert (quarantined[0] / "metadata.json").is_file()


def test_worker_cap_requires_explicit_override(tmp_path):
    manifest = _write_manifest(tmp_path, seeds=(7,))
    rejected = _run_cli(
        manifest, tmp_path / "artifacts", "--dry-run", "--workers", "5"
    )
    allowed = _run_cli(
        manifest,
        tmp_path / "artifacts",
        "--dry-run",
        "--workers",
        "5",
        "--allow-oversubscription",
    )

    assert rejected.returncode == 2
    assert "--allow-oversubscription" in rejected.stderr
    assert allowed.returncode == 0, allowed.stderr


def test_failed_run_produces_nonzero_exit_status(tmp_path):
    manifest = _write_manifest(tmp_path, algorithm="unknown", seeds=(7,))
    result = _run_cli(manifest, tmp_path / "artifacts", "--workers", "1")

    assert result.returncode == 1
    runs = _records(result.stdout, "RUN ")
    assert runs[0]["status"] == "failed"
    assert runs[0]["error_type"] == "ValueError"
