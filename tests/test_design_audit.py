from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = (
    ROOT / "experiments" / "manifests" / "calibration-core-screen-v1.yaml",
    ROOT / "experiments" / "manifests" / "calibration-restart-screen-v1.yaml",
    ROOT / "experiments" / "manifests" / "calibration-cone-screen-v1.yaml",
)


def _copied_manifests(tmp_path, mutation=None):
    paths = []
    for source in MANIFESTS:
        document = yaml.safe_load(source.read_text(encoding="utf-8"))
        if mutation is not None:
            mutation(source.name, document)
        destination = tmp_path / source.name
        destination.write_text(
            yaml.safe_dump(document, sort_keys=False), encoding="utf-8"
        )
        paths.append(destination)
    return paths


def test_staged_calibration_design_is_complete_disjoint_and_resource_exact():
    from hldbea.design_audit import audit_calibration_design

    report = audit_calibration_design(list(MANIFESTS))

    assert report.passes
    assert report.violations == ()
    assert report.run_count == 321
    assert report.total_evaluations == 642_000
    assert report.seed_set == (41001, 41002, 41003)
    assert report.objective_counts == (2, 3, 5, 8, 10, 15)
    assert set(report.manifest_hashes) == {path.name for path in MANIFESTS}
    assert all(len(digest) == 64 for digest in report.manifest_hashes.values())
    assert report.factor_grids["core"]["lambda"] == (
        0.0,
        0.01,
        0.05,
        0.1,
        0.2,
        0.5,
    )
    assert report.factor_grids["core"]["k"] == (0.5, 1.0, 2.0)
    assert report.factor_grids["restart"]["restart_theta_zero"] == (
        0.8,
        0.9,
        0.95,
        0.99,
    )
    assert report.factor_grids["restart"]["restart_window"] == (10, 20, 30, 50)
    assert report.factor_grids["restart"]["restart_delta_hv"] == (
        0.00001,
        0.0001,
        0.001,
    )
    assert report.factor_grids["restart"]["restart_fraction"] == (
        0.1,
        0.2,
        0.3,
        0.5,
    )
    assert report.factor_grids["cone"]["schedule_kinds"] == (
        "fixed",
        "linear",
        "saturating",
    )


def test_runner_dry_run_storage_estimates_match_design_audit(tmp_path):
    from hldbea.design_audit import audit_calibration_design

    report = audit_calibration_design(list(MANIFESTS))
    for manifest in MANIFESTS:
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "experiments" / "run.py"),
                "--manifest",
                str(manifest),
                "--artifact-root",
                str(tmp_path / "raw"),
                "--workers",
                "4",
                "--dry-run",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stderr
        plan = json.loads(result.stdout.splitlines()[0].removeprefix("PLAN "))
        expected = report.per_manifest[manifest.name]
        assert plan["planned"] == expected["run_count"]
        assert plan["estimated_max_bytes"] == expected["estimated_max_bytes"]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda name, doc: doc.update(seeds=[41001, 41002, 51001])
            if "core" in name
            else None,
            "validation seed",
        ),
        (
            lambda name, doc: doc.update(stage="validation")
            if "core" in name
            else None,
            "stage",
        ),
        (
            lambda name, doc: doc["execution"].update(save_history=True)
            if "core" in name
            else None,
            "history",
        ),
        (
            lambda name, doc: doc["execution"].update(evaluation_budget=2100)
            if "core" in name
            else None,
            "budget",
        ),
        (
            lambda name, doc: doc["algorithms"][1].update(id="nsga2")
            if "core" in name
            else None,
            "algorithm",
        ),
        (
            lambda name, doc: doc["algorithms"].__setitem__(
                slice(None),
                [
                    algorithm
                    for algorithm in doc["algorithms"]
                    if algorithm["variant"] != "lambda-0p5"
                ],
            )
            if "core" in name
            else None,
            "grid",
        ),
        (
            lambda name, doc: doc["algorithms"][7]["parameters"].update(
                **{"lambda": 0.2, "selection_mode": "global"}
            )
            if "core" in name
            else None,
            "OFAT",
        ),
        (
            lambda name, doc: doc["problems"][0].update(name="dtlz2")
            if "core" in name
            else None,
            "validation problem",
        ),
    ],
)
def test_design_audit_fails_closed_on_boundary_or_grid_violation(
    tmp_path, mutation, message
):
    from hldbea.design_audit import audit_calibration_design

    report = audit_calibration_design(_copied_manifests(tmp_path, mutation))

    assert not report.passes
    assert any(message.lower() in violation.lower() for violation in report.violations)


def test_design_audit_rejects_duplicate_run_ids():
    from hldbea.design_audit import audit_calibration_design

    report = audit_calibration_design([MANIFESTS[0], MANIFESTS[0], *MANIFESTS[1:]])

    assert not report.passes
    assert any("duplicate run" in item.lower() for item in report.violations)
