import hashlib
import json
from pathlib import Path

from hldbea.run_spec import expand_manifest, load_manifest


ROOT = Path(__file__).resolve().parents[1]
VALIDATION = ROOT / "results" / "reports" / "integration-smoke-validation.json"
TABLE_PROVENANCE = (
    ROOT / "results" / "tables" / "integration-smoke-tables-provenance.json"
)
FIGURE_PROVENANCE = (
    ROOT / "results" / "figures" / "integration-smoke-figures-provenance.json"
)
CALIBRATION = ROOT / "experiments" / "manifests" / "calibration-v1.yaml"
STAGED_CALIBRATION = tuple(
    ROOT / "experiments" / "manifests" / name
    for name in (
        "calibration-core-screen-v1.yaml",
        "calibration-restart-screen-v1.yaml",
        "calibration-cone-screen-v1.yaml",
    )
)
HARNESS_REPORT = ROOT / "docs" / "experiment-harness-report.md"
RESERVED_VALIDATION_SEEDS = set(range(51001, 51031))
RESERVED_VALIDATION_PROBLEMS = {"dtlz2", "dtlz3", "dtlz4", "wfg2", "wfg3", "wfg9"}


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_integration_validation_report_is_a_complete_analysis_gate():
    report = json.loads(VALIDATION.read_text(encoding="utf-8"))

    assert report["counts"] == {
        "planned": 16,
        "completed": 16,
        "valid": 16,
        "invalid": 0,
        "failed": 0,
        "missing": 0,
        "duplicate": 0,
    }
    assert report["gates"] == {"artifacts": True, "analysis": True}
    assert all(not findings for findings in report["issues"].values())
    assert set(report["evidence"]["geometry_hashes"]) == {
        "dtlz2-m3",
        "dtlz2-m5",
        "wfg2-m3",
        "wfg2-m5",
    }
    assert all(
        len(value) == 64
        for value in report["evidence"]["geometry_hashes"].values()
    )
    assert report["evidence"]["paired_seeds"]
    assert all(
        seeds == [31001, 31002]
        for seeds in report["evidence"]["paired_seeds"].values()
    )


def test_generated_tables_and_figures_have_validated_source_provenance():
    table = json.loads(TABLE_PROVENANCE.read_text(encoding="utf-8"))
    figure = json.loads(FIGURE_PROVENANCE.read_text(encoding="utf-8"))
    validation = json.loads(VALIDATION.read_text(encoding="utf-8"))

    for provenance in (table, figure):
        assert provenance["manifest_id"] == validation["manifest_id"]
        assert provenance["validation_gates"]["analysis"] is True
        assert len(provenance["source_run_ids"]) == validation["counts"]["valid"]
        assert len(set(provenance["source_run_ids"])) == len(
            provenance["source_run_ids"]
        )
        output_root = TABLE_PROVENANCE.parent if provenance is table else FIGURE_PROVENANCE.parent
        for filename, expected in provenance["outputs"].items():
            output = output_root / filename
            assert output.is_file()
            assert _sha256(output) == expected

    assert _sha256(ROOT / table["statistics_file"]) == table["statistics_sha256"]
    assert figure["selected_problem_objectives"] >= 5
    assert set(figure["median_run_ids"].values()) <= set(figure["source_run_ids"])


def test_calibration_manifest_is_disjoint_and_limited_to_predeclared_grids():
    manifest = load_manifest(CALIBRATION)
    specs = expand_manifest(manifest)

    assert manifest["stage"] == "calibration"
    assert set(manifest["seeds"]).isdisjoint(RESERVED_VALIDATION_SEEDS)
    assert {problem["name"] for problem in manifest["problems"]}.isdisjoint(
        RESERVED_VALIDATION_PROBLEMS
    )
    assert len(specs) == (
        len(manifest["seeds"])
        * len(manifest["problems"])
        * len(manifest["algorithms"])
    )
    baseline = manifest["algorithms"][0]["parameters"]
    allowed_grid_fields = {
        "k",
        "cone_epsilon",
        "local_search_max_iter",
        "restart_window",
        "restart_delta_hv",
    }
    for algorithm in manifest["algorithms"]:
        assert algorithm["id"] == "hldbea"
        changed = {
            key
            for key, value in algorithm["parameters"].items()
            if baseline.get(key) != value
        }
        assert changed <= allowed_grid_fields
        assert len(changed) <= 1


def test_staged_calibration_is_audited_before_any_screen_execution():
    from hldbea.design_audit import audit_calibration_design

    report = audit_calibration_design(list(STAGED_CALIBRATION))

    assert report.passes
    assert report.run_count == 321
    assert report.total_evaluations == 642_000
    assert set(report.seed_set).isdisjoint(RESERVED_VALIDATION_SEEDS)


def test_harness_report_records_reproduction_evidence_and_limitations():
    report = HARNESS_REPORT.read_text(encoding="utf-8")

    for required in (
        "## Acceptance result",
        "## Exact environment",
        "## Commands executed",
        "## Run counts and resource use",
        "## Dependency and comparator gates",
        "## Known limitations",
        "196 passed",
        "integration-smoke-v1",
        "calibration-v1",
    ):
        assert required in report
