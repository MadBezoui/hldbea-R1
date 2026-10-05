from dataclasses import asdict
import json
from pathlib import Path
import shutil

import pytest
import yaml


def _record(problem, variant, hv, igd_plus, runtime, seed=41001):
    return {
        "problem_id": problem,
        "variant": variant,
        "seed": seed,
        "metrics": {"hv": hv, "igd_plus": igd_plus},
        "runtime_seconds": runtime,
    }


def _rank_fixture():
    return [
        _record("calibration-p1", "alpha", 3_000_000.0, 1.0, 3.0),
        _record("calibration-p1", "beta", 2_000_000.0, 2.0, 2.0),
        _record("calibration-p1", "gamma", 1_000_000.0, 3.0, 1.0),
        _record("calibration-p2", "alpha", 0.1, 0.3, 3.0),
        _record("calibration-p2", "beta", 0.3, 0.2, 2.0),
        _record("calibration-p2", "gamma", 0.2, 0.1, 1.0),
    ]


def test_ranking_orients_metrics_per_block_and_uses_declared_tie_breaks():
    from hldbea.calibration import rank_complete_blocks

    ranking = rank_complete_blocks(_rank_fixture())

    assert [item.variant for item in ranking] == ["beta", "alpha", "gamma"]
    assert ranking[0].median_rank == pytest.approx(0.5)
    assert ranking[0].worst_quartile_rank == pytest.approx(0.5)
    assert ranking[1].median_rank == pytest.approx(0.5)
    assert ranking[1].worst_quartile_rank == pytest.approx(0.0)
    assert ranking[0].block_count == 2
    assert ranking[0].metric_observations == 4


def test_ranking_uses_runtime_then_lexical_id_after_quality_ties():
    from hldbea.calibration import rank_complete_blocks

    records = [
        _record("calibration-p1", "zeta", 1.0, 1.0, 2.0),
        _record("calibration-p1", "beta", 1.0, 1.0, 1.0),
        _record("calibration-p1", "alpha", 1.0, 1.0, 1.0),
    ]

    ranking = rank_complete_blocks(records)

    assert [item.variant for item in ranking] == ["alpha", "beta", "zeta"]


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda rows: rows.pop(), "complete"),
        (lambda rows: rows.append(dict(rows[0])), "duplicate"),
        (
            lambda rows: rows[0]["metrics"].update(hv=float("nan")),
            "finite",
        ),
        (lambda rows: rows[0].update(seed=51001), "validation seed"),
        (
            lambda rows: rows[0].update(problem_id="validation-dtlz2-m3"),
            "validation problem",
        ),
    ],
)
def test_ranking_fails_closed_on_incomparable_or_leaking_records(mutator, message):
    from hldbea.calibration import rank_complete_blocks

    records = _rank_fixture()
    mutator(records)

    with pytest.raises(ValueError, match=message):
        rank_complete_blocks(records)


def test_top_selection_rejects_count_above_available_variants():
    from hldbea.calibration import rank_complete_blocks, select_top

    ranking = rank_complete_blocks(_rank_fixture())

    with pytest.raises(ValueError, match="available"):
        select_top(ranking, 4)


def test_confirmation_manifest_and_shortlist_json_are_byte_deterministic():
    from hldbea.calibration import (
        CandidateConfiguration,
        canonical_json_text,
        confirmation_manifest_text,
    )

    candidates = [
        CandidateConfiguration(
            variant="baseline",
            parameters={"k": 1.0, "lambda": 0.1},
            sources=("core:baseline",),
        ),
        CandidateConfiguration(
            variant="combined-best",
            parameters={"k": 0.5, "lambda": 0.2},
            sources=("core:k-0p5", "restart:window-30"),
        ),
    ]
    payload = {
        "schema_version": 1,
        "candidates": [asdict(candidate) for candidate in candidates],
    }

    assert confirmation_manifest_text(candidates) == confirmation_manifest_text(
        candidates
    )
    assert canonical_json_text(payload) == canonical_json_text(payload)
    document = yaml.safe_load(confirmation_manifest_text(candidates))
    assert document["seeds"] == [41001, 41002, 41003, 41004, 41005]
    assert [problem["id"] for problem in document["problems"]] == [
        "calibration-dtlz1-m3",
        "calibration-dtlz5-m5",
        "calibration-dtlz6-m10",
        "calibration-wfg1-m5",
        "calibration-wfg4-m10",
    ]
    assert document["execution"] == {
        "population_size": 100,
        "evaluation_budget": 10000,
        "checkpoints": [100, 500, 1000, 2000, 5000, 10000],
        "save_history": False,
    }


def _small_manifest(stage="calibration", seed=41001, problem="dtlz1"):
    return {
        "schema_version": 1,
        "manifest_id": "calibration-loader-test",
        "stage": stage,
        "seeds": [seed],
        "problems": [
            {
                "id": f"calibration-{problem}-m2",
                "name": problem,
                "n_obj": 2,
                "n_var": 6 if problem == "dtlz1" else 11,
                "parameters": {},
                "transform": {"kind": "identity"},
            }
        ],
        "algorithms": [
            {
                "id": "nsga2",
                "variant": "standard",
                "parameters": {
                    "crossover_probability": 0.9,
                    "crossover_eta": 20,
                    "mutation_probability": "inverse_n_var",
                    "mutation_eta": 20,
                },
            }
        ],
        "execution": {
            "population_size": 4,
            "evaluation_budget": 8,
            "checkpoints": [4, 8],
            "save_history": False,
        },
    }


def _write_manifest(path: Path, document):
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")


def test_loader_requires_analysis_valid_calibration_artifacts(tmp_path):
    from hldbea.calibration import load_validated_records
    from hldbea.run_spec import expand_manifest
    from hldbea.runner import execute_run

    document = _small_manifest()
    manifest = tmp_path / "manifest.yaml"
    _write_manifest(manifest, document)
    spec = expand_manifest(document)[0]
    outcome = execute_run(spec, tmp_path / "raw")

    records, provenance = load_validated_records(manifest, tmp_path / "raw")

    assert len(records) == 1
    assert records[0]["problem_id"] == "calibration-dtlz1-m2"
    assert records[0]["variant"] == "standard"
    assert set(provenance["artifact_hashes"]) == {spec.run_id}
    assert len(provenance["artifact_hashes"][spec.run_id]) == 64

    (outcome.artifact_path / "events.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="analysis validation"):
        load_validated_records(manifest, tmp_path / "raw")


@pytest.mark.parametrize(
    ("document", "message"),
    [
        (_small_manifest(stage="validation"), "stage"),
        (_small_manifest(seed=51001), "validation seed"),
        (_small_manifest(problem="dtlz2"), "validation problem"),
    ],
)
def test_loader_rejects_wrong_stage_or_validation_overlap(tmp_path, document, message):
    from hldbea.calibration import load_validated_records

    manifest = tmp_path / "manifest.yaml"
    _write_manifest(manifest, document)

    with pytest.raises(ValueError, match=message):
        load_validated_records(manifest, tmp_path / "raw")


def test_freezer_selects_ranked_variant_and_preserves_exact_parameters():
    from hldbea.calibration import freeze_configuration

    parameters = {
        "alpha": {"k": 1.0, "lambda": 0.1},
        "beta": {"k": 0.5, "lambda": 0.2},
        "gamma": {"k": 2.0, "lambda": 0.0},
    }

    frozen = freeze_configuration(_rank_fixture(), parameters)

    assert frozen.variant == "beta"
    assert frozen.parameters == parameters["beta"]
    assert frozen.selection_rule_version == "percentile-rank-v1"
