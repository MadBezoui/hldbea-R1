import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_descriptive_summary_uses_median_linear_iqr_and_deterministic_bootstrap():
    from hldbea.statistics import summarize

    summary = summarize([1.0, 2.0, 3.0, 4.0], bootstrap_samples=2_000, seed=17)
    replay = summarize([1.0, 2.0, 3.0, 4.0], bootstrap_samples=2_000, seed=17)
    singleton = summarize([7.0], bootstrap_samples=100, seed=99)

    assert summary.n == 4
    assert summary.median == 2.5
    assert summary.q1 == 1.75
    assert summary.q3 == 3.25
    assert summary.iqr == 1.5
    assert summary == replay
    assert singleton.ci_low == singleton.ci_high == 7.0


def test_paired_wilcoxon_has_literal_result_and_direction_metadata():
    from hldbea.statistics import paired_wilcoxon

    result = paired_wilcoxon(
        {1: 2.0, 2: 4.0, 3: 6.0},
        {1: 1.0, 2: 2.0, 3: 3.0},
        direction="higher",
    )

    assert result.n_pairs == 3
    assert result.n_effective == 3
    assert result.statistic == 0.0
    assert result.p_value == 0.25
    assert result.zero_method == "wilcox"
    assert result.direction == "higher"
    assert result.status == "ok"


def test_paired_wilcoxon_all_zero_and_missing_pair_policies_are_explicit():
    from hldbea.statistics import paired_wilcoxon

    all_zero = paired_wilcoxon([1.0, 2.0], [1.0, 2.0], direction="lower")
    assert all_zero.statistic == 0.0
    assert all_zero.p_value == 1.0
    assert all_zero.n_effective == 0
    assert all_zero.status == "all_zero"

    with pytest.raises(ValueError, match="paired keys"):
        paired_wilcoxon({1: 1.0, 2: 2.0}, {1: 1.5}, direction="higher")
    omitted = paired_wilcoxon(
        {1: 1.0, 2: 2.0},
        {1: 1.5},
        direction="higher",
        failure_policy="omit",
    )
    assert omitted.n_pairs == 1
    assert omitted.missing_pairs == ("2",)


def test_wilcox_zero_policy_drops_zero_differences_before_exact_test():
    from hldbea.statistics import paired_wilcoxon

    result = paired_wilcoxon([1.0, 2.0], [1.0, 1.0], direction="higher")

    assert result.n_pairs == 2
    assert result.n_effective == 1
    assert result.statistic == 0.0
    assert result.p_value == 1.0


def test_holm_step_down_is_monotone_in_sorted_raw_p_values():
    from hldbea.statistics import holm_step_down

    adjusted = holm_step_down(
        {"h1": 0.01, "h2": 0.04, "h3": 0.03}, family="hv-confirmatory"
    )

    assert adjusted["h1"].p_raw == 0.01
    assert adjusted["h1"].p_adjusted == pytest.approx(0.03)
    assert adjusted["h3"].p_adjusted == pytest.approx(0.06)
    assert adjusted["h2"].p_adjusted == pytest.approx(0.06)
    assert all(item.family == "hv-confirmatory" for item in adjusted.values())


def test_vargha_delaney_handles_ties_and_metric_direction():
    from hldbea.statistics import vargha_delaney_a12

    higher = vargha_delaney_a12([3.0, 4.0], [1.0, 2.0], direction="higher")
    lower = vargha_delaney_a12([3.0, 4.0], [1.0, 2.0], direction="lower")
    tied = vargha_delaney_a12([1.0], [1.0], direction="higher")

    assert higher.a12 == 1.0
    assert lower.a12 == 0.0
    assert tied.a12 == 0.5
    assert higher.interpretation == "first_better"
    assert lower.interpretation == "second_better"


def test_friedman_uses_complete_blocks_and_reports_oriented_mean_ranks():
    from hldbea.statistics import friedman_complete_blocks

    result = friedman_complete_blocks(
        {
            "b1": {"a": 1.0, "b": 2.0, "c": 3.0},
            "b2": {"a": 2.0, "b": 1.0, "c": 3.0},
        },
        direction="lower",
    )

    assert result.n_blocks == 2
    assert result.algorithms == ("a", "b", "c")
    assert result.statistic == pytest.approx(3.0)
    assert result.p_value == pytest.approx(0.22313016014842982)
    assert result.mean_ranks == {"a": 1.5, "b": 1.5, "c": 3.0}

    with pytest.raises(ValueError, match="complete"):
        friedman_complete_blocks(
            {"b1": {"a": 1, "b": 2, "c": 3}, "b2": {"a": 2, "b": 1}},
            direction="lower",
        )


def test_metric_direction_registry_rejects_undeclared_metrics():
    from hldbea.statistics import metric_direction

    assert metric_direction("hv") == "higher"
    assert metric_direction("igd_plus") == "lower"
    with pytest.raises(ValueError, match="undeclared"):
        metric_direction("accuracy-ish")


def _document():
    common = {
        "crossover_probability": 0.9,
        "crossover_eta": 20,
        "mutation_probability": "inverse_n_var",
        "mutation_eta": 20,
    }
    return {
        "schema_version": 1,
        "manifest_id": "statistics-cli",
        "stage": "smoke",
        "seeds": [7, 19],
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
            {"id": name, "variant": "standard", "parameters": common}
            for name in ("nsga2", "spea2")
        ],
        "execution": {
            "population_size": 4,
            "evaluation_budget": 8,
            "checkpoints": [4, 8],
            "save_history": False,
        },
    }


def _run_cli(
    tmp_path,
    *,
    omit_last=False,
    document=None,
    reference_variant=None,
    metrics=None,
):
    from hldbea.run_spec import expand_manifest
    from hldbea.runner import execute_run

    document = _document() if document is None else document
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    specs = expand_manifest(document)
    for spec in specs[:-1] if omit_last else specs:
        assert execute_run(spec, tmp_path / "artifacts").status == "budget_exhausted"
    output = tmp_path / "statistics.json"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    command = [
            sys.executable,
            str(ROOT / "analysis" / "statistics.py"),
            "--manifest",
            str(manifest),
            "--artifact-root",
            str(tmp_path / "artifacts"),
            "--reference-algorithm",
            "nsga2",
            "--output",
            str(output),
            "--bootstrap-samples",
            "200",
        ]
    if reference_variant is not None:
        command.extend(["--reference-variant", reference_variant])
    if metrics is not None:
        command.extend(["--metrics", *metrics])
    result = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result, output


def test_statistics_cli_preserves_raw_and_holm_adjusted_p_values(tmp_path):
    result, output = _run_cli(tmp_path)

    assert result.returncode == 0, result.stderr
    document = json.loads(output.read_text(encoding="utf-8"))
    assert document["validation_gate"] == "analysis"
    assert document["validation_passed"] is True
    assert document["paired_tests"]
    for comparison in document["paired_tests"]:
        assert 0 <= comparison["p_raw"] <= 1
        assert comparison["p_raw"] <= comparison["p_adjusted"] <= 1
        assert comparison["family"]
        assert comparison["direction"] in {"higher", "lower"}


def test_statistics_cli_reads_score_diagnostics_as_declared_metrics(tmp_path):
    result, output = _run_cli(tmp_path, metrics=("phi_nz", "r_fz"))

    assert result.returncode == 0, result.stderr
    document = json.loads(output.read_text(encoding="utf-8"))
    assert document["metric_directions"] == {
        "phi_nz": "higher",
        "r_fz": "lower",
    }
    assert {item["metric"] for item in document["summaries"]} == {
        "phi_nz",
        "r_fz",
    }


def test_statistics_cli_refuses_invalid_or_unpaired_dataset(tmp_path):
    result, output = _run_cli(tmp_path, omit_last=True)

    assert result.returncode == 1
    assert "validation gate" in result.stderr
    assert not output.exists()


def test_statistics_cli_selects_reference_variant_for_calibration(tmp_path):
    document = _document()
    common = document["algorithms"][0]["parameters"]
    document["algorithms"] = [
        {"id": "nsga2", "variant": variant, "parameters": dict(common)}
        for variant in ("baseline", "candidate")
    ]

    result, output = _run_cli(
        tmp_path,
        document=document,
        reference_variant="baseline",
    )

    assert result.returncode == 0, result.stderr
    statistics = json.loads(output.read_text(encoding="utf-8"))
    assert statistics["reference_algorithm"] == "nsga2"
    assert statistics["reference_variant"] == "baseline"
    assert {
        (item["reference"], item["competitor"])
        for item in statistics["paired_tests"]
    } == {("nsga2:baseline", "nsga2:candidate")}


def test_friedman_fully_tied_blocks_report_no_difference():
    from hldbea.statistics import friedman_complete_blocks

    blocks = {seed: {"a": 0.0, "b": 0.0, "c": 0.0} for seed in range(5)}
    result = friedman_complete_blocks(blocks, direction="lower")
    assert result.statistic == 0.0
    assert result.p_value == 1.0
    assert result.mean_ranks == {"a": 2.0, "b": 2.0, "c": 2.0}
