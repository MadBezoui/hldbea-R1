import json
from pathlib import Path

import numpy as np
import pytest


def _summary(problem, algorithm, metric, direction, median, n=5):
    return {
        "problem_id": problem,
        "algorithm": algorithm,
        "metric": metric,
        "direction": direction,
        "n": n,
        "median": median,
        "q1": median - 0.1,
        "q3": median + 0.1,
        "iqr": 0.2,
        "ci_low": median - 0.2,
        "ci_high": median + 0.2,
        "confidence": 0.95,
        "bootstrap_samples": 1000,
        "bootstrap_seed": 7,
    }


def _statistics_document():
    summaries = [
        _summary("p1", "hldbea:full", "hv", "higher", 0.8),
        _summary("p1", "nsga2:standard", "hv", "higher", 0.6),
        _summary("p1", "hldbea:full", "igd_plus", "lower", 0.2),
        _summary("p1", "nsga2:standard", "igd_plus", "lower", 0.3),
    ]
    paired = [
        {
            "problem_id": "p1",
            "reference": "hldbea:full",
            "competitor": "nsga2:standard",
            "metric": metric,
            "direction": direction,
            "family": f"final-{metric}",
            "hypothesis": f"p1|hldbea:full-vs-nsga2:standard|{metric}",
            "p_raw": 0.01,
            "p_adjusted": 0.02,
            "reject_holm": True,
            "effect_size": {"a12": 0.75, "direction": direction},
        }
        for metric, direction in (("hv", "higher"), ("igd_plus", "lower"))
    ]
    return {
        "schema_version": 1,
        "manifest_id": "unit",
        "validation_passed": True,
        "reference_algorithm": "hldbea",
        "alpha": 0.05,
        "summaries": summaries,
        "paired_tests": paired,
        "friedman_tests": [],
    }


def test_best_cells_respect_metric_direction_and_ties():
    from hldbea.reporting import best_algorithms

    summaries = [
        _summary("p", "a", "hv", "higher", 0.8),
        _summary("p", "b", "hv", "higher", 0.6),
        _summary("p", "a", "igd_plus", "lower", 0.2),
        _summary("p", "b", "igd_plus", "lower", 0.2),
    ]

    assert best_algorithms(summaries, "p", "hv", "higher") == ("a",)
    assert best_algorithms(summaries, "p", "igd_plus", "lower") == ("a", "b")


@pytest.mark.parametrize(
    ("p_adjusted", "a12", "expected"),
    [(0.01, 0.7, "+"), (0.01, 0.3, "-"), (0.2, 0.9, "="), (0.01, 0.5, "=")],
)
def test_comparison_sign_semantics_are_consistent(p_adjusted, a12, expected):
    from hldbea.reporting import comparison_sign

    assert comparison_sign(p_adjusted, a12, alpha=0.05) == expected


def test_win_tie_loss_totals_and_median_run_tie_break():
    from hldbea.reporting import select_median_run, win_tie_loss

    assert win_tie_loss(["+", "=", "-", "+"]) == {
        "wins": 2,
        "ties": 1,
        "losses": 1,
    }
    selected = select_median_run(
        [
            {"run_id": "b", "hv": 0.2},
            {"run_id": "a", "hv": 0.2},
            {"run_id": "c", "hv": 0.9},
        ]
    )
    assert selected["run_id"] == "b"


def test_tables_are_derived_from_statistics_and_have_auditable_filenames(tmp_path):
    from hldbea.reporting import write_comparison_tables

    outputs = write_comparison_tables(
        _statistics_document(), tmp_path, stem="integration-smoke"
    )

    assert {path.name for path in outputs} == {
        "integration-smoke-results.csv",
        "integration-smoke-results.md",
        "integration-smoke-results.tex",
        "integration-smoke-results-summary.json",
    }
    markdown = (tmp_path / "integration-smoke-results.md").read_text()
    csv_text = (tmp_path / "integration-smoke-results.csv").read_text()
    summary = json.loads(
        (tmp_path / "integration-smoke-results-summary.json").read_text()
    )
    assert "0.8" in markdown and "0.6" in markdown
    assert "0.02" in csv_text and "0.75" in csv_text
    assert summary["win_tie_loss"] == {"wins": 2, "ties": 0, "losses": 0}
    assert all(path.stat().st_size > 0 for path in outputs)


def test_colorblind_safe_publication_figures_are_created_under_agg(tmp_path):
    from hldbea.reporting import (
        figure_filename,
        group_rotation_records,
        plot_box_violin,
        plot_convergence,
        plot_normalized_heatmap,
        plot_parallel_coordinates,
        plot_pca_projection,
        plot_radviz,
        plot_rotation_curve,
    )

    assert figure_filename("convergence", "DTLZ2 M5", "HV") == (
        "convergence__dtlz2-m5__hv.png"
    )
    population = np.array(
        [
            [0.1, 0.8, 0.3, 0.7, 0.2],
            [0.2, 0.7, 0.4, 0.6, 0.3],
            [0.3, 0.6, 0.5, 0.5, 0.4],
            [0.4, 0.5, 0.6, 0.4, 0.5],
            [0.5, 0.4, 0.7, 0.3, 0.6],
        ]
    )
    convergence = [
        {
            "algorithm": algorithm,
            "seed": seed,
            "evaluation": evaluation,
            "hv": value + 0.01 * seed,
        }
        for algorithm, value in (("hldbea", 0.3), ("nsga2", 0.2))
        for seed in (1, 2)
        for evaluation in (10, 20, 30)
    ]
    finals = [
        {"algorithm": "hldbea", "hv": 0.7},
        {"algorithm": "hldbea", "hv": 0.8},
        {"algorithm": "nsga2", "hv": 0.5},
        {"algorithm": "nsga2", "hv": 0.6},
    ]
    rotations = [
        {"problem_family": "dtlz2-m5", "algorithm": "hldbea", "angle": 0, "hv": 0.8},
        {"problem_family": "dtlz2-m5", "algorithm": "hldbea", "angle": 30, "hv": 0.7},
        {"problem_family": "dtlz2-m5", "algorithm": "nsga2", "angle": 0, "hv": 0.6},
        {"problem_family": "dtlz2-m5", "algorithm": "nsga2", "angle": 30, "hv": 0.5},
    ]
    grouped = group_rotation_records(
        rotations
        + [
            {
                "problem_family": "wfg2-m5",
                "algorithm": "hldbea",
                "angle": 0,
                "hv": 0.4,
            }
        ]
    )
    assert tuple(grouped) == ("dtlz2-m5", "wfg2-m5")
    assert len(grouped["dtlz2-m5"]) == 4
    outputs = [
        plot_convergence(convergence, tmp_path / "convergence.png", metric="hv"),
        plot_box_violin(finals, tmp_path / "box-violin.png", metric="hv"),
        plot_parallel_coordinates(population, tmp_path / "parallel.png"),
        plot_radviz(population, tmp_path / "radviz.png"),
        plot_pca_projection(population, tmp_path / "pca.png"),
        plot_normalized_heatmap(population, tmp_path / "heatmap.png"),
        plot_rotation_curve(rotations, tmp_path / "rotation.png", metric="hv"),
    ]

    assert all(path.is_file() and path.stat().st_size > 1_000 for path in outputs)


def test_rotation_descriptor_groups_rotated_front_angles_into_one_family():
    from types import SimpleNamespace

    from hldbea.reporting import rotation_experiment_descriptor

    def spec(angle):
        return SimpleNamespace(
            problem_name="rdtlz2",
            n_obj=3,
            n_var=12,
            problem_parameters={"angle_degrees": angle},
            transform={"kind": "identity"},
        )

    zero = rotation_experiment_descriptor(spec(0.0))
    thirty = rotation_experiment_descriptor(spec(30.0))

    assert zero is not None
    assert thirty is not None
    assert zero[0] == thirty[0]
    assert zero[1] == 0.0
    assert thirty[1] == 30.0


def test_mechanism_summary_reports_failure_and_charged_events_by_variant():
    from hldbea.reporting import summarize_mechanism_records

    rows = summarize_mechanism_records(
        [
            {
                "algorithm": "hldbea:full",
                "hv": 0.0,
                "igd_plus": 5.0,
                "solver_calls": 3,
                "solver_evaluations": 6,
                "solver_accepted": 1,
                "solver_gain": 0.2,
                "restart_triggered": 2,
                "restart_replacements": 10,
                "wall_seconds": 4.0,
            },
            {
                "algorithm": "hldbea:full",
                "hv": 0.4,
                "igd_plus": 3.0,
                "solver_calls": 1,
                "solver_evaluations": 2,
                "solver_accepted": 0,
                "solver_gain": 0.0,
                "restart_triggered": 0,
                "restart_replacements": 0,
                "wall_seconds": 2.0,
            },
        ]
    )

    assert rows == [
        {
            "algorithm": "hldbea:full",
            "n": 2,
            "nonzero_hv": 1,
            "nonzero_hv_rate": 0.5,
            "median_hv": 0.2,
            "median_igd_plus": 4.0,
            "solver_calls": 4,
            "solver_evaluations": 8,
            "solver_accepted": 1,
            "solver_accepted_gain": 0.2,
            "restart_triggers": 2,
            "restart_replacements": 10,
            "runs_with_restart": 1,
            "median_wall_seconds": 3.0,
        }
    ]
