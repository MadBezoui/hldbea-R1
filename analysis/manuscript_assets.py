"""Generate compact manuscript tables and figures from validated v2 artifacts.

Every number is read from immutable run artifacts and the statistics reports
written by ``analysis/statistics.py``; nothing is typed by hand.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

LABELS = {
    "hldbea:full": "HLDBEA",
    "hldbea:axis-sum": "HLDBEA (axis-sum, default)",
    "hldbea:axis-union": "axis-union",
    "hldbea:box-union": "box-union",
    "hldbea:knn-union": "kNN-union",
    "hldbea:objective-adaptive": "adaptive objective",
    "hldbea:selection-global": "global-rank survival",
    "hldbea:candidate-global-rank": "global-rank candidates",
    "hldbea:candidate-random": "random candidates",
    "hldbea:cone-strict": "HLDBEA strict",
    "hldbea:cone-fixed-0p1": "HLDBEA fixed $0.10$",
    "hldbea:cone-linear": "HLDBEA linear",
    "hldbea:cone-saturating": "HLDBEA saturating",
    "hldbea:core-no-ls": "core, no LS",
    "hldbea:core-with-ls": "core + LS",
    "hldbea:augmented-no-ls": "augmented, no LS",
    "hldbea:augmented-with-ls": "augmented + LS",
    "hldbea:augmented-restart-no-ls": "augmented + restart, no LS",
    "nsga2:standard": "NSGA-II",
    "nsga3:standard": "NSGA-III",
    "spea2:standard": "SPEA2",
    "rvea:standard": "RVEA",
    "moead:standard": "MOEA/D",
    "sms_emoa:standard": "SMS-EMOA",
    "age_moea:standard": "AGE-MOEA",
    "maoea_hap:upstream-default": "MaOEA-HAP",
    "fdsea:upstream-default": "FDSEA",
}


def _label(key: str) -> str:
    return LABELS.get(key, key)


def _problem(problem_id: str) -> str:
    name = problem_id.replace("validation-", "")
    if name.startswith(("rdtlz2-m3-a", "rlinear-m3-a")):
        family = "RDTLZ2" if name.startswith("rdtlz2") else "Rotated linear"
        return f"{family} ${name.split('-a')[-1]}^\\circ$"
    family, _, m = name.partition("-m")
    return f"{family.upper()} ($m={m}$)"


def _fmt(value: float) -> str:
    if value == 0:
        return "0"
    magnitude = abs(value)
    if magnitude >= 100:
        return f"{value:.1f}"
    if magnitude >= 1:
        return f"{value:.3f}"
    return f"{value:.4f}" if magnitude >= 0.01 else f"{value:.2e}"


def load_runs(artifact_root: Path, manifest: str) -> list[dict]:
    runs = []
    for metadata_path in sorted((artifact_root / manifest).glob("*/metadata.json")):
        metadata = json.loads(metadata_path.read_text())
        spec = metadata["spec"]
        run = metadata["run_metadata"]
        events = json.loads((metadata_path.parent / "events.json").read_text())
        local = events.get("local_search", [])
        runs.append(
            {
                "dir": metadata_path.parent,
                "problem": spec["problem"]["id"],
                "label": f"{spec['algorithm']['id']}:{spec['algorithm']['variant']}",
                "seed": spec["seed"],
                "run_id": metadata["run_id"],
                "hv": float(run["metrics"]["hv"]),
                "igd_plus": float(run["metrics"]["igd_plus"]),
                "values": {
                    **{k: float(v) for k, v in run["metrics"].items()},
                    **{k: float(v) for k, v in run["score_diagnostics"].items()},
                },
                "budget": int(spec["execution"]["evaluation_budget"]),
                "geometry": run["geometry"],
                "solver_calls": len(local),
                "solver_fe": int(sum(int(e["evaluations"]) for e in local)),
                "solver_accepted": int(sum(bool(e["accepted"]) for e in local)),
                "solver_moved": int(
                    sum(
                        bool(e["accepted"])
                        and not np.allclose(e["f_before"], e["f_after"])
                        for e in local
                    )
                ),
            }
        )
    return runs


class AssetError(RuntimeError):
    """Raised when a block is missing, invalid or inconsistent with its report."""


def load_validated_block(manifest_path: Path, artifact_root: Path, statistics_path: Path):
    """Load a block only if it passes validation and matches its statistics."""

    from hldbea.validation import validate_dataset

    if not manifest_path.exists():
        raise AssetError(f"missing manifest {manifest_path}")
    report = validate_dataset(manifest_path, artifact_root)
    if not report.passes("analysis"):
        raise AssetError(f"{manifest_path.stem} fails the analysis validation gate")
    if not statistics_path.exists():
        raise AssetError(f"missing statistics report {statistics_path}")
    statistics = json.loads(statistics_path.read_text())
    if statistics.get("manifest_id") != manifest_path.stem or not statistics.get(
        "validation_passed"
    ):
        raise AssetError(f"{statistics_path.name} does not belong to {manifest_path.stem}")
    if report.valid != report.planned:
        raise AssetError(f"{manifest_path.stem}: {report.valid} of {report.planned} runs are valid")
    valid_ids = set(report.evidence["valid_run_ids"])
    runs = [run for run in load_runs(artifact_root, manifest_path.stem)
            if run["dir"].name in valid_ids]
    if len(runs) != report.planned:
        raise AssetError(f"{manifest_path.stem}: run inventory does not match the manifest")
    verify_statistics(runs, statistics, statistics_path.name)
    return runs


def verify_statistics(runs, statistics, name):
    """Check that a statistical report was computed from exactly these runs.

    The observation digest binds the report to the analysed values, and the
    medians, paired Wilcoxon p-values and Holm decisions used by the tables are
    recomputed from the runs and compared with the report.
    """

    from hldbea.statistics import holm_step_down, observations_digest, paired_wilcoxon

    metrics = sorted(statistics["metric_directions"])
    records = [
        (run["run_id"], run["problem"], run["label"], run["seed"],
         {m: run["values"][m] for m in metrics})
        for run in runs
    ]
    if observations_digest(records) != statistics.get("observations_digest"):
        raise AssetError(f"{name}: observation digest does not match the validated runs")
    by_key = {(run["problem"], run["label"], run["seed"]): run["values"] for run in runs}
    for item in statistics["summaries"]:
        values = [v[item["metric"]] for (p, l, _), v in by_key.items()
                  if p == item["problem_id"] and l == item["algorithm"]]
        if len(values) != item["n"] or not np.isclose(np.median(values), item["median"], rtol=0, atol=1e-12):
            raise AssetError(f"{name}: summary for {item['problem_id']} {item['algorithm']} differs")
    families = defaultdict(dict)
    for test in statistics["paired_tests"]:
        seeds = sorted(seed for (p, l, seed) in by_key
                       if p == test["problem_id"] and l == test["reference"])
        first = {s: by_key[(test["problem_id"], test["reference"], s)][test["metric"]] for s in seeds}
        second = {s: by_key[(test["problem_id"], test["competitor"], s)][test["metric"]] for s in seeds}
        result = paired_wilcoxon(first, second, direction=test["direction"],
                                 zero_method="wilcox", failure_policy="raise",
                                 alternative="two-sided")
        if not np.isclose(result.p_value, test["p_raw"], rtol=1e-12, atol=0):
            raise AssetError(f"{name}: p-value of {test['hypothesis']} differs")
        families[test["family"]][test["hypothesis"]] = result.p_value
    decisions = {}
    for family, values in families.items():
        for hypothesis, outcome in holm_step_down(values, family=family, alpha=statistics["alpha"]).items():
            decisions[(family, hypothesis)] = outcome.reject
    for test in statistics["paired_tests"]:
        if decisions[(test["family"], test["hypothesis"])] != test["reject_holm"]:
            raise AssetError(f"{name}: Holm decision of {test['hypothesis']} differs")


def publish(staging: Path, final: Path) -> None:
    """Replace the output directory only after every asset was written."""

    if final.exists():
        backup = final.with_name(final.name + ".previous")
        if backup.exists():
            shutil.rmtree(backup)
        final.rename(backup)
        staging.rename(final)
        for keep in ("README.md",):
            if (backup / keep).exists() and not (final / keep).exists():
                shutil.copy2(backup / keep, final / keep)
        shutil.rmtree(backup)
    else:
        staging.rename(final)


def load_signs(statistics_path: Path) -> tuple[str, dict]:
    """Map (problem, competitor, metric) -> sign of the competitor vs reference."""

    report = json.loads(statistics_path.read_text())
    reference = f"{report['reference_algorithm']}:{report['reference_variant']}"
    signs = {}
    for test in report["paired_tests"]:
        if not test["reject_holm"]:
            sign = "\\approx"
        elif test["effect_size"]["interpretation"] == "first_better":
            sign = "-"  # reference significantly better than competitor
        else:
            sign = "+"
        signs[(test["problem_id"], test["competitor"], test["metric"])] = sign
    return reference, signs


def medians(runs: list[dict]) -> dict:
    grouped = defaultdict(list)
    for run in runs:
        grouped[(run["problem"], run["label"])].append(run)
    summary = {}
    for key, items in grouped.items():
        hv = np.array([item["hv"] for item in items])
        igd = np.array([item["igd_plus"] for item in items])
        summary[key] = {
            "n": len(items),
            "hv": float(np.median(hv)),
            "hv_q": (float(np.quantile(hv, 0.25)), float(np.quantile(hv, 0.75))),
            "igd_plus": float(np.median(igd)),
            "igd_q": (float(np.quantile(igd, 0.25)), float(np.quantile(igd, 0.75))),
            "hv_positive": int(np.sum(hv > 0)),
        }
    return summary


def comparison_table(runs, statistics_path, order, caption, label, positive=False):
    reference, signs = load_signs(statistics_path)
    summary = medians(runs)
    problems = sorted({problem for problem, _ in summary})
    columns = "|l|l|r|c|r|c|" + ("r|" if positive else "")
    lines = [
        "\\begin{table}[!htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\small",
        "\\setlength{\\tabcolsep}{3pt}",
        f"\\begin{{tabular}}{{{columns}}}",
        "\\hline",
        "\\textbf{Problem} & \\textbf{Algorithm} & \\textbf{HV} & \\textbf{vs ref.} & \\textbf{IGD$^+$} & \\textbf{vs ref.} "
        + ("& \\textbf{HV$>0$} " if positive else "")
        + "\\\\ \\hline",
    ]
    for problem in problems:
        present = [key for key in order if (problem, key) in summary]
        best_hv = max(summary[(problem, key)]["hv"] for key in present)
        best_igd = min(summary[(problem, key)]["igd_plus"] for key in present)
        for index, key in enumerate(present):
            row = summary[(problem, key)]
            hv = _fmt(row["hv"])
            igd = _fmt(row["igd_plus"])
            if row["hv"] == best_hv and best_hv > 0:
                hv = f"\\textbf{{{hv}}}"
            if row["igd_plus"] == best_igd:
                igd = f"\\textbf{{{igd}}}"
            if key == reference:
                hv_sign = igd_sign = "ref."
            else:
                hv_sign = f"${signs.get((problem, key, 'hv'), '?')}$"
                igd_sign = f"${signs.get((problem, key, 'igd_plus'), '?')}$"
            cells = [
                _problem(problem) if index == 0 else "",
                _label(key),
                hv,
                hv_sign,
                igd,
                igd_sign,
            ]
            if positive:
                cells.append(f"{row['hv_positive']}/{row['n']}")
            lines.append(" & ".join(cells) + " \\\\")
        lines.append("\\hline")
    lines += ["\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def mechanism_rows(blocks: dict[str, list[dict]]) -> tuple[str, dict]:
    lines = [
        "\\begin{table}[!htbp]",
        "\\centering",
        "\\caption{Audited deterministic-track activity of every HLDBEA configuration in the "
        "confirmatory blocks (30 paired seeds per configuration). \\emph{Moved} counts accepted "
        "SLSQP iterates whose objective vector differs from the starting point, and the FE share is "
        "relative to the total charged budget.}",
        "\\label{tab:solver_audit}",
        "\\small",
        "\\begin{tabular}{|l|r|r|r|r|r|}",
        "\\hline",
        "\\textbf{Block} & \\textbf{Runs} & \\textbf{SLSQP calls} & \\textbf{FE share} & "
        "\\textbf{Accepted} & \\textbf{Moved} \\\\ \\hline",
    ]
    numbers = {}
    for name, runs in blocks.items():
        own = [run for run in runs if run["label"].startswith("hldbea:") and run["solver_calls"]]
        if not own:
            continue
        calls = sum(run["solver_calls"] for run in own)
        fe = sum(run["solver_fe"] for run in own)
        budget = sum(run["budget"] for run in own)
        accepted = sum(run["solver_accepted"] for run in own)
        moved = sum(run["solver_moved"] for run in own)
        numbers[name] = {
            "runs": len(own),
            "calls": calls,
            "fe_share": fe / budget,
            "accepted": accepted,
            "moved": moved,
        }
        lines.append(
            f"{name} & {len(own)} & {calls:,} & {100 * fe / budget:.1f}\\% & "
            f"{accepted:,} & {moved:,} \\\\".replace(",", "{,}")
        )
    lines += ["\\hline", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines), numbers


def rotation_figure(runs, output: Path) -> dict:
    summary = medians(runs)
    prefix = next(iter(summary))[0].rsplit("-a", 1)[0]
    angles = sorted({int(p.split("-a")[-1]) for p, _ in summary})
    labels = ["hldbea:full", "nsga2:standard", "nsga3:standard"]
    styles = {"hldbea:full": "o-", "nsga2:standard": "s--", "nsga3:standard": "^-."}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    values = {}
    for metric, axis, title in (
        ("hv", axes[0], "HV (higher is better)"),
        ("igd_plus", axes[1], "IGD$^+$ (lower is better)"),
    ):
        for key in labels:
            rows = [summary[(f"{prefix}-a{a}", key)] for a in angles]
            med = np.array([row[metric] for row in rows])
            quart = np.array([row["hv_q" if metric == "hv" else "igd_q"] for row in rows])
            axis.errorbar(
                angles,
                med,
                yerr=np.abs(quart.T - med),
                fmt=styles[key],
                capsize=3,
                label=_label(key),
            )
            values.setdefault(key, {})[metric] = med.tolist()
        axis.set_xticks(angles)
        axis.set_xlabel("Rotation angle (degrees)")
        axis.set_title(title, fontsize=9)
        axis.grid(alpha=0.3)
    axes[1].set_yscale("log")
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
    return {"angles": angles, "medians": values}


def rotation_invariant_table(runs, output: Path) -> dict:
    """Euclidean IGD to the rotated reference front, which a rigid rotation leaves unchanged."""

    from hldbea.metrics import build_reference_geometry
    from hldbea.problems import ProblemSpec
    from hldbea.statistics import holm_step_down, paired_wilcoxon

    references = {}
    igd = {}
    for run in runs:
        problem = run["problem"]
        if problem not in references:
            spec = json.loads((run["dir"] / "metadata.json").read_text())["spec"]["problem"]
            references[problem] = np.asarray(
                build_reference_geometry(ProblemSpec(**spec)).reference_set, dtype=float
            )
        front = np.load(run["dir"] / "arrays.npz")["F"]
        distances = np.linalg.norm(references[problem][:, None, :] - front[None, :, :], axis=2)
        igd[(problem, run["label"], run["seed"])] = float(np.mean(np.min(distances, axis=1)))
    problems = sorted(references, key=lambda p: int(p.rsplit("-a", 1)[-1]))
    reference = "hldbea:full"
    competitors = ["nsga2:standard", "nsga3:standard"]
    pvalues, directions = {}, {}
    for problem in problems:
        seeds = sorted(s for (p, l, s) in igd if p == problem and l == reference)
        for competitor in competitors:
            first = {s: igd[(problem, reference, s)] for s in seeds}
            second = {s: igd[(problem, competitor, s)] for s in seeds}
            result = paired_wilcoxon(first, second, direction="lower", zero_method="wilcox",
                                     failure_policy="raise", alternative="two-sided")
            hypothesis = f"{problem}|{competitor}"
            pvalues[hypothesis] = result.p_value
            directions[hypothesis] = np.median(list(second.values())) - np.median(list(first.values()))
    holm = holm_step_down(pvalues, family="rotation-euclidean-igd", alpha=0.05)
    def median(problem, label):
        return float(np.median([v for (p, l, _), v in igd.items() if p == problem and l == label]))
    lines = [
        "\\begin{table}[!htbp]", "\\centering",
        "\\caption{Euclidean IGD to the rotated reference front, which a rigid rotation leaves "
        "unchanged, for the rotated linear front. Values are medians over 30 paired seeds. "
        "Signs compare each baseline with HLDBEA by paired Wilcoxon tests with Holm correction "
        "over the ten comparisons: $+$ baseline significantly better, $-$ significantly worse, "
        "$\\approx$ no significant difference.}",
        "\\label{tab:rotation_igd}", "\\small",
        "\\begin{tabular}{|l|r|rc|rc|}", "\\hline",
        "\\textbf{Angle} & \\textbf{HLDBEA} & \\textbf{NSGA-II} & & \\textbf{NSGA-III} & \\\\ \\hline",
    ]
    numbers = {}
    for problem in problems:
        angle = problem.rsplit("-a", 1)[-1]
        cells = [f"${angle}^\\circ$", _fmt(median(problem, reference))]
        numbers[angle] = {reference: median(problem, reference)}
        for competitor in competitors:
            hypothesis = f"{problem}|{competitor}"
            if not holm[hypothesis].reject:
                sign = "\\approx"
            else:
                sign = "+" if directions[hypothesis] < 0 else "-"
            cells += [_fmt(median(problem, competitor)), f"${sign}$"]
            numbers[angle][competitor] = median(problem, competitor)
            numbers[angle][competitor + "|p_holm"] = holm[hypothesis].p_adjusted
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\hline", "\\end{tabular}", "\\end{table}", ""]
    output.write_text("\n".join(lines))
    return numbers


def dtlz3_figures(runs, convergence_output: Path, distribution_output: Path) -> None:
    order = [
        "hldbea:core-no-ls", "hldbea:core-with-ls", "hldbea:augmented-no-ls",
        "hldbea:augmented-with-ls", "hldbea:augmented-restart-no-ls",
        "hldbea:full", "nsga3:standard",
    ]
    trajectories = defaultdict(list)
    finals = defaultdict(list)
    for run in runs:
        points = json.loads((run["dir"] / "checkpoints.json").read_text())
        trajectories[run["label"]].append(
            [(c["target_evaluation"], c["igd_plus"]) for c in points]
        )
        finals[run["label"]].append(run["igd_plus"])
    fig, axis = plt.subplots(figsize=(6.0, 3.6))
    for key in order:
        series = np.array(trajectories[key])
        evaluations = series[0, :, 0]
        values = series[:, :, 1]
        median = np.median(values, axis=0)
        axis.plot(evaluations, median, marker="o", markersize=3, label=_label(key).replace("$", ""))
        axis.fill_between(
            evaluations, np.quantile(values, 0.25, axis=0),
            np.quantile(values, 0.75, axis=0), alpha=0.12,
        )
    axis.set_yscale("log")
    axis.set_xlabel("Objective evaluations")
    axis.set_ylabel("IGD$^+$")
    axis.grid(alpha=0.3)
    axis.legend(fontsize=6, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(convergence_output)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(6.0, 3.6))
    axis.boxplot([finals[key] for key in order], showfliers=True)
    axis.set_xticks(range(1, len(order) + 1))
    axis.set_xticklabels([_label(key).replace("$", "") for key in order], rotation=30, ha="right", fontsize=7)
    axis.set_yscale("log")
    axis.set_ylabel("Final IGD$^+$")
    axis.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(distribution_output)
    plt.close(fig)


def manyobjective_figure(runs, output: Path) -> dict:
    labels = [
        "hldbea:cone-strict",
        "hldbea:cone-fixed-0p1",
        "nsga3:standard",
        "maoea_hap:upstream-default",
    ]
    problems = ["validation-dtlz2-m10", "validation-wfg2-m15"]
    fig, axes = plt.subplots(
        len(problems), len(labels), figsize=(9.0, 4.2), sharey="row"
    )
    chosen = {}
    for row, problem in enumerate(problems):
        for col, key in enumerate(labels):
            items = sorted(
                (run for run in runs if run["problem"] == problem and run["label"] == key),
                key=lambda run: run["hv"],
            )
            median_run = items[len(items) // 2]
            chosen[f"{problem}|{key}"] = {"seed": median_run["seed"], "hv": median_run["hv"]}
            front = np.load(median_run["dir"] / "arrays.npz")["F"]
            ideal = np.asarray(median_run["geometry"]["ideal"], dtype=float)
            nadir = np.asarray(median_run["geometry"]["nadir"], dtype=float)
            scaled = (front - ideal) / np.where(nadir > ideal, nadir - ideal, 1.0)
            axis = axes[row, col]
            positions = np.arange(1, scaled.shape[1] + 1)
            for vector in scaled:
                axis.plot(positions, vector, color="tab:blue", alpha=0.15, linewidth=0.6)
            axis.plot(positions, np.median(scaled, axis=0), color="tab:orange", linewidth=1.5)
            axis.axhline(1.0, color="black", linewidth=0.6, linestyle=":")
            axis.set_ylim(-0.05, 2.0)
            axis.set_xticks(positions[:: max(1, len(positions) // 5)])
            axis.tick_params(labelsize=6)
            if row == 0:
                axis.set_title(_label(key).replace("$", ""), fontsize=8)
            if col == 0:
                axis.set_ylabel(_problem(problem).replace("$", ""), fontsize=8)
            axis.text(
                0.97, 0.95, f"HV={median_run['hv']:.3g}", transform=axis.transAxes,
                ha="right", va="top", fontsize=6,
            )
    fig.supxlabel("Objective index", fontsize=8)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)
    return chosen


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=Path("results/raw"))
    parser.add_argument("--reports", type=Path, default=Path("results/reports"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifests", type=Path, default=Path("experiments/manifests"))
    parser.add_argument("--version", default="v3")
    args = parser.parse_args(argv)
    final_out = args.output_dir
    final_out.parent.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix=".assets-", dir=final_out.parent))
    version = args.version

    def manifest(block):
        return f"reviewer-{block}-validation-{version}"

    def stats(block):
        return args.reports / f"{manifest(block)}-statistics.json"

    blocks = {}
    numbers = {}
    sign_note = (
        "Values are medians over 30 paired seeds, and bold marks the best median. "
        "Signs compare each row with the reference row (ref.) by paired Wilcoxon tests with "
        "Holm correction at $\\alpha=0.05$: $+$ significantly better, $-$ significantly worse, "
        "$\\approx$ no significant difference."
    )

    specs = [
        (
            "modern-benchmarks",
            [
                "hldbea:full", "maoea_hap:upstream-default", "fdsea:upstream-default",
                "nsga2:standard", "nsga3:standard", "spea2:standard", "rvea:standard",
                "moead:standard", "sms_emoa:standard", "age_moea:standard",
            ],
            "Equal-budget IMOP validation with post-2024 and indicator-based baselines "
            "(20{,}000 FEs, population 100). " + sign_note,
            "tab:imop_recent",
            False,
        ),
        (
            "geometry-ablation",
            [
                "hldbea:axis-sum", "hldbea:axis-union", "hldbea:box-union",
                "hldbea:knn-union", "hldbea:objective-adaptive", "hldbea:selection-global",
                "hldbea:candidate-global-rank", "hldbea:candidate-random",
            ],
            "Controlled neighborhood, objective-policy and survivor-selection ablation "
            "(20{,}000 FEs). Each row changes one component of the default. " + sign_note,
            "tab:geometry_ablation",
            False,
        ),
        (
            "manyobjective",
            [
                "hldbea:cone-fixed-0p1", "hldbea:cone-strict", "hldbea:cone-linear",
                "hldbea:cone-saturating", "nsga3:standard", "rvea:standard",
                "maoea_hap:upstream-default", "fdsea:upstream-default",
            ],
            "Many-objective validation at $m\\in\\{10,15\\}$ (10{,}000 FEs, population 100). "
            "HV is estimated with a fixed 65{,}536-point scrambled Sobol sample. " + sign_note,
            "tab:manyobjective",
            False,
        ),
        (
            "rotation",
            ["hldbea:full", "nsga2:standard", "nsga3:standard"],
            "Rotated linear front ($m=3$, 10{,}000 FEs). The angle rotates the triangular "
            "Pareto front rigidly within its own plane, so the Pareto set and the shape of "
            "the front are the same at every angle. " + sign_note,
            "tab:rotation",
            False,
        ),
        (
            "dtlz3",
            [
                "hldbea:core-no-ls", "hldbea:core-with-ls", "hldbea:augmented-no-ls",
                "hldbea:augmented-with-ls", "hldbea:augmented-restart-no-ls",
                "hldbea:full", "nsga3:standard",
            ],
            "Audited DTLZ3 component validation (46{,}000 FEs). " + sign_note,
            "tab:dtlz3_remedy",
            True,
        ),
    ]
    for block, order, caption, label, positive in specs:
        runs = load_validated_block(
            args.manifests / f"{manifest(block)}.yaml",
            args.artifact_root,
            stats(block),
        )
        blocks[block] = runs
        table = comparison_table(runs, stats(block), order, caption, label, positive)
        (out / f"table-{block}.tex").write_text(table)
        numbers[block] = {
            f"{problem}|{key}": value
            for (problem, key), value in medians(runs).items()
        }
    names = {
        "dtlz3": "DTLZ3 components",
        "modern-benchmarks": "IMOP baselines",
        "geometry-ablation": "Geometry ablation",
        "manyobjective": "Many-objective",
        "rotation": "Rotation",
    }
    solver_table, solver_numbers = mechanism_rows(
        {names[block]: runs for block, runs in blocks.items()}
    )
    (out / "table-solver-audit.tex").write_text(solver_table)
    numbers["solver_audit"] = solver_numbers
    numbers["rotation_euclidean_igd"] = rotation_invariant_table(
        blocks["rotation"], out / "table-rotation-igd.tex"
    )
    if "rotation" in blocks:
        numbers["rotation_figure"] = rotation_figure(
            blocks["rotation"], out / "rotation-sensitivity.pdf"
        )
    dtlz3_figures(
        blocks["dtlz3"],
        out / "dtlz3-validation-igd-convergence.pdf",
        out / "dtlz3-validation-igd-distribution.pdf",
    )
    if "manyobjective" in blocks:
        numbers["manyobjective_figure"] = manyobjective_figure(
            blocks["manyobjective"], out / "manyobjective-parallel-coordinates.pdf"
        )
    (out / "manuscript-numbers.json").write_text(
        json.dumps(numbers, indent=2, sort_keys=True, default=str)
    )
    publish(out, final_out)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssetError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
