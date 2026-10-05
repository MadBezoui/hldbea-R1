"""Auditable tables and colorblind-safe publication figures."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .run_spec import canonical_sha256
from .runtime import configure_headless_matplotlib
from .transforms import pca_coordinates


COLORBLIND_PALETTE = (
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
)


def _finite(value: Any, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")
    if not slug:
        raise ValueError("filename components must not be empty")
    return slug


def figure_filename(kind: str, *components: str, extension: str = "png") -> str:
    extension = extension.lower().lstrip(".")
    if extension not in {"png", "pdf", "svg"}:
        raise ValueError("unsupported figure extension")
    return "__".join(_slug(part) for part in (kind, *components)) + f".{extension}"


def best_algorithms(
    summaries: Sequence[Mapping[str, Any]],
    problem_id: str,
    metric: str,
    direction: str,
) -> tuple[str, ...]:
    if direction not in {"higher", "lower"}:
        raise ValueError("direction must be 'higher' or 'lower'")
    candidates = [
        item
        for item in summaries
        if item.get("problem_id") == problem_id and item.get("metric") == metric
    ]
    if not candidates:
        raise ValueError(f"no summaries for {problem_id}/{metric}")
    values = [_finite(item.get("median"), "median") for item in candidates]
    optimum = max(values) if direction == "higher" else min(values)
    return tuple(
        sorted(
            str(item["algorithm"])
            for item, value in zip(candidates, values)
            if np.isclose(value, optimum, rtol=1e-12, atol=1e-15)
        )
    )


def comparison_sign(p_adjusted: float, a12: float, *, alpha: float) -> str:
    p_adjusted = _finite(p_adjusted, "adjusted p-value")
    a12 = _finite(a12, "A12")
    alpha = _finite(alpha, "alpha")
    if not 0 <= p_adjusted <= 1 or not 0 <= a12 <= 1 or not 0 < alpha < 1:
        raise ValueError("p-value, A12, or alpha is outside its valid interval")
    if p_adjusted > alpha or np.isclose(a12, 0.5):
        return "="
    return "+" if a12 > 0.5 else "-"


def win_tie_loss(signs: Iterable[str]) -> dict[str, int]:
    totals = {"wins": 0, "ties": 0, "losses": 0}
    mapping = {"+": "wins", "=": "ties", "-": "losses"}
    for sign in signs:
        if sign not in mapping:
            raise ValueError(f"unknown comparison sign: {sign}")
        totals[mapping[sign]] += 1
    return totals


def select_median_run(runs: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    if not runs:
        raise ValueError("runs must not be empty")
    checked = []
    for run in runs:
        run_id = run.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("each run needs a non-empty run_id")
        checked.append((_finite(run.get("hv"), "run HV"), run_id, run))
    checked.sort(key=lambda item: (item[0], item[1]))
    return checked[(len(checked) - 1) // 2][2]


def group_rotation_records(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, list[Mapping[str, Any]]]:
    """Partition rotation observations so unrelated problems are never pooled."""

    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        family = record.get("problem_family")
        if not isinstance(family, str) or not family:
            raise ValueError("each rotation record needs a problem_family")
        grouped.setdefault(family, []).append(record)
    return {family: grouped[family] for family in sorted(grouped)}


def rotation_experiment_descriptor(spec) -> tuple[str, float] | None:
    """Return a common family key and angle for a rotation-design run."""

    parameters = dict(spec.problem_parameters)
    if spec.problem_name in {"rdtlz2", "rlinear"}:
        angle = parameters.pop("angle_degrees", None)
    else:
        transform = dict(spec.transform)
        if transform.get("kind") == "identity":
            angle = 0.0
        elif transform.get("kind") == "rotation":
            angle = transform.get("angle_degrees")
        else:
            return None
    if (
        not isinstance(angle, (int, float))
        or isinstance(angle, bool)
        or not np.isfinite(angle)
    ):
        return None
    family_document = {
        "name": spec.problem_name,
        "n_obj": spec.n_obj,
        "n_var": spec.n_var,
        "parameters": parameters,
    }
    family = (
        f"{spec.problem_name}-m{spec.n_obj}-n{spec.n_var}-"
        f"{canonical_sha256(family_document)[:8]}"
    )
    return family, float(angle)


def summarize_mechanism_records(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Aggregate solver/restart evidence without discarding failed runs."""

    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        algorithm = record.get("algorithm")
        if not isinstance(algorithm, str) or not algorithm:
            raise ValueError("each mechanism record needs an algorithm")
        grouped.setdefault(algorithm, []).append(record)
    rows = []
    for algorithm in sorted(grouped):
        items = grouped[algorithm]
        hv = np.asarray([_finite(item.get("hv"), "HV") for item in items])
        igd = np.asarray(
            [_finite(item.get("igd_plus"), "IGD+") for item in items]
        )
        wall = np.asarray(
            [_finite(item.get("wall_seconds"), "wall seconds") for item in items]
        )

        def total(field: str) -> int:
            values = [item.get(field) for item in items]
            if any(
                not isinstance(value, (int, np.integer))
                or isinstance(value, (bool, np.bool_))
                or value < 0
                for value in values
            ):
                raise ValueError(f"{field} values must be non-negative integers")
            return int(sum(values))

        restart_counts = [int(item["restart_triggered"]) for item in items]
        rows.append(
            {
                "algorithm": algorithm,
                "n": len(items),
                "nonzero_hv": int(np.count_nonzero(hv > 0.0)),
                "nonzero_hv_rate": float(np.mean(hv > 0.0)),
                "median_hv": float(np.median(hv)),
                "median_igd_plus": float(np.median(igd)),
                "solver_calls": total("solver_calls"),
                "solver_evaluations": total("solver_evaluations"),
                "solver_accepted": total("solver_accepted"),
                "solver_accepted_gain": float(
                    sum(_finite(item.get("solver_gain"), "solver gain") for item in items)
                ),
                "restart_triggers": total("restart_triggered"),
                "restart_replacements": total("restart_replacements"),
                "runs_with_restart": int(np.count_nonzero(restart_counts)),
                "median_wall_seconds": float(np.median(wall)),
            }
        )
    return rows


def _summary_lookup(document: Mapping[str, Any]) -> dict[tuple[str, str, str], Mapping[str, Any]]:
    summaries = document.get("summaries")
    if not isinstance(summaries, list):
        raise ValueError("statistics document has no summaries")
    lookup = {}
    for summary in summaries:
        if not isinstance(summary, Mapping):
            raise ValueError("statistics summaries must be mappings")
        key = (summary["problem_id"], summary["algorithm"], summary["metric"])
        if key in lookup:
            raise ValueError(f"duplicate statistical summary: {key}")
        lookup[key] = summary
    return lookup


def comparison_rows(document: Mapping[str, Any]) -> list[dict[str, Any]]:
    if document.get("validation_passed") is not True:
        raise ValueError("statistics document did not pass validation")
    summaries = document.get("summaries")
    paired = document.get("paired_tests")
    if not isinstance(paired, list):
        raise ValueError("statistics document has no paired tests")
    lookup = _summary_lookup(document)
    alpha = _finite(document.get("alpha"), "alpha")
    rows = []
    for comparison in paired:
        problem = comparison["problem_id"]
        metric = comparison["metric"]
        reference = comparison["reference"]
        competitor = comparison["competitor"]
        direction = comparison["direction"]
        left = lookup[(problem, reference, metric)]
        right = lookup[(problem, competitor, metric)]
        effect = comparison.get("effect_size")
        if not isinstance(effect, Mapping):
            raise ValueError("comparison effect size is missing")
        a12 = _finite(effect.get("a12"), "A12")
        p_adjusted = _finite(comparison.get("p_adjusted"), "adjusted p-value")
        best = best_algorithms(summaries, problem, metric, direction)
        rows.append(
            {
                "problem_id": problem,
                "metric": metric,
                "direction": direction,
                "reference": reference,
                "competitor": competitor,
                "n_reference": int(left["n"]),
                "n_competitor": int(right["n"]),
                "reference_median": float(left["median"]),
                "reference_q1": float(left["q1"]),
                "reference_q3": float(left["q3"]),
                "reference_ci_low": float(left["ci_low"]),
                "reference_ci_high": float(left["ci_high"]),
                "competitor_median": float(right["median"]),
                "competitor_q1": float(right["q1"]),
                "competitor_q3": float(right["q3"]),
                "competitor_ci_low": float(right["ci_low"]),
                "competitor_ci_high": float(right["ci_high"]),
                "reference_best": reference in best,
                "competitor_best": competitor in best,
                "p_raw": float(comparison["p_raw"]),
                "p_adjusted": p_adjusted,
                "a12": a12,
                "sign": comparison_sign(p_adjusted, a12, alpha=alpha),
                "family": comparison["family"],
            }
        )
    return sorted(rows, key=lambda row: (row["problem_id"], row["metric"], row["competitor"]))


def _estimate_text(row: Mapping[str, Any], prefix: str) -> str:
    return (
        f"{row[prefix + '_median']:.4g} "
        f"[{row[prefix + '_q1']:.4g}, {row[prefix + '_q3']:.4g}]; "
        f"CI [{row[prefix + '_ci_low']:.4g}, {row[prefix + '_ci_high']:.4g}]"
    )


def _latex_escape(value: str) -> str:
    return (
        value.replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
    )


def write_comparison_tables(
    statistics_document: Mapping[str, Any], output_dir: str | Path, *, stem: str
) -> tuple[Path, ...]:
    rows = comparison_rows(statistics_document)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    base = _slug(stem)
    csv_path = destination / f"{base}-results.csv"
    markdown_path = destination / f"{base}-results.md"
    latex_path = destination / f"{base}-results.tex"
    summary_path = destination / f"{base}-results-summary.json"

    fieldnames = list(rows[0]) if rows else []
    buffer = io.StringIO()
    if fieldnames:
        writer = csv.DictWriter(buffer, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    csv_path.write_text(buffer.getvalue(), encoding="utf-8")

    markdown = [
        "# Validated comparison results",
        "",
        "`+` means the reference is significantly better after Holm correction; "
        "`-` means significantly worse; `=` means no corrected difference.",
        "",
        "| Problem | Metric | Reference n, median [IQR]; 95% CI | Comparator n, median [IQR]; 95% CI | p raw | p Holm | A12 | Sign |",
        "|---|---|---|---|---:|---:|---:|:---:|",
    ]
    latex_rows = []
    for row in rows:
        left = _estimate_text(row, "reference")
        right = _estimate_text(row, "competitor")
        if row["reference_best"]:
            left = f"**{left}**"
        if row["competitor_best"]:
            right = f"**{right}**"
        markdown.append(
            f"| {row['problem_id']} | {row['metric']} | {row['reference']} "
            f"(n={row['n_reference']}): {left} | {row['competitor']} "
            f"(n={row['n_competitor']}): {right} | {row['p_raw']:.4g} | "
            f"{row['p_adjusted']:.4g} | {row['a12']:.3f} | {row['sign']} |"
        )
        latex_left = _latex_escape(
            f"{row['reference']}: {_estimate_text(row, 'reference')}"
        )
        latex_right = _latex_escape(
            f"{row['competitor']}: {_estimate_text(row, 'competitor')}"
        )
        if row["reference_best"]:
            latex_left = r"\textbf{" + latex_left + "}"
        if row["competitor_best"]:
            latex_right = r"\textbf{" + latex_right + "}"
        latex_rows.append(
            f"{_latex_escape(row['problem_id'])} & {_latex_escape(row['metric'])} & "
            f"{latex_left} & {latex_right} & {row['p_adjusted']:.4g} & "
            f"{row['a12']:.3f} & {row['sign']} \\\\"
        )
    totals = win_tie_loss(row["sign"] for row in rows)
    markdown.extend(
        [
            "",
            f"Reference totals: {totals['wins']} wins / {totals['ties']} ties / "
            f"{totals['losses']} losses.",
            "",
        ]
    )
    markdown_path.write_text("\n".join(markdown), encoding="utf-8")
    latex = [
        r"\begin{tabular}{llllrrc}",
        r"\toprule",
        r"Problem & Metric & Reference & Comparator & $p_{Holm}$ & $A_{12}$ & Sign \\",
        r"\midrule",
        *latex_rows,
        r"\bottomrule",
        r"\end{tabular}",
        "",
    ]
    latex_path.write_text("\n".join(latex), encoding="utf-8")
    summary_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "manifest_id": statistics_document.get("manifest_id"),
                "comparison_count": len(rows),
                "win_tie_loss": totals,
                "sign_semantics": {
                    "+": "reference significantly better after Holm correction",
                    "-": "reference significantly worse after Holm correction",
                    "=": "no significant corrected difference",
                },
            },
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return csv_path, markdown_path, latex_path, summary_path


def _pyplot():
    configure_headless_matplotlib()
    import matplotlib.pyplot as plt

    return plt


def _save(fig, output: str | Path) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    _pyplot().close(fig)
    return path


def _objective_matrix(F: np.ndarray) -> np.ndarray:
    values = np.asarray(F, dtype=float)
    if values.ndim != 2 or len(values) < 2 or values.shape[1] < 2:
        raise ValueError("F must have at least two rows and two objectives")
    if not np.all(np.isfinite(values)):
        raise ValueError("F must contain only finite values")
    return values


def _normalize_columns(F: np.ndarray) -> np.ndarray:
    values = _objective_matrix(F)
    low = np.min(values, axis=0)
    span = np.max(values, axis=0) - low
    return np.divide(
        values - low,
        span,
        out=np.full_like(values, 0.5),
        where=span > 0,
    )


def plot_convergence(
    records: Sequence[Mapping[str, Any]], output: str | Path, *, metric: str
) -> Path:
    if not records:
        raise ValueError("convergence records must not be empty")
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    algorithms = sorted({str(record["algorithm"]) for record in records})
    for index, algorithm in enumerate(algorithms):
        subset = [record for record in records if record["algorithm"] == algorithm]
        evaluations = sorted({int(record["evaluation"]) for record in subset})
        medians = [
            float(
                np.median(
                    [_finite(record[metric], metric) for record in subset if int(record["evaluation"]) == evaluation]
                )
            )
            for evaluation in evaluations
        ]
        ax.plot(
            evaluations,
            medians,
            marker="o",
            color=COLORBLIND_PALETTE[index % len(COLORBLIND_PALETTE)],
            label=algorithm,
        )
    ax.set(xlabel="Function evaluations", ylabel=metric, title=f"Convergence: {metric}")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    return _save(fig, output)


def plot_box_violin(
    records: Sequence[Mapping[str, Any]], output: str | Path, *, metric: str
) -> Path:
    if not records:
        raise ValueError("final metric records must not be empty")
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    algorithms = sorted({str(record["algorithm"]) for record in records})
    samples = [
        [_finite(record[metric], metric) for record in records if record["algorithm"] == algorithm]
        for algorithm in algorithms
    ]
    violin = ax.violinplot(samples, showmedians=True, showextrema=False)
    for index, body in enumerate(violin["bodies"]):
        body.set_facecolor(COLORBLIND_PALETTE[index % len(COLORBLIND_PALETTE)])
        body.set_alpha(0.55)
    ax.boxplot(samples, widths=0.18, showfliers=False)
    ax.set_xticks(range(1, len(algorithms) + 1), algorithms, rotation=20, ha="right")
    ax.set(ylabel=metric, title=f"Final {metric} distribution")
    ax.grid(axis="y", alpha=0.25)
    return _save(fig, output)


def plot_parallel_coordinates(F: np.ndarray, output: str | Path) -> Path:
    normalized = _normalize_columns(F)
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = np.arange(normalized.shape[1])
    for row in normalized:
        ax.plot(x, row, color=COLORBLIND_PALETTE[0], alpha=0.28, linewidth=1)
    ax.plot(x, np.median(normalized, axis=0), color=COLORBLIND_PALETTE[1], linewidth=2.5, label="Median")
    ax.set_xticks(x, [f"f{i + 1}" for i in x])
    ax.set(ylabel="Within-run normalized objective", ylim=(-0.03, 1.03), title="Parallel coordinates")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    return _save(fig, output)


def plot_radviz(F: np.ndarray, output: str | Path) -> Path:
    normalized = _normalize_columns(F)
    desirability = 1.0 - normalized
    angles = np.linspace(0.0, 2.0 * np.pi, normalized.shape[1], endpoint=False)
    anchors = np.column_stack((np.cos(angles), np.sin(angles)))
    denominator = np.sum(desirability, axis=1, keepdims=True)
    weights = np.divide(
        desirability,
        denominator,
        out=np.full_like(desirability, 1.0 / normalized.shape[1]),
        where=denominator > 0,
    )
    coordinates = weights @ anchors
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.scatter(coordinates[:, 0], coordinates[:, 1], s=28, color=COLORBLIND_PALETTE[0], alpha=0.72)
    ax.scatter(anchors[:, 0], anchors[:, 1], s=55, color=COLORBLIND_PALETTE[1], marker="D")
    for index, (x, y) in enumerate(anchors):
        ax.text(1.12 * x, 1.12 * y, f"f{index + 1}", ha="center", va="center")
    ax.set(aspect="equal", xlim=(-1.25, 1.25), ylim=(-1.25, 1.25), title="Radviz objective projection")
    ax.axis("off")
    return _save(fig, output)


def plot_pca_projection(F: np.ndarray, output: str | Path) -> Path:
    values = _objective_matrix(F)
    pca = pca_coordinates(values)
    coordinates = pca.transformed[:, :2]
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    ax.scatter(coordinates[:, 0], coordinates[:, 1], s=32, color=COLORBLIND_PALETTE[2], alpha=0.75)
    ax.set(xlabel="PC1", ylabel="PC2", title="PCA objective projection")
    ax.grid(alpha=0.22)
    return _save(fig, output)


def plot_normalized_heatmap(F: np.ndarray, output: str | Path) -> Path:
    normalized = _normalize_columns(F)
    order = np.argsort(np.sum(normalized, axis=1), kind="stable")
    displayed = normalized[order[:100]]
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(6.8, max(3.2, min(8.0, 0.16 * len(displayed)))))
    image = ax.imshow(displayed, aspect="auto", cmap="cividis", vmin=0, vmax=1)
    ax.set_xticks(np.arange(displayed.shape[1]), [f"f{i + 1}" for i in range(displayed.shape[1])])
    ax.set(xlabel="Objective", ylabel="Solutions (ordered)", title="Normalized objective heatmap")
    fig.colorbar(image, ax=ax, label="Within-run normalized value")
    return _save(fig, output)


def plot_rotation_curve(
    records: Sequence[Mapping[str, Any]], output: str | Path, *, metric: str
) -> Path:
    if not records:
        raise ValueError("rotation records must not be empty")
    families = {record.get("problem_family") for record in records}
    if len(families) != 1 or not all(
        isinstance(family, str) and family for family in families
    ):
        raise ValueError("a rotation curve must contain exactly one problem family")
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    algorithms = sorted({str(record["algorithm"]) for record in records})
    for index, algorithm in enumerate(algorithms):
        subset = [record for record in records if record["algorithm"] == algorithm]
        angles = sorted({float(record["angle"]) for record in subset})
        values = [
            float(np.median([_finite(record[metric], metric) for record in subset if float(record["angle"]) == angle]))
            for angle in angles
        ]
        ax.plot(angles, values, marker="o", color=COLORBLIND_PALETTE[index % len(COLORBLIND_PALETTE)], label=algorithm)
    ax.set(xlabel="Objective rotation angle (degrees)", ylabel=metric, title=f"Rotation sensitivity: {metric}")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    return _save(fig, output)
