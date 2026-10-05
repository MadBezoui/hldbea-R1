#!/usr/bin/env python3
"""Run the predeclared final-metric statistical analysis on valid artifacts."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import uuid

from hldbea.artifacts import artifact_path
from hldbea.run_spec import expand_manifest, load_manifest
from hldbea.statistics import (
    friedman_complete_blocks,
    holm_step_down,
    matched_pairs_rank_biserial,
    metric_direction,
    observations_digest,
    paired_wilcoxon,
    summarize,
    vargha_delaney_a12,
)
from hldbea.validation import validate_dataset


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze a complete, validation-gated experiment dataset."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--reference-algorithm", default="hldbea")
    parser.add_argument(
        "--reference-variant",
        help="exact variant to use when the reference algorithm has several variants",
    )
    parser.add_argument("--metrics", nargs="+", default=("hv", "igd_plus"))
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261003)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--output", type=Path)
    return parser


def _read_metadata(path: Path) -> dict:
    return json.loads((path / "metadata.json").read_text(encoding="utf-8"))


def _derived_seed(base: int, label: str) -> int:
    digest = hashlib.sha256(label.encode("utf-8")).hexdigest()
    return (base + int(digest[:8], 16)) % (2**32)


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def analyze(
    manifest: Path,
    artifact_root: Path,
    *,
    reference_algorithm: str,
    reference_variant: str | None,
    metrics: tuple[str, ...],
    bootstrap_samples: int,
    bootstrap_seed: int,
    alpha: float,
) -> dict:
    report = validate_dataset(manifest, artifact_root)
    if not report.passes("analysis"):
        raise RuntimeError("dataset does not pass the analysis validation gate")
    directions = {metric: metric_direction(metric) for metric in metrics}
    specs = sorted(
        expand_manifest(load_manifest(manifest)), key=lambda spec: spec.run_id
    )
    observations: dict[tuple[str, str, int], dict[str, float]] = {}
    digest_records = []
    algorithm_ids: dict[tuple[str, str], str] = {}
    for spec in specs:
        envelope = _read_metadata(artifact_path(artifact_root, spec))
        run_metadata = envelope["run_metadata"]
        values = {
            **run_metadata["metrics"],
            **run_metadata["score_diagnostics"],
        }
        label = f"{spec.algorithm_id}:{spec.variant}"
        algorithm_ids[(spec.problem_id, label)] = spec.algorithm_id
        observations[(spec.problem_id, label, spec.seed)] = {
            metric: float(values[metric]) for metric in metrics
        }
        digest_records.append(
            (
                spec.run_id,
                spec.problem_id,
                label,
                spec.seed,
                observations[(spec.problem_id, label, spec.seed)],
            )
        )

    problems = sorted({key[0] for key in observations})
    summaries = []
    paired = []
    raw_families: dict[str, dict[str, float]] = {}
    for problem in problems:
        algorithms = sorted(
            {label for observed_problem, label, _ in observations if observed_problem == problem}
        )
        references = [
            label
            for label in algorithms
            if algorithm_ids[(problem, label)] == reference_algorithm
            and (
                reference_variant is None
                or label == f"{reference_algorithm}:{reference_variant}"
            )
        ]
        if len(references) != 1:
            raise ValueError(
                f"problem {problem} has {len(references)} variants for reference "
                f"algorithm {reference_algorithm}"
                + (
                    f" and variant {reference_variant}"
                    if reference_variant is not None
                    else ""
                )
                + "; exactly one is required"
            )
        reference = references[0]
        seeds = sorted(
            seed
            for observed_problem, label, seed in observations
            if observed_problem == problem and label == reference
        )
        for algorithm in algorithms:
            for metric in metrics:
                values = [
                    observations[(problem, algorithm, seed)][metric] for seed in seeds
                ]
                label = f"{problem}|{algorithm}|{metric}"
                summary = summarize(
                    values,
                    bootstrap_samples=bootstrap_samples,
                    seed=_derived_seed(bootstrap_seed, label),
                )
                summaries.append(
                    {
                        "problem_id": problem,
                        "algorithm": algorithm,
                        "metric": metric,
                        "direction": directions[metric],
                        **asdict(summary),
                    }
                )
        for competitor in (algorithm for algorithm in algorithms if algorithm != reference):
            for metric in metrics:
                reference_values = {
                    seed: observations[(problem, reference, seed)][metric]
                    for seed in seeds
                }
                competitor_values = {
                    seed: observations[(problem, competitor, seed)][metric]
                    for seed in seeds
                }
                wilcoxon_result = paired_wilcoxon(
                    reference_values,
                    competitor_values,
                    direction=directions[metric],
                    zero_method="wilcox",
                    failure_policy="raise",
                    alternative="two-sided",
                )
                effect = vargha_delaney_a12(
                    [reference_values[seed] for seed in seeds],
                    [competitor_values[seed] for seed in seeds],
                    direction=directions[metric],
                )
                paired_effect = matched_pairs_rank_biserial(
                    [reference_values[seed] for seed in seeds],
                    [competitor_values[seed] for seed in seeds],
                    direction=directions[metric],
                )
                family = f"final-{metric}"
                hypothesis = f"{problem}|{reference}-vs-{competitor}"
                raw_families.setdefault(family, {})[hypothesis] = wilcoxon_result.p_value
                paired.append(
                    {
                        "problem_id": problem,
                        "reference": reference,
                        "competitor": competitor,
                        "metric": metric,
                        "direction": directions[metric],
                        "family": family,
                        "hypothesis": hypothesis,
                        "p_raw": wilcoxon_result.p_value,
                        "wilcoxon": asdict(wilcoxon_result),
                        "effect_size": asdict(effect),
                        "paired_rank_biserial": paired_effect,
                    }
                )

    corrections = {
        family: holm_step_down(values, family=family, alpha=alpha)
        for family, values in raw_families.items()
    }
    for comparison in paired:
        correction = corrections[comparison["family"]][comparison["hypothesis"]]
        comparison["p_adjusted"] = correction.p_adjusted
        comparison["reject_holm"] = correction.reject

    friedman = []
    for problem in problems:
        algorithms = sorted(
            {label for observed_problem, label, _ in observations if observed_problem == problem}
        )
        seeds = sorted(
            {seed for observed_problem, _, seed in observations if observed_problem == problem}
        )
        for metric in metrics:
            if len(algorithms) < 3:
                friedman.append(
                    {
                        "problem_id": problem,
                        "metric": metric,
                        "direction": directions[metric],
                        "applicable": False,
                        "reason": "fewer than three algorithms",
                    }
                )
                continue
            blocks = {
                str(seed): {
                    algorithm: observations[(problem, algorithm, seed)][metric]
                    for algorithm in algorithms
                }
                for seed in seeds
            }
            friedman.append(
                {
                    "problem_id": problem,
                    "metric": metric,
                    "applicable": True,
                    **asdict(
                        friedman_complete_blocks(
                            blocks, direction=directions[metric]
                        )
                    ),
                }
            )
    return {
        "schema_version": 1,
        "manifest_id": report.manifest_id,
        "artifact_root": str(artifact_root),
        "validation_gate": "analysis",
        "validation_passed": True,
        "observations_digest": observations_digest(digest_records),
        "family_sizes": {family: len(values) for family, values in raw_families.items()},
        "reference_algorithm": reference_algorithm,
        "reference_variant": reference_variant,
        "metric_directions": directions,
        "bootstrap": {
            "confidence": 0.95,
            "samples": bootstrap_samples,
            "base_seed": bootstrap_seed,
        },
        "alpha": alpha,
        "summaries": summaries,
        "paired_tests": sorted(
            paired, key=lambda item: (item["family"], item["hypothesis"])
        ),
        "friedman_tests": friedman,
    }


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.bootstrap_samples < 1:
        parser.error("--bootstrap-samples must be positive")
    if args.bootstrap_seed < 0:
        parser.error("--bootstrap-seed must be non-negative")
    try:
        document = analyze(
            args.manifest,
            args.artifact_root,
            reference_algorithm=args.reference_algorithm,
            reference_variant=args.reference_variant,
            metrics=tuple(args.metrics),
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
            alpha=args.alpha,
        )
    except RuntimeError as exc:
        print(f"statistics refused: validation gate failure: {exc}", file=sys.stderr)
        return 1
    except (FileNotFoundError, KeyError, OSError, TypeError, ValueError) as exc:
        print(f"statistics error: {exc}", file=sys.stderr)
        return 2
    text = json.dumps(
        document,
        allow_nan=False,
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
    ) + "\n"
    if args.output:
        _write_atomic(args.output, text)
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
