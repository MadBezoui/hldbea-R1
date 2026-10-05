#!/usr/bin/env python3
"""Generate publication figures only from a validation-gated dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

from hldbea.artifacts import artifact_path
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
    rotation_experiment_descriptor,
    select_median_run,
)
from hldbea.run_spec import expand_manifest, load_manifest
from hldbea.validation import validate_dataset


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate validated figures.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--problem-id")
    parser.add_argument("--stem", default="experiment")
    parser.add_argument("--format", choices=("png", "pdf", "svg"), default="png")
    return parser


def _artifact_data(root: Path, spec) -> tuple[dict, list[dict], np.ndarray]:
    path = artifact_path(root, spec)
    envelope = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    checkpoints = json.loads((path / "checkpoints.json").read_text(encoding="utf-8"))
    with np.load(path / "arrays.npz", allow_pickle=False) as archive:
        objectives = archive["F"].copy()
    return envelope["run_metadata"], checkpoints, objectives


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        validation = validate_dataset(args.manifest, args.artifact_root)
        if not validation.passes("analysis"):
            print("figures refused: dataset failed the analysis gate", file=sys.stderr)
            return 1
        specs = sorted(expand_manifest(load_manifest(args.manifest)), key=lambda item: item.run_id)
        problems = {}
        for spec in specs:
            problems.setdefault(spec.problem_id, spec.n_obj)
        if args.problem_id:
            if args.problem_id not in problems:
                raise ValueError(f"unknown problem id: {args.problem_id}")
            selected_problem = args.problem_id
        else:
            selected_problem = sorted(problems, key=lambda key: (-problems[key], key))[0]
        selected_specs = [spec for spec in specs if spec.problem_id == selected_problem]
        args.output_dir.mkdir(parents=True, exist_ok=True)
        final_records = []
        convergence_records = []
        candidates: dict[str, list[dict]] = {}
        matrices: dict[str, np.ndarray] = {}
        spec_by_run = {}
        for spec in selected_specs:
            metadata, checkpoints, objectives = _artifact_data(args.artifact_root, spec)
            algorithm = f"{spec.algorithm_id}:{spec.variant}"
            final_records.append(
                {
                    "algorithm": algorithm,
                    "run_id": spec.run_id,
                    **metadata["metrics"],
                }
            )
            for checkpoint in checkpoints:
                convergence_records.append(
                    {
                        "algorithm": algorithm,
                        "seed": spec.seed,
                        **checkpoint,
                    }
                )
            candidates.setdefault(algorithm, []).append(
                {
                    "run_id": spec.run_id,
                    "hv": metadata["metrics"]["hv"],
                }
            )
            matrices[spec.run_id] = objectives
            spec_by_run[spec.run_id] = spec

        outputs = []
        for metric in ("hv", "igd_plus"):
            outputs.append(
                plot_convergence(
                    convergence_records,
                    args.output_dir
                    / figure_filename(
                        "convergence", selected_problem, metric, extension=args.format
                    ),
                    metric=metric,
                )
            )
            outputs.append(
                plot_box_violin(
                    final_records,
                    args.output_dir
                    / figure_filename(
                        "box-violin", selected_problem, metric, extension=args.format
                    ),
                    metric=metric,
                )
            )

        median_runs = {}
        for algorithm in sorted(candidates):
            selected = select_median_run(candidates[algorithm])
            run_id = selected["run_id"]
            median_runs[algorithm] = run_id
            objectives = matrices[run_id]
            label = algorithm.replace(":", "-")
            outputs.extend(
                [
                    plot_parallel_coordinates(
                        objectives,
                        args.output_dir
                        / figure_filename(
                            "parallel-coordinates",
                            selected_problem,
                            label,
                            extension=args.format,
                        ),
                    ),
                    plot_radviz(
                        objectives,
                        args.output_dir
                        / figure_filename(
                            "radviz", selected_problem, label, extension=args.format
                        ),
                    ),
                    plot_pca_projection(
                        objectives,
                        args.output_dir
                        / figure_filename(
                            "pca", selected_problem, label, extension=args.format
                        ),
                    ),
                    plot_normalized_heatmap(
                        objectives,
                        args.output_dir
                        / figure_filename(
                            "heatmap", selected_problem, label, extension=args.format
                        ),
                    ),
                ]
            )

        rotation_records = []
        for spec in specs:
            descriptor = rotation_experiment_descriptor(spec)
            if descriptor is None:
                continue
            family, angle = descriptor
            metadata, _, _ = _artifact_data(args.artifact_root, spec)
            rotation_records.append(
                {
                    "problem_family": family,
                    "problem_id": spec.problem_id,
                    "algorithm": f"{spec.algorithm_id}:{spec.variant}",
                    "angle": float(angle),
                    **metadata["metrics"],
                }
            )
        rotation_families = {}
        for family, family_records in group_rotation_records(rotation_records).items():
            angles = sorted({record["angle"] for record in family_records})
            rotation_families[family] = {
                "angles": angles,
                "problem_ids": sorted({record["problem_id"] for record in family_records}),
            }
            if len(angles) < 2:
                continue
            for metric in ("hv", "igd_plus"):
                outputs.append(
                    plot_rotation_curve(
                        family_records,
                        args.output_dir
                        / figure_filename(
                            "rotation", family, metric, extension=args.format
                        ),
                        metric=metric,
                    )
                )

        provenance = {
            "schema_version": 1,
            "kind": "figures",
            "manifest_id": validation.manifest_id,
            "artifact_root": str(args.artifact_root),
            "validation_gates": validation.gates,
            "selected_problem": selected_problem,
            "selected_problem_objectives": problems[selected_problem],
            "median_run_policy": "ascending HV rank, lower median, run-id tie break",
            "median_run_ids": median_runs,
            "rotation_families": rotation_families,
            "source_run_ids": sorted(spec.run_id for spec in specs),
            "outputs": {path.name: _sha256(path) for path in outputs},
        }
        provenance_path = args.output_dir / f"{args.stem}-figures-provenance.json"
        provenance_path.write_text(
            json.dumps(provenance, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        for path in (*outputs, provenance_path):
            print(path)
        return 0
    except (FileNotFoundError, KeyError, OSError, TypeError, ValueError) as exc:
        print(f"figures error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
