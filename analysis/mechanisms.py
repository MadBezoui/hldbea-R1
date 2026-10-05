#!/usr/bin/env python3
"""Generate validation-gated solver, restart, and failure summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from hldbea.artifacts import artifact_path
from hldbea.reporting import summarize_mechanism_records
from hldbea.run_spec import expand_manifest, load_manifest
from hldbea.validation import validate_dataset


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    return parser


def _markdown(manifest_id: str, rows: list[dict]) -> str:
    lines = [
        "# Mechanism and failure summary",
        "",
        f"Manifest: `{manifest_id}`. All rows passed the analysis validation gate.",
        "",
        "| Algorithm | n | HV>0 | Rate | Median HV | Median IGD+ | Solver calls | Solver FE | Accepted | Accepted HV gain | Restart triggers | Runs restarted | Replacements | Median wall s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {algorithm} | {n} | {nonzero_hv} | {nonzero_hv_rate:.3f} | "
            "{median_hv:.6g} | {median_igd_plus:.6g} | {solver_calls} | "
            "{solver_evaluations} | {solver_accepted} | "
            "{solver_accepted_gain:.6g} | {restart_triggers} | "
            "{runs_with_restart} | {restart_replacements} | "
            "{median_wall_seconds:.3f} |".format(**row)
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        validation = validate_dataset(args.manifest, args.artifact_root)
        if not validation.passes("analysis"):
            print("mechanism report refused: analysis gate failed", file=sys.stderr)
            return 1
        records = []
        for spec in expand_manifest(load_manifest(args.manifest)):
            envelope = json.loads(
                (artifact_path(args.artifact_root, spec) / "metadata.json").read_text(
                    encoding="utf-8"
                )
            )
            metadata = envelope["run_metadata"]
            solver = metadata["solver_summary"]
            restart = metadata["restart_summary"]
            records.append(
                {
                    "algorithm": f"{spec.algorithm_id}:{spec.variant}",
                    "hv": metadata["metrics"]["hv"],
                    "igd_plus": metadata["metrics"]["igd_plus"],
                    "solver_calls": solver["calls"],
                    "solver_evaluations": solver["charged_evaluations"],
                    "solver_accepted": solver["accepted"],
                    "solver_gain": solver["accepted_hv_gain"],
                    "restart_triggered": restart["triggered"],
                    "restart_replacements": restart["replacements"],
                    "wall_seconds": metadata["runtime"]["wall_seconds"],
                }
            )
        rows = summarize_mechanism_records(records)
        document = {
            "schema_version": 1,
            "manifest_id": validation.manifest_id,
            "validation_passed": True,
            "source_run_ids": validation.evidence["valid_run_ids"],
            "rows": rows,
        }
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(document, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        args.markdown_output.write_text(
            _markdown(validation.manifest_id, rows), encoding="utf-8"
        )
        print(args.json_output)
        print(args.markdown_output)
        return 0
    except (FileNotFoundError, KeyError, OSError, TypeError, ValueError) as exc:
        print(f"mechanism report error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
