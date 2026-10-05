#!/usr/bin/env python3
"""Generate comparison tables only from a validation-gated dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

from hldbea.reporting import write_comparison_tables
from hldbea.run_spec import expand_manifest, load_manifest
from hldbea.validation import validate_dataset


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate validated result tables.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--statistics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stem", default="experiment")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        validation = validate_dataset(args.manifest, args.artifact_root)
        if not validation.passes("analysis"):
            print("tables refused: dataset failed the analysis gate", file=sys.stderr)
            return 1
        statistics = json.loads(args.statistics.read_text(encoding="utf-8"))
        if (
            statistics.get("validation_passed") is not True
            or statistics.get("manifest_id") != validation.manifest_id
        ):
            raise ValueError("statistics provenance does not match the validated manifest")
        outputs = write_comparison_tables(
            statistics, args.output_dir, stem=args.stem
        )
        specs = expand_manifest(load_manifest(args.manifest))
        provenance = {
            "schema_version": 1,
            "kind": "tables",
            "manifest_id": validation.manifest_id,
            "artifact_root": str(args.artifact_root),
            "validation_gates": validation.gates,
            "statistics_file": str(args.statistics),
            "statistics_sha256": _sha256(args.statistics),
            "source_run_ids": sorted(spec.run_id for spec in specs),
            "outputs": {path.name: _sha256(path) for path in outputs},
        }
        provenance_path = args.output_dir / f"{args.stem}-tables-provenance.json"
        provenance_path.write_text(
            json.dumps(provenance, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        for path in (*outputs, provenance_path):
            print(path)
        return 0
    except (FileNotFoundError, KeyError, OSError, TypeError, ValueError) as exc:
        print(f"tables error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
