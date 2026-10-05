#!/usr/bin/env python3
"""Build the calibration shortlist or freeze the confirmed configuration."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
import uuid

import yaml

from hldbea.calibration import (
    build_shortlist,
    canonical_json_text,
    freeze_configuration,
    load_validated_records,
)
from hldbea.run_spec import expand_manifest, load_manifest


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("shortlist", "freeze"), required=True)
    parser.add_argument("--manifest", action="append", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path)
    parser.add_argument("--config-output", type=Path)
    parser.add_argument("--shortlist-report", type=Path)
    parser.add_argument("--top", type=int, default=5)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.mode == "shortlist":
            if args.manifest_output is None:
                raise ValueError("shortlist mode requires --manifest-output")
            report, manifest_text = build_shortlist(
                args.manifest, args.artifact_root, top=args.top
            )
            _write_atomic(args.manifest_output, manifest_text)
            _write_atomic(args.report_output, canonical_json_text(report))
        else:
            if len(args.manifest) != 1:
                raise ValueError("freeze mode requires exactly one manifest")
            if args.config_output is None or args.shortlist_report is None:
                raise ValueError(
                    "freeze mode requires --config-output and --shortlist-report"
                )
            records, provenance = load_validated_records(
                args.manifest[0], args.artifact_root
            )
            shortlist = json.loads(args.shortlist_report.read_text(encoding="utf-8"))
            candidate_parameters = {
                item["variant"]: item["parameters"]
                for item in shortlist["candidates"]
            }
            frozen = freeze_configuration(records, candidate_parameters)
            config = {
                "schema_version": 1,
                "config_id": "hldbea-production-v1",
                "algorithm": {
                    "id": "hldbea",
                    "variant": "production",
                    "parameters": frozen.parameters,
                },
                "selection": {
                    "rule": frozen.selection_rule_version,
                    "confirmation_variant": frozen.variant,
                    "rank": asdict(frozen.rank),
                },
            }
            report = {
                "schema_version": 1,
                "mode": "freeze",
                "frozen": asdict(frozen),
                "source": provenance,
            }
            _write_atomic(
                args.config_output,
                yaml.safe_dump(config, sort_keys=False, default_flow_style=False),
            )
            _write_atomic(args.report_output, canonical_json_text(report))
    except (FileNotFoundError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"calibration error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
