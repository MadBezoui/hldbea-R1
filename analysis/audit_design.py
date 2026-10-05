#!/usr/bin/env python3
"""Audit staged calibration manifests before dispatch."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import uuid

from hldbea.design_audit import audit_calibration_design


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = audit_calibration_design(args.manifest)
    _write_atomic(
        args.output,
        json.dumps(
            report.to_dict(),
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    if not report.passes:
        for violation in report.violations:
            print(f"design audit: {violation}", file=sys.stderr)
        return 1
    print(
        f"design audit PASS: {report.run_count} runs, "
        f"{report.total_evaluations} planned FE",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
