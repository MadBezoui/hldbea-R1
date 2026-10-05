#!/usr/bin/env python3
"""Validate a manifest dataset and emit machine- and human-readable gates."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import uuid

from hldbea.validation import validate_dataset


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate experiment artifacts.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument(
        "--gate", choices=("artifacts", "analysis"), default="analysis"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        report = validate_dataset(args.manifest, args.artifact_root)
    except (FileNotFoundError, OSError, TypeError, ValueError) as exc:
        print(f"validation error: {exc}", file=sys.stderr)
        return 2
    json_text = json.dumps(
        report.to_dict(),
        allow_nan=False,
        ensure_ascii=True,
        indent=2,
        sort_keys=True,
    ) + "\n"
    markdown_text = report.to_markdown()
    if args.json_output:
        _write_atomic(args.json_output, json_text)
    else:
        print(json_text, end="")
    if args.markdown_output:
        _write_atomic(args.markdown_output, markdown_text)
    else:
        print(markdown_text, end="")
    print(
        f"validation gate {args.gate}: "
        f"{'PASS' if report.passes(args.gate) else 'FAIL'}",
        file=sys.stderr,
    )
    return 0 if report.passes(args.gate) else 1


if __name__ == "__main__":
    raise SystemExit(main())
