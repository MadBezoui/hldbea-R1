#!/usr/bin/env python3
"""Run a strict experiment manifest with bounded, resumable parallelism."""

from __future__ import annotations

import argparse
from concurrent.futures import CancelledError, ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
import io
import json
import multiprocessing
import os
from pathlib import Path
import sys
import uuid

from hldbea.artifacts import artifact_path, artifact_status
from hldbea.design_audit import estimate_run_bytes
from hldbea.run_spec import RunSpec, expand_manifest, load_manifest
from hldbea.runner import RunOutcome, execute_run


SAFE_WORKER_CAP = 4


def _default_workers() -> int:
    return min(SAFE_WORKER_CAP, os.cpu_count() or 1)


def _json_line(prefix: str, document: dict) -> None:
    print(
        prefix
        + json.dumps(
            document,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ),
        flush=True,
    )


def _worker(spec: RunSpec, artifact_root: str) -> RunOutcome:
    # Legacy HLDBEA constructors print configuration banners. Keep the CLI's
    # stdout as deterministic JSON lines; all scientific events live in the
    # immutable artifact rather than in scheduler-dependent process output.
    with redirect_stdout(io.StringIO()):
        return execute_run(spec, artifact_root)


def _archive_invalid(artifact_root: Path, spec: RunSpec) -> Path:
    source = artifact_path(artifact_root, spec)
    if not source.is_dir():
        raise FileNotFoundError(f"invalid artifact disappeared before quarantine: {source}")
    parent = artifact_root / spec.manifest_id / "invalid" / spec.run_id
    parent.mkdir(parents=True, exist_ok=True)
    destination = parent / f"attempt-{uuid.uuid4().hex}"
    os.replace(source, destination)
    return destination


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Execute an immutable HLDBEA experiment manifest."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=_default_workers())
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--only-run-id")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--allow-oversubscription",
        action="store_true",
        help=f"allow more than the conservative cap of {SAFE_WORKER_CAP} workers",
    )
    return parser


def _inventory(specs: list[RunSpec], artifact_root: Path) -> dict[str, int]:
    counts = {"valid": 0, "invalid": 0, "failed": 0, "missing": 0}
    for spec in specs:
        counts[artifact_status(artifact_root, spec)] += 1
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.workers > SAFE_WORKER_CAP and not args.allow_oversubscription:
        parser.error(
            f"--workers above {SAFE_WORKER_CAP} require --allow-oversubscription"
        )

    try:
        specs = sorted(
            expand_manifest(load_manifest(args.manifest)),
            key=lambda item: item.run_id,
        )
    except (FileNotFoundError, OSError, ValueError) as exc:
        parser.error(str(exc))
    if args.only_run_id is not None:
        specs = [spec for spec in specs if spec.run_id == args.only_run_id]
        if not specs:
            parser.error(f"unknown --only-run-id: {args.only_run_id}")

    statuses = {spec.run_id: artifact_status(args.artifact_root, spec) for spec in specs}
    inventory = _inventory(specs, args.artifact_root)
    plan = {
        "estimated_max_bytes": sum(estimate_run_bytes(spec) for spec in specs),
        "failed": inventory["failed"],
        "invalid": inventory["invalid"],
        "missing": inventory["missing"],
        "planned": len(specs),
        "run_ids": [spec.run_id for spec in specs],
        "valid": inventory["valid"],
        "workers": args.workers,
    }
    _json_line("PLAN ", plan)
    if args.dry_run:
        return 0

    occupied = [
        spec.run_id
        for spec in specs
        if statuses[spec.run_id] in {"valid", "invalid"}
    ]
    if occupied and not args.resume:
        print(
            "ERROR existing complete or invalid artifacts require --resume: "
            + ", ".join(occupied),
            file=sys.stderr,
        )
        return 2

    skipped = [spec for spec in specs if statuses[spec.run_id] == "valid"]
    scheduled = [spec for spec in specs if statuses[spec.run_id] != "valid"]
    for spec in scheduled:
        if statuses[spec.run_id] == "invalid":
            _archive_invalid(args.artifact_root, spec)

    outcomes: list[RunOutcome] = []
    worker_errors: list[dict[str, str]] = []
    if scheduled:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=context,
        ) as executor:
            future_to_spec = {
                executor.submit(_worker, spec, str(args.artifact_root)): spec
                for spec in scheduled
            }
            failed = False
            for future in as_completed(future_to_spec):
                spec = future_to_spec[future]
                try:
                    outcome = future.result()
                except CancelledError:
                    continue
                except Exception as exc:  # defensive guard for worker-process crashes
                    worker_errors.append(
                        {
                            "run_id": spec.run_id,
                            "status": "failed",
                            "error_type": type(exc).__name__,
                            "error_message": str(exc),
                        }
                    )
                    failed = True
                else:
                    outcomes.append(outcome)
                    failed = failed or outcome.status == "failed"
                if failed and args.fail_fast:
                    for pending in future_to_spec:
                        pending.cancel()

    records = [
        {
            "run_id": outcome.run_id,
            "status": outcome.status,
            **(
                {
                    "error_type": outcome.error_type,
                    "error_message": outcome.error_message,
                }
                if outcome.status == "failed"
                else {}
            ),
        }
        for outcome in outcomes
    ] + worker_errors
    for record in sorted(records, key=lambda item: item["run_id"]):
        _json_line("RUN ", record)

    failures = sum(record["status"] == "failed" for record in records)
    _json_line(
        "SUMMARY ",
        {
            "failed": failures,
            "scheduled": len(scheduled),
            "skipped_valid": len(skipped),
            "succeeded": len(records) - failures,
        },
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
