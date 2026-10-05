"""Replay DTLZ3 runs with dense checkpoints to compare them per generation.

Under an equal evaluation budget the deterministic track also reduces the
number of generations of the evolutionary track. Replaying the same runs with
a checkpoint every few generations shows how the configurations compare at an
equal number of generations. Checkpoints only read the population, so the
final arrays must match the archived runs bit for bit, and the script stops
otherwise.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import sys
import tempfile

from hldbea.artifacts import arrays_digest, artifact_path
from hldbea.run_spec import expand_manifest, load_manifest
from hldbea.runner import execute_run


def _replay(task):
    spec, dense, artifact_root = task
    archived = artifact_path(Path(artifact_root), spec)
    with tempfile.TemporaryDirectory() as scratch:
        with redirect_stdout(io.StringIO()):
            outcome = execute_run(replace(spec, checkpoints=dense), scratch)
        if outcome.status == "failed":
            return {"run_id": spec.run_id, "error": outcome.error_message}
        replayed = Path(outcome.artifact_path)
        records = json.loads((replayed / "checkpoints.json").read_text())
        identical = arrays_digest(replayed) == arrays_digest(archived)
    return {
        "run_id": spec.run_id,
        "variant": spec.variant,
        "seed": spec.seed,
        "arrays_sha256": arrays_digest(archived),
        "identical": identical,
        "trajectory": [
            [r["evaluation"], r["generation"], r["hv"], r["igd_plus"]] for r in records
        ],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, default=Path("results/raw"))
    parser.add_argument("--variants", nargs="+", required=True)
    parser.add_argument("--step", type=int, default=460)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    specs = [
        s for s in expand_manifest(load_manifest(args.manifest))
        if s.algorithm_id == "hldbea" and s.variant in args.variants
    ]
    budget = specs[0].evaluation_budget
    dense = tuple(range(args.step, budget + 1, args.step))
    if dense[-1] != budget:
        dense += (budget,)
    tasks = [(spec, dense, str(args.artifact_root)) for spec in specs]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(_replay, tasks, chunksize=2))
    report = {
        "manifest_id": args.manifest.stem,
        "step": args.step,
        "variants": sorted(args.variants),
        "runs": sorted(results, key=lambda r: r["run_id"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True))
    failed = [r for r in results if "error" in r or not r["identical"]]
    if failed:
        print(f"{len(failed)} replays differ from the archive", file=sys.stderr)
        return 1
    print(f"{len(results)}/{len(results)} replays identical")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
