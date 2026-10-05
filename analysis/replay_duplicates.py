"""Replay the HLDBEA runs of a validated block to count duplicates.

The duplicate counters were added after the v3 campaigns. They only read the
population, so re-executing a run from its manifest specification must give
the archived decision and objective arrays bit for bit. The script stops on
the first difference, and the report keeps, for every run, a digest of the
archived arrays so that the manuscript generator can bind it to the archive.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile

from hldbea.artifacts import arrays_digest, artifact_path
from hldbea.run_spec import expand_manifest, load_manifest
from hldbea.runner import execute_run


def _replay(task):
    spec, artifact_root = task
    archived = artifact_path(Path(artifact_root), spec)
    with tempfile.TemporaryDirectory() as scratch:
        with redirect_stdout(io.StringIO()):
            outcome = execute_run(spec, scratch)
        if outcome.status == "failed":
            return {"run_id": spec.run_id, "error": outcome.error_message}
        replayed = Path(outcome.artifact_path)
        identical = arrays_digest(replayed) == arrays_digest(archived)
        metadata = json.loads((replayed / "metadata.json").read_text())["run_metadata"]
    return {
        "run_id": spec.run_id,
        "problem_id": spec.problem_id,
        "variant": spec.variant,
        "seed": spec.seed,
        "arrays_sha256": arrays_digest(archived),
        "identical": identical,
        "duplicate_summary": metadata["duplicate_summary"],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, default=Path("results/raw"))
    parser.add_argument("--algorithm", default="hldbea")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    manifest = load_manifest(args.manifest)
    specs = [s for s in expand_manifest(manifest) if s.algorithm_id == args.algorithm]
    tasks = [(spec, str(args.artifact_root)) for spec in specs]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(_replay, tasks, chunksize=4))
    failed = [r for r in results if "error" in r or not r["identical"]]
    report = {
        "manifest_id": args.manifest.stem,
        "algorithm": args.algorithm,
        "replayed": len(results),
        "identical": sum(bool(r.get("identical")) for r in results),
        "runs": sorted(results, key=lambda r: r["run_id"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True))
    if failed:
        print(f"{len(failed)} replays differ from the archive", file=sys.stderr)
        return 1
    print(f"{report['identical']}/{report['replayed']} replays identical")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
