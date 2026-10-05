"""Auditable execution of one immutable experiment run."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib import metadata as importlib_metadata
import os
from pathlib import Path
import platform
import resource
import socket
import hashlib
import subprocess
import sys
import time
from typing import Any

import numpy as np
from pymoo.core.callback import Callback
from pymoo.optimize import minimize

from .artifacts import write_failure_artifact, write_run_artifact
from .evaluation import BudgetExhausted, EvaluationLedger
from .metrics import (
    ReferenceGeometry,
    build_reference_geometry,
    compute_metrics,
    compute_score_diagnostics,
)
from .problems import ProblemSpec, build_problem
from .registry import build_algorithm
from .run_spec import RunSpec


@dataclass(frozen=True)
class RunOutcome:
    run_id: str
    status: str
    artifact_path: Path
    metrics: dict[str, float] | None = None
    error_type: str | None = None
    error_message: str | None = None


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


class MetricCheckpointCallback(Callback):
    def __init__(
        self,
        ledger: EvaluationLedger,
        geometry: ReferenceGeometry,
        targets: tuple[int, ...],
        *,
        k: float,
        cone_epsilon: float,
        neighborhood_mode: str,
        score_aggregation: str,
    ) -> None:
        super().__init__()
        self.ledger = ledger
        self.geometry = geometry
        self.targets = targets
        self.k = k
        self.cone_epsilon = cone_epsilon
        self.neighborhood_mode = neighborhood_mode
        self.score_aggregation = score_aggregation
        self.records: list[dict[str, Any]] = []
        self._next = 0

    def notify(self, algorithm):
        used = self.ledger.used
        if algorithm.pop is None or self._next >= len(self.targets):
            return
        pending = []
        while self._next < len(self.targets) and self.targets[self._next] <= used:
            pending.append(self.targets[self._next])
            self._next += 1
        if not pending:
            return
        objectives = np.asarray(algorithm.pop.get("F"), dtype=float)
        metrics = compute_metrics(objectives, self.geometry)
        diagnostics = compute_score_diagnostics(
            objectives,
            k=self.k,
            cone_epsilon=self.cone_epsilon,
            neighborhood_mode=self.neighborhood_mode,
            score_aggregation=self.score_aggregation,
        )
        for target in pending:
            self.records.append(
                {
                    "target_evaluation": target,
                    "evaluation": used,
                    "generation": int(getattr(algorithm, "n_gen", 0)),
                    "population_size": len(objectives),
                    **metrics,
                    **diagnostics,
                }
            )


def _source_digest(source_root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(source_root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        digest.update(str(path.relative_to(source_root)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _git_metadata() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        # Untracked outputs also make a tree dirty. Record whether tracked
        # sources differ from the commit and a digest of the executed code.
        source_diff = subprocess.run(
            ["git", "diff", "HEAD", "--", "src", "experiments", "analysis"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        return {
            "commit": commit,
            "dirty": dirty,
            "tracked_sources_modified": bool(source_diff.strip()),
            "source_diff_sha256": hashlib.sha256(source_diff.encode("utf-8")).hexdigest(),
            "source_sha256": _source_digest(root / "src"),
        }
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"commit": "unavailable", "dirty": None, "error": str(exc)}


def _dependency_versions() -> dict[str, str]:
    packages = {
        "numpy": "numpy",
        "scipy": "scipy",
        "pandas": "pandas",
        "pymoo": "pymoo",
        "matplotlib": "matplotlib",
        "pyyaml": "PyYAML",
        "numba": "numba",
        "llvmlite": "llvmlite",
    }
    versions = {"python": platform.python_version()}
    for key, distribution in packages.items():
        try:
            versions[key] = importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError:
            versions[key] = "not-installed"
    return versions


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _solver_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    displacements = []
    for event in events:
        before = np.asarray(event.get("f_before", []), dtype=float)
        after = np.asarray(event.get("f_after", []), dtype=float)
        if before.shape == after.shape and before.size:
            displacements.append(float(np.linalg.norm(after - before)))
    charged = sum(int(event.get("evaluations", 0)) for event in events)
    accepted_hv_gain = float(
        sum(float(event.get("hv_gain", 0.0)) for event in events)
    )
    return {
        "calls": len(events),
        "successful": sum(bool(event.get("success")) for event in events),
        "feasible": sum(bool(event.get("feasible")) for event in events),
        "accepted": sum(bool(event.get("accepted")) for event in events),
        "charged_evaluations": charged,
        "accepted_hv_gain": accepted_hv_gain,
        "gain_per_evaluation": accepted_hv_gain / charged if charged else 0.0,
        "displacements": displacements,
    }


def execute_run(spec: RunSpec, artifact_root: str | Path) -> RunOutcome:
    if not isinstance(spec, RunSpec):
        raise TypeError("spec must be a RunSpec")
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    git = _git_metadata()
    environment = _dependency_versions()
    worker = {
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "platform": platform.platform(),
        "machine": platform.machine(),
    }

    try:
        problem_spec = ProblemSpec.from_run_spec(spec)
        problem = build_problem(problem_spec)
        geometry = build_reference_geometry(problem_spec)
        ledger = EvaluationLedger(spec.evaluation_budget)
        algorithm = build_algorithm(spec, problem, ledger)
        k = float(spec.algorithm_parameters.get("k", 1.0))
        declared_cone_schedule = spec.algorithm_parameters.get("cone_epsilon", 0.0)
        cone_epsilon = float(
            getattr(algorithm, "_resolved_cone_epsilon", declared_cone_schedule)
        )
        neighborhood_mode = spec.algorithm_parameters.get(
            "neighborhood_mode", "axis"
        )
        score_aggregation = spec.algorithm_parameters.get(
            "score_aggregation", "sum"
        )
        callback = MetricCheckpointCallback(
            ledger,
            geometry,
            spec.checkpoints,
            k=k,
            cone_epsilon=cone_epsilon,
            neighborhood_mode=neighborhood_mode,
            score_aggregation=score_aggregation,
        )
        boundary_message = None
        try:
            minimize(
                problem,
                algorithm,
                ("n_eval", spec.evaluation_budget),
                seed=spec.seed,
                callback=callback,
                verbose=False,
                copy_algorithm=False,
                save_history=spec.save_history,
            )
        except BudgetExhausted as exc:
            boundary_message = str(exc)

        if algorithm.pop is None or len(algorithm.pop) == 0:
            raise RuntimeError("algorithm terminated without a population")
        objectives = np.asarray(algorithm.pop.get("F"), dtype=float)
        decisions = np.asarray(algorithm.pop.get("X"), dtype=float)
        metrics = compute_metrics(objectives, geometry)
        diagnostics = compute_score_diagnostics(
            objectives,
            k=k,
            cone_epsilon=cone_epsilon,
            neighborhood_mode=neighborhood_mode,
            score_aggregation=score_aggregation,
        )
        local_events = _jsonable(getattr(algorithm, "local_search_events", []))
        restart_events = _jsonable(getattr(algorithm, "restart_events", []))
        status = (
            "budget_exhausted"
            if boundary_message is not None or ledger.remaining == 0
            else "completed"
        )
        runtime = {
            "wall_seconds": float(time.perf_counter() - wall_start),
            "cpu_seconds": float(time.process_time() - cpu_start),
            "peak_rss_bytes": _peak_rss_bytes(),
        }
        metadata = {
            "termination_status": status,
            "termination_message": boundary_message,
            "evaluations": {
                "total": ledger.used,
                "evolutionary": ledger.evolutionary,
                "solver": ledger.solver,
            },
            "runtime": runtime,
            "metrics": metrics,
            "score_diagnostics": diagnostics,
            "declared_cone_schedule": _jsonable(declared_cone_schedule),
            "resolved_cone_epsilon": cone_epsilon,
            "geometry": geometry.metadata(),
            "algorithm": asdict(algorithm._algorithm_descriptor),
            "solver_summary": _solver_summary(local_events),
            "restart_summary": {
                "checks": len(restart_events),
                "triggered": sum(
                    event.get("status") == "triggered" for event in restart_events
                ),
                "replacements": sum(
                    int(event.get("replacement_count", 0))
                    for event in restart_events
                ),
            },
            "population_size_final": len(objectives),
            "history_saved": bool(spec.save_history),
            "history_length": len(getattr(algorithm, "history", [])),
            "git": git,
            "environment": environment,
            "worker": worker,
        }
        arrays = {"X": decisions, "F": objectives}
        raw_scores = algorithm.pop.get("ScoreRaw")
        if raw_scores is not None:
            candidate = np.asarray(raw_scores)
            if candidate.shape == (len(objectives),) and np.issubdtype(
                candidate.dtype, np.number
            ):
                arrays["ScoreRaw"] = candidate
        path = write_run_artifact(
            artifact_root,
            spec,
            _jsonable(metadata),
            arrays,
            {"local_search": local_events, "restart": restart_events},
            _jsonable(callback.records),
        )
        return RunOutcome(
            run_id=spec.run_id,
            status=status,
            artifact_path=path,
            metrics=metrics,
        )
    except Exception as exc:
        failure = write_failure_artifact(
            artifact_root,
            spec,
            exc,
            metadata={
                "git": git,
                "environment": environment,
                "worker": worker,
                "wall_seconds": float(time.perf_counter() - wall_start),
                "cpu_seconds": float(time.process_time() - cpu_start),
            },
        )
        return RunOutcome(
            run_id=spec.run_id,
            status="failed",
            artifact_path=failure,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )
