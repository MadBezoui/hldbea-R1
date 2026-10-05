"""Tested core components for HLDBEA."""

from .artifacts import (
    RunArtifact,
    ValidationResult,
    artifact_status,
    validate_run_artifact,
    write_failure_artifact,
    write_run_artifact,
)
from .evaluation import BudgetExhausted, EvaluationLedger, LedgerEvaluator
from .local_search import LocalSearchResult, select_objective, solve_epsilon_constraint
from .metrics import (
    ReferenceGeometry,
    build_reference_geometry,
    compute_hypervolume,
    compute_metrics,
    compute_score_diagnostics,
    hypervolume_estimator_metadata,
    metric_reference_bounds,
    metric_reference_directions,
    metric_reference_point,
    normalize_objectives,
    pareto_front_for_metrics,
)
from .problems import ObjectiveTransformProblem, ProblemSpec, build_problem
from .registry import (
    AlgorithmDescriptor,
    AlgorithmUnavailable,
    available_algorithms,
    build_algorithm,
)
from .reporting import (
    best_algorithms,
    comparison_sign,
    figure_filename,
    select_median_run,
    win_tie_loss,
    write_comparison_tables,
)
from .runtime import configure_headless_matplotlib
from .restart import RestartDecision, RestartPolicy, RestartState, update_restart_state
from .run_spec import RunSpec, expand_manifest, load_manifest
from .runner import MetricCheckpointCallback, RunOutcome, execute_run
from .scoring import ScoreResult, compute_local_scores, pareto_nondominated_mask
from .selection import rank_and_crowding_indices, select_survivor_indices
from .statistics import (
    A12Result,
    DescriptiveSummary,
    FriedmanResult,
    HolmResult,
    PairedWilcoxonResult,
    friedman_complete_blocks,
    holm_step_down,
    metric_direction,
    paired_wilcoxon,
    summarize,
    vargha_delaney_a12,
)
from .transforms import PCAResult, orthonormal_rotation, pca_coordinates, rotate_objectives
from .validation import DatasetReport, validate_dataset

__all__ = [
    "BudgetExhausted",
    "AlgorithmDescriptor",
    "AlgorithmUnavailable",
    "A12Result",
    "DescriptiveSummary",
    "EvaluationLedger",
    "DatasetReport",
    "LedgerEvaluator",
    "LocalSearchResult",
    "FriedmanResult",
    "HolmResult",
    "PCAResult",
    "ProblemSpec",
    "ObjectiveTransformProblem",
    "ReferenceGeometry",
    "RunArtifact",
    "RunOutcome",
    "ScoreResult",
    "RestartDecision",
    "RestartPolicy",
    "RestartState",
    "RunSpec",
    "ValidationResult",
    "MetricCheckpointCallback",
    "PairedWilcoxonResult",
    "artifact_status",
    "available_algorithms",
    "best_algorithms",
    "build_algorithm",
    "build_problem",
    "build_reference_geometry",
    "compute_local_scores",
    "compute_hypervolume",
    "compute_metrics",
    "compute_score_diagnostics",
    "comparison_sign",
    "configure_headless_matplotlib",
    "pareto_nondominated_mask",
    "metric_reference_directions",
    "metric_reference_bounds",
    "metric_reference_point",
    "hypervolume_estimator_metadata",
    "normalize_objectives",
    "pareto_front_for_metrics",
    "orthonormal_rotation",
    "pca_coordinates",
    "rank_and_crowding_indices",
    "select_survivor_indices",
    "select_median_run",
    "expand_manifest",
    "execute_run",
    "figure_filename",
    "friedman_complete_blocks",
    "holm_step_down",
    "load_manifest",
    "metric_direction",
    "paired_wilcoxon",
    "rotate_objectives",
    "select_objective",
    "solve_epsilon_constraint",
    "summarize",
    "update_restart_state",
    "validate_run_artifact",
    "validate_dataset",
    "vargha_delaney_a12",
    "win_tie_loss",
    "write_failure_artifact",
    "write_run_artifact",
    "write_comparison_tables",
]
