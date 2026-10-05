# HLDBEA core-correctness evidence report

Date: 2026-10-03  
Branch: `codex/reviewer-evidence-rebuild`  
Scope: implementation correctness and smoke readiness only

## Outcome

The legacy HLDBEA execution path now delegates its core decisions to tested,
explicit primitives. Raw local score, distance augmentation, global Pareto
status, and false-zero diagnostics are stored separately. Survivor selection
uses the raw zero-score stratum; its global rank-and-crowding alternative is a
real ablation. Restart detection uses raw scores and an explicit metric
reference point. Evolutionary and SLSQP calls share one function-evaluation
ledger. The exact-infill objective policy is declared and reproducible.

No comparative performance conclusion follows from this work. The manifest in
`experiments/manifests/core-smoke.yaml` is a correctness matrix, not a results
table.

## Defect-to-evidence map

| Repaired risk | Implementation evidence | Regression evidence |
|---|---|---|
| GUI/Qt import blocked headless runs | `hldbea.runtime.configure_headless_matplotlib` | `tests/test_runtime.py` and CLI smoke |
| Local score discarded globally dominated neighbours | `hldbea.scoring.compute_local_scores` | `tests/test_scoring.py` |
| Non-strict excluded-axis comparison changed the score definition | strict `<` plus explicit `cone_epsilon` | `tests/test_scoring.py` |
| Fitness augmentation was mistaken for raw score | separate `ScoreRaw`, `ScoreAxis`, `FitDistance`, `Fit` | `tests/test_legacy_adapter.py` |
| Survival partitioned on maximal augmented fitness | `hldbea.selection.select_survivor_indices` | `tests/test_selection.py`, `tests/test_legacy_adapter.py` |
| Restart gated on `Fit >= 0` and guessed a nadir point | `hldbea.restart` plus structured restart events | `tests/test_restart.py`, `tests/test_legacy_adapter.py` |
| SLSQP calls were absent from FE totals | `EvaluationLedger` and `LedgerEvaluator` | `tests/test_evaluation.py`, `tests/test_local_search.py` |
| Exact infill chose a hidden random objective | round-robin/adaptive/random policy interface | `tests/test_local_search.py`, `tests/test_legacy_adapter.py` |
| Rejected/infeasible local-search points were not auditable | `LocalSearchResult` and per-attempt event log | `tests/test_local_search.py` |
| Objective rotations/PCA were not reproducible | seeded orthonormal transforms and PCA coordinates | `tests/test_transforms.py` |
| More-than-three-objective CLI used an undefined `ref_dirs` | deterministic reference directions and metric front | `tests/test_legacy_adapter.py`, DTLZ2 five-objective smoke |

## Verified environment

The project-local virtual environment used Python 3.10.1, NumPy 1.26.4,
SciPy 1.14.1, pandas 2.3.3, pymoo 0.6.1.3, Matplotlib 3.10.7,
PyYAML 6.0.2, and pytest 8.4.1. SciPy is pinned because the previously
available 1.15.3 wheel on this host could not load `_spropack`.

## Verification commands

```text
MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q
MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1 .venv/bin/python nibea_test.py --help
MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1 .venv/bin/python nibea_test.py --test-single --pb DTLZ2 --n_var 14 --n_obj 5 --n_gen 2 --n_pop 10
git diff --check
```

Latest clean result: 102 tests passed in 24.85 s; `git diff --check` returned
zero; the five-objective CLI smoke returned zero and emitted finite GD, IGD,
and IGD+ values (HV was zero for this deliberately tiny two-generation run).

The full suite must be rerun after any change to scoring, evaluation accounting,
selection, restart, metric reference geometry, or legacy adapters.

## Known scope limits

- The matrix has not produced confirmatory benchmark results and supports no
  superiority, robustness, scalability, or statistical-significance claim.
- The five-objective DTLZ2 run is a two-generation integration smoke only.
- AGE-MOEA remains an optional baseline whose pymoo implementation requires
  `numba`; it is imported lazily so unrelated HLDBEA runs do not fail.
- This phase does not yet provide resumable experiment orchestration,
  multiprocessing, raw-result schemas, statistical tests, publication figures,
  or manuscript tables.
- Analytical/reference fronts are deterministic, but the confirmatory study
  must freeze every problem-specific HV reference point and reference set in
  its own manifest before execution.

## Prerequisites for the experiment-harness plan

1. Freeze benchmark instances, algorithms, versions, seeds, population sizes,
   and a single shared FE budget per instance.
2. Implement atomic per-run artifacts containing configuration, environment,
   git revision, evolutionary FE, solver FE, runtime, final decisions, final
   objectives, and all restart/local-search events.
3. Add resume/skip validation and bounded worker concurrency.
4. Validate every baseline independently, including optional dependencies and
   many-objective population/reference-direction compatibility.
5. Predeclare HV/IGD+/epsilon metrics, failure handling, Wilcoxon pairing,
   Holm correction, effect sizes, and aggregation rules.
6. Generate tables and figures only from validated raw artifacts, then revise
   the manuscript so every numerical claim has a traceable source.
