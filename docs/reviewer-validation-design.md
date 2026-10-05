# Reviewer-driven validation design

This design maps every new run to an explicit major-revision request. It is
separate from the three-seed calibration screens and uses the reserved seeds
`51001` through `51030`. Runs are immutable and resumable; every algorithm in a
problem block receives the same population size, checkpoints, and total
objective-evaluation budget.

| Manifest | Runs | Budget/run | Direct reviewer coverage |
|---|---:|---:|---|
| `reviewer-dtlz3-validation-v1` | 210 | 46,000 FE | R1.2, R1.5 |
| `reviewer-modern-benchmarks-validation-v1` | 900 | 20,000 FE | R1.1, R3.1, R3.2, R4.4 |
| `reviewer-geometry-ablation-validation-v1` | 720 | 20,000 FE | R1.4, R1.6, R2.4 |
| `reviewer-manyobjective-validation-v1` | 960 | 10,000 FE | R1.3, R2.2, R3.3, R4.5 |
| `reviewer-rotation-validation-v1` | 360 | 10,000 FE | R2.3, R4.6 |

The complete design contains 3,150 runs and 55.26 million charged objective
evaluations. It is executed sequentially with one worker and one numerical
thread. No result from a validation seed is used to change a parameter.

## DTLZ3 mechanism block

The six HLDBEA configurations distinguish the core local score, augmented
fitness, two-iteration deterministic refinement, and restart. NSGA-III is a
reference-point control. Checkpoint trajectories and the recorded local-search
and restart events permit statements about failure rate, solver efficiency, and
whether SLSQP worsens entrapment; final HV alone is not used to infer mechanism.

## Modern benchmark/comparator block

IMOP3, IMOP4, and IMOP7 provide disconnected, irregular/degenerate, and
multi-region fronts from a benchmark suite published in 2019. The common block
contains HLDBEA, NSGA-II, SPEA2, NSGA-III, RVEA, MOEA/D, SMS-EMOA, AGE-MOEA,
MaOEA-HAP (2026), and FDSEA (2026). Provenance and source hashes are recorded in
`docs/imop-benchmark-provenance.md` and
`docs/post-2024-comparator-provenance.md`.

NSMA is not silently omitted. `docs/nsma-exclusion-audit.md` records why its
gradient/time-based/best-of-five protocol is not mixed into the equal-FE table.

## Geometry and neighbourhood block

DTLZ2, DTLZ7, WFG3, and WFG9 cover smooth, disconnected, degenerate, and
deceptive/nonseparable cases. The comparisons isolate:

- local versus global candidate selection;
- round-robin versus adaptive objective selection;
- directional axis bands with sum versus union aggregation;
- a lower orthotope (`box-union`);
- normalized objective-space k-nearest neighbours (`knn-union`).

## Many-objective block

DTLZ2 and WFG2 are run at 10 and 15 objectives. Four HLDBEA cone rules are
predeclared: strict, fixed 0.1, linear, and saturating. NSGA-III, RVEA,
MaOEA-HAP, and FDSEA are the many-objective controls. HV above five objectives
uses a fixed 65,536-point scrambled Sobol design shared by every run; IGD+ uses
the same deterministic problem reference geometry.

## Rotation block

`rdtlz2` rotates only the positive-unit-front direction and then adds the
standard DTLZ2 distance term equally to all objectives:

`F_theta(x) = R_theta h(x) + offset_theta + g(x) 1`.

For every off-front decision (`g > 0`), the decision with identical position
variables and `g = 0` dominates it. This avoids the invalid assumption that an
arbitrary rotation of the full objective vector preserves Pareto dominance.
Angles 0, 15, 30, and 45 degrees are compared for HLDBEA, NSGA-II, and
NSGA-III.

## Statistical contract

Final metrics are summarized by means, standard deviations, medians, and
deterministic 95% bootstrap intervals. Seed-paired comparisons use the
two-sided Wilcoxon signed-rank test, with Holm correction separately over the
HV and IGD+ hypothesis families, plus direction-normalized Vargha--Delaney
effect sizes. Friedman complete-block tests summarize multi-algorithm
comparisons. Negative or null outcomes remain in the reports.
