# NSMA exclusion audit for Reviewer 4.4

## Identity

The method called “Lapucci-H” in the submitted manuscript is the
**Non-dominated Sorting Memetic Algorithm (NSMA)** of Lapucci, Mansueto, and
Schoen, *Mathematical Programming Computation* 15, 227–267 (2023), DOI
`10.1007/s12532-022-00231-3`.

- Article: <https://link.springer.com/article/10.1007/s12532-022-00231-3>
- Official implementation: <https://github.com/pierlumanzu/nsma>

## Verified algorithmic mismatch

NSMA inherits NSGA-II selection, then chooses rank-zero points above a
crowding-distance quantile for Front Multi-Objective Projected Gradient
(FMOPG) refinement. FMOPG computes common descent directions, solves a
direction subproblem, and performs a line search. It is therefore not a
drop-in objective-only evolutionary comparator and it is not accurately
described as a “global-rank SLSQP hybrid.”

The official environment declares TensorFlow and optionally Gurobi; the
repository also provides a SciPy/HiGHS alternative to Gurobi. Dependency
availability is not the decisive exclusion reason.

## Verified protocol mismatch

The article's main experiments:

- stop most methods after a two-minute wall-clock limit;
- execute NSMA and NSGA-II five times per problem;
- select only the run with the best purity as the reported output;
- compare against gradient and derivative-free methods with different
  computational primitives.

The HLDBEA revision protocol instead predeclares 30 seeds, reports every run,
and enforces a common objective-evaluation budget that also charges HLDBEA's
solver calls.

## Decision

NSMA is excluded from the equal-FE tables because the revision does not have a
validated, source-faithful rule that converts its objective, gradient,
direction-subproblem, and line-search work to the objective-only FE ledger.
Counting only objective vectors would grant NSMA uncharged derivative
information; treating each gradient component as one FE would be an invented
cost model. Reproducing the paper's time-limit/best-of-five protocol would
likewise break the paired-seed FE design.

The manuscript and response must state this explicitly. SMS-EMOA is not
excluded and is executed in the common FE validation block.
