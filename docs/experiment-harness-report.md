# Experiment harness acceptance report

Date: 2026-10-03 (Europe/Paris)

## Acceptance result

The evidence-oriented harness passes its integration acceptance gate for
`integration-smoke-v1`. All 16 planned runs completed and validated: 16 valid,
0 invalid, 0 failed, 0 missing, and 0 duplicate active artifacts. Every run
used exactly 80 objective-function evaluations, all eight problem/algorithm
groups contain the paired seeds 31001 and 31002, and each of the four problem
instances has one stable 64-character reference-geometry hash.

Serial and two-worker executions were compared after removing only runtime and
worker identity metadata. All 16 semantic hashes matched. That check exposed
and led to the repair of pymoo's stochastic WFG reference-front generation:
reference construction now uses a problem-derived seed and restores NumPy's
global RNG state. A second contamination defect was also found and repaired:
every SPEA2 run now receives a fresh normalization/survival object.

The final regression command reports **196 passed**. The remaining warnings
are pymoo 0.6.1.3 deprecation messages for its `Individual.feasible` property;
they do not alter results or gate status.

## Exact environment

- Python 3.10.1
- macOS 27.2, arm64
- 10 logical CPUs; four-worker conservative execution cap
- NumPy 1.26.4
- SciPy 1.14.1
- pandas 2.3.3
- pymoo 0.6.1.3
- Matplotlib 3.10.7 with the Agg batch backend
- PyYAML 6.0.2
- numba 0.61.2 and llvmlite 0.44.0
- pytest 8.4.1

The dependency set is declared in `pyproject.toml`. Run artifacts also retain
the exact interpreter/dependency versions, platform, worker, Git commit and
dirty-state flag used for each execution.

## Commands executed

```bash
# Enumerate and execute the two-seed integration matrix.
PYTHONPATH=src MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python experiments/run.py \
  --manifest experiments/manifests/integration-smoke.yaml \
  --artifact-root results/raw --workers 2

# Validate structural and analysis gates.
PYTHONPATH=src MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python analysis/validate_runs.py \
  --manifest experiments/manifests/integration-smoke.yaml \
  --artifact-root results/raw \
  --json-output results/reports/integration-smoke-validation.json \
  --markdown-output results/reports/integration-smoke-validation.md \
  --gate analysis

# Predeclared paired statistics.
PYTHONPATH=src MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python analysis/statistics.py \
  --manifest experiments/manifests/integration-smoke.yaml \
  --artifact-root results/raw --reference-algorithm hldbea \
  --output results/reports/integration-smoke-statistics.json \
  --bootstrap-samples 10000 --bootstrap-seed 20261003

# Tables and publication figures with provenance hashes.
PYTHONPATH=src MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python analysis/tables.py \
  --manifest experiments/manifests/integration-smoke.yaml \
  --artifact-root results/raw \
  --statistics results/reports/integration-smoke-statistics.json \
  --output-dir results/tables --stem integration-smoke

PYTHONPATH=src MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python analysis/figures.py \
  --manifest experiments/manifests/integration-smoke.yaml \
  --artifact-root results/raw --output-dir results/figures \
  --stem integration-smoke --format png

# Complete verification.
MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python -m pytest -q
git diff --check
```

The algorithm-adapter test executes every admitted adapter (HLDBEA, NSGA-II,
SPEA2, NSGA-III, RVEA, MOEA/D, SMS-EMOA and AGE-MOEA) on both DTLZ2 and WFG2
under an exact shared FE ledger.

## Run counts and resource use

The integration design is 2 algorithms × 4 problems × 2 seeds = 16 runs. It
charged 1,280 FE in total: 750 evolutionary FE and 530 local-solver FE. From
the immutable per-run metadata, summed wall time is 2.474244 s and summed CPU
time is 2.017037 s; per-run wall time ranges from 0.048858 s to 0.386195 s with
a median of 0.131058 s. Maximum recorded process RSS is 101,744,640 bytes.
Every run records commit `8064017806f16554492ee4c10490ac06cb27528f` and
`git.dirty=false`.

Exact retained file sizes at acceptance were:

- raw integration artifacts: 64 files, 121,566 bytes;
- validation/statistical reports: 3 files, 21,020 bytes;
- generated tables and their provenance: 5 files, 8,152 bytes;
- generated figures and their provenance: 13 files, 681,836 bytes.

The dry run for `calibration-v1` expands to 110 runs (11 one-factor variants ×
2 calibration-only problems × 5 calibration-only seeds) and reports a
conservative maximum footprint of 2,259,560,160 bytes. It has not been
executed. With only 9.1 GiB free at handoff, it remains capped at four workers
and must pass a fresh disk gate immediately before dispatch.

## Dependency and comparator gates

All eight currently admitted Python adapters passed DTLZ2 and WFG2 smoke
execution with exact budgets. AGE-MOEA's optional numba dependency is installed
and pinned, so no admitted adapter was excluded for a missing dependency.

No result in this report is labeled as MaOEA-PGTS, MaOEA-ISAE,
Lapucci-H/Sindhya-H, “BiO-IM”, or another recent comparator. Those methods
remain outside the executed registry until an identifiable paper, compatible
reference implementation or independently testable exact implementation, and
license/reproducibility audit all pass. The unidentified “BiO-IM / Modern
Geometry-Driven MOEA” label is explicitly prohibited rather than silently
substituted.

## Calibration and validation separation

`calibration-v1` uses only DTLZ1 and WFG1, seeds 41001–41005, and a
one-factor-at-a-time grid over reviewer-relevant neighborhood size, cone
relaxation, local-search depth and restart sensitivity. The next frozen
validation design reserves DTLZ2/3/4 and WFG2/3/9 (including rotated cases) and
seeds 51001–51030. Calibration and validation therefore share neither problem
instances nor random seeds. Calibration outcomes may select parameters, but
validation seeds must remain unopened until the configuration is frozen. The
statistics CLI accepts `--reference-variant baseline`, so the calibration grid
can be compared without conflating its eleven HLDBEA variants.

## Known limitations

- `integration-smoke-v1` is an engineering gate, not performance evidence: it
  has only two seeds, two algorithms, four small-budget cases and no rotated
  objective instances.
- Friedman is correctly marked not applicable because the smoke matrix has
  fewer than three algorithms. Its Wilcoxon results have very low power and
  must not support manuscript superiority claims.
- The calibration campaign is defined but intentionally not yet executed.
- Reviewer-requested ablations, DTLZ3 restart analysis, rotated/PCA studies,
  8/10/15-objective validation, and qualifying recent comparators belong to
  the next frozen campaign.
- Raw artifacts are reproducible local evidence under ignored `results/raw/`;
  the committed validation report, statistics, tables, figures and provenance
  files are the compact reviewable evidence index.

This report accepts the harness only. It does not validate the unsupported
performance claims in the pre-existing manuscript; those must be replaced
after calibration and the reserved validation campaign are complete.
