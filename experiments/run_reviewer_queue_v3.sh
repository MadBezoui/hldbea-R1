#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg
export PYTHONPATH=src

python_bin=".venv/bin/python"
artifact_root="results/raw"
workers=3

run_block() {
  local manifest="$1"
  local stem="$2"
  local reference_algorithm="$3"
  local reference_variant="$4"

  "$python_bin" experiments/run.py \
    --manifest "$manifest" \
    --artifact-root "$artifact_root" \
    --workers "$workers" \
    --resume

  "$python_bin" analysis/validate_runs.py \
    --manifest "$manifest" \
    --artifact-root "$artifact_root" \
    --json-output "results/reports/${stem}-validation.json" \
    --markdown-output "results/reports/${stem}-validation.md" \
    --gate analysis

  "$python_bin" analysis/statistics.py \
    --manifest "$manifest" \
    --artifact-root "$artifact_root" \
    --reference-algorithm "$reference_algorithm" \
    --reference-variant "$reference_variant" \
    --metrics hv igd_plus phi_nz r_fz \
    --bootstrap-samples 10000 \
    --bootstrap-seed 20261003 \
    --output "results/reports/${stem}-statistics.json"

  "$python_bin" analysis/tables.py \
    --manifest "$manifest" \
    --artifact-root "$artifact_root" \
    --statistics "results/reports/${stem}-statistics.json" \
    --output-dir results/tables \
    --stem "$stem"

  "$python_bin" analysis/figures.py \
    --manifest "$manifest" \
    --artifact-root "$artifact_root" \
    --output-dir results/figures \
    --stem "$stem" \
    --format png

  "$python_bin" analysis/mechanisms.py \
    --manifest "$manifest" \
    --artifact-root "$artifact_root" \
    --json-output "results/reports/${stem}-mechanisms.json" \
    --markdown-output "results/reports/${stem}-mechanisms.md"
}

# v3: per-variable mutation (mutation_scope) for every pymoo-based algorithm,
# a rigidly rotated linear front, and a candidate-selection ablation.
for screen in core restart cone; do
  manifest="experiments/manifests/calibration-${screen}-screen-v3.yaml"
  "$python_bin" experiments/run.py --manifest "$manifest" \
    --artifact-root "$artifact_root" --workers "$workers" --resume
  "$python_bin" analysis/validate_runs.py --manifest "$manifest" \
    --artifact-root "$artifact_root" \
    --json-output "results/reports/calibration-${screen}-screen-v3-validation.json" \
    --markdown-output "results/reports/calibration-${screen}-screen-v3-validation.md" \
    --gate analysis
done
"$python_bin" analysis/calibrate.py --mode shortlist \
  --manifest experiments/manifests/calibration-core-screen-v3.yaml \
  --manifest experiments/manifests/calibration-restart-screen-v3.yaml \
  --manifest experiments/manifests/calibration-cone-screen-v3.yaml \
  --artifact-root "$artifact_root" \
  --report-output results/reports/calibration-shortlist-v3.json \
  --manifest-output results/reports/calibration-confirm-v3.yaml \
  || echo "calibration shortlist failed, continuing with validation blocks"

run_block experiments/manifests/reviewer-rotation-validation-v3.yaml \
  reviewer-rotation-validation-v3 hldbea full
run_block experiments/manifests/reviewer-geometry-ablation-validation-v3.yaml \
  reviewer-geometry-ablation-validation-v3 hldbea axis-sum
run_block experiments/manifests/reviewer-manyobjective-validation-v3.yaml \
  reviewer-manyobjective-validation-v3 hldbea cone-fixed-0p1
run_block experiments/manifests/reviewer-dtlz3-validation-v3.yaml \
  reviewer-dtlz3-validation-v3 hldbea core-no-ls
run_block experiments/manifests/reviewer-modern-benchmarks-validation-v3.yaml \
  reviewer-modern-benchmarks-validation-v3 hldbea full
