#!/usr/bin/env bash
# Targeted DTLZ3 ablation: does SLSQP degrade DTLZ3 because of the duplicate
# parents injected by rejected calls, or by itself?
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg PYTHONPATH=src
manifest=experiments/manifests/reviewer-dtlz3-duplicates-v3.yaml
stem=reviewer-dtlz3-duplicates-v3
.venv/bin/python experiments/run.py --manifest "$manifest" --artifact-root results/raw --workers 4 --resume
.venv/bin/python analysis/validate_runs.py --manifest "$manifest" --artifact-root results/raw \
  --json-output "results/reports/${stem}-validation.json" \
  --markdown-output "results/reports/${stem}-validation.md" --gate analysis
.venv/bin/python analysis/statistics.py --manifest "$manifest" --artifact-root results/raw \
  --reference-algorithm hldbea --reference-variant core-no-ls \
  --metrics hv igd_plus phi_nz r_fz --bootstrap-samples 10000 --bootstrap-seed 20261003 \
  --output "results/reports/${stem}-statistics.json"
.venv/bin/python analysis/mechanisms.py --manifest "$manifest" --artifact-root results/raw \
  --json-output "results/reports/${stem}-mechanisms.json" \
  --markdown-output "results/reports/${stem}-mechanisms.md"
# Same runs replayed with a checkpoint every five generations, to compare the
# configurations at an equal number of generations.
.venv/bin/python analysis/generation_matched.py --manifest "$manifest" --artifact-root results/raw \
  --variants core-no-ls core-with-ls core-no-ls-eliminate core-with-ls-eliminate \
  --step 460 --workers 4 --output "results/reports/${stem}-generations.json"
