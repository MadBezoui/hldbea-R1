#!/usr/bin/env bash
# Replays every HLDBEA run of the five confirmatory v3 blocks with the duplicate
# counters. The counters only read the population, so each replay must match
# the archived arrays bit for bit.
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg PYTHONPATH=src
workers="${WORKERS:-5}"
for block in manyobjective dtlz3 geometry-ablation modern-benchmarks rotation; do
  stem="reviewer-${block}-validation-v3"
  .venv/bin/python analysis/replay_duplicates.py \
    --manifest "experiments/manifests/${stem}.yaml" --artifact-root results/raw \
    --workers "$workers" --output "results/reports/${stem}-duplicates.json"
done
