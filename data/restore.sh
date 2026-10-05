#!/usr/bin/env bash
# Reassemble the raw outputs of the confirmatory runs and unpack them into results/raw.
set -euo pipefail
cd "$(dirname "$0")"
cat hldbea-v3-raw-runs.tar.gz.part-* > hldbea-v3-raw-runs.tar.gz
shasum -a 256 -c hldbea-v3-raw-runs.tar.gz.sha256
mkdir -p ../results/raw
tar -xzf hldbea-v3-raw-runs.tar.gz -C ../results/raw
rm hldbea-v3-raw-runs.tar.gz
echo "Raw runs restored in results/raw"
