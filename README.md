# HLDBEA, code and data for the revised paper

This repository contains the implementation of HLDBEA, a hybrid local dominance-based evolutionary algorithm for continuous multiobjective optimization, together with everything needed to reproduce the confirmatory experiments, tables and figures of the paper "A hybrid local dominance-based evolutionary algorithm for continuous multiobjective optimization" (El-Ghani Iftissen, Boualem Brahmi and Madani Bezoui).

## Contents

`src/` holds the algorithm, the benchmarks (including the IMOP problems and the rotated linear front), the ports of MaOEA-HAP and FDSEA, the evaluation ledger and the experiment runner. `experiments/manifests/` lists every run of the paper: the three calibration screens and the five confirmatory blocks use the files ending in `-v3`. Older manifests, the legacy `configs/` files, `nibea_test.py` and the smoke-test outputs in `results/` are kept because some tests use them. `analysis/` validates the runs, computes the statistics and produces the tables and figures. `results/reports/` and `results/tables/` contain the validation reports, statistical reports and tables of the v3 runs. `data/` holds the raw outputs of all 3,801 v3 runs, split into two parts to fit GitHub's file size limit. `docs/` documents the provenance of the comparators and benchmarks, the NSMA exclusion and the source state of the v3 runs.

## Installation

Python 3.10 is the reference interpreter.

```bash
python3.10 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m pip install --no-deps -e .
```

`requirements-lock.txt` gives the exact versions used for the paper (NumPy 1.26.4, SciPy 1.14.1, pymoo 0.6.1.3, numba 0.61.2).

## Reproducing the tables and figures from the raw runs

```bash
bash data/restore.sh
PYTHONPATH=src MPLBACKEND=Agg .venv/bin/python analysis/manuscript_assets.py --output-dir paper-assets
```

The first command reassembles the archive, checks its SHA-256 (`3d4f54e54ef5ab6ec7cfd16fded9454a68064ceb2869621029d83c2c802ee1ab`) and unpacks it into `results/raw`. The second validates every block, checks that each statistical report was computed from exactly these runs (observation digest, medians, paired Wilcoxon p-values and Holm decisions are recomputed), and writes the seven tables, the four figures and the numbers quoted in the text. It refuses to write anything if a check fails.

## Re-running the experiments

```bash
bash experiments/run_reviewer_queue_v3.sh
```

This runs the calibration screens and the five confirmatory blocks with three workers and one numerical thread per worker, validates each block and writes the statistical reports. Runs are resumable and every completed run is stored as an immutable, checksummed artifact. The full campaign takes several hours on a laptop, most of it spent on SMS-EMOA.

## Tests

```bash
PYTHONPATH=src MPLBACKEND=Agg .venv/bin/python -m pytest -q
```

## Statistical protocol

Each block compares one reference configuration with every other configuration on every problem using two-sided paired Wilcoxon signed-rank tests on 30 seeds, discarding zero differences. Holm correction is applied within one family per block and indicator. Reports also contain bootstrap intervals, the matched-pairs rank-biserial correlation and the Vargha-Delaney effect size.
