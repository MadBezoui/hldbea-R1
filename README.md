# HLDBEA, code and data for the revised paper

This repository contains the implementation of HLDBEA, a hybrid local dominance-based evolutionary algorithm for continuous multiobjective optimization, together with everything needed to reproduce the experiments, tables and figures of the paper "A hybrid local dominance-based evolutionary algorithm for continuous multiobjective optimization" (El-Ghani Iftissen, Boualem Brahmi and Madani Bezoui).

## Contents

`src/` holds the algorithm, the benchmarks (including the IMOP problems and the rotated linear front), the ports of MaOEA-HAP and FDSEA, the evaluation ledger and the experiment runner. `experiments/manifests/` lists every run of the paper in the files ending in `-v3`. Older manifests, the legacy `configs/` files, `nibea_test.py` and the smoke-test outputs in `results/` are kept because some tests use them. `analysis/` validates the runs, computes the statistics and produces the tables and figures. `results/reports/` and `results/tables/` contain the validation reports, statistical reports, duplicate replay reports and tables of the v3 runs. `results/paper-assets/` holds the output of the generator for the submitted manuscript, and its `manuscript-numbers.json` also lists the ideal point, nadir point and HV reference point of every rotation angle. `docs/` documents the provenance of the comparators and benchmarks, the NSMA exclusion and the source state of the v3 runs.

`data/` holds the raw outputs of 4,071 v3 runs in one archive split into three parts to fit GitHub's file size limit:

| Runs | Content |
|---:|---|
| 3,480 | Five confirmatory blocks (DTLZ3 210, IMOP 900, component ablation 960, many objectives 960, rotation 450) |
| 270 | Follow-up DTLZ3 ablation of duplicate handling |
| 321 | Three calibration screens (cone 96, core 108, restart 117) |

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

The first command reassembles `hldbea-v3-raw-runs.tar.gz`, checks its SHA-256 against `data/hldbea-v3-raw-runs.tar.gz.sha256` and unpacks it into `results/raw`. The second validates every block and checks that each statistical report was computed from exactly these runs: the observation digest, medians, paired Wilcoxon p-values, Holm decisions and effect sizes are recomputed. It also checks that each duplicate replay report covers the validated HLDBEA runs and matches their archived arrays. It then writes the tables, the figures and the numbers quoted in the text, and refuses to write anything if a check fails.

## Re-running the experiments

```bash
bash experiments/run_reviewer_queue_v3.sh
bash experiments/run_dtlz3_duplicates_v3.sh
bash experiments/replay_duplicates_v3.sh
```

The first script runs the calibration screens and the five confirmatory blocks with three workers and one numerical thread per worker, validates each block and writes the statistical reports. The second runs the follow-up DTLZ3 ablation and replays four of its configurations with dense checkpoints. The third replays every HLDBEA run of the confirmatory blocks with the duplicate counters and checks that each replay reproduces the archived run bit for bit. Runs are resumable and every completed run is stored as an immutable, checksummed artifact. The full campaign takes several hours on a laptop, most of it spent on SMS-EMOA.

## Tests

```bash
PYTHONPATH=src MPLBACKEND=Agg .venv/bin/python -m pytest -q
```

## Statistical protocol

Each block compares one reference configuration with every other configuration on every problem using two-sided paired Wilcoxon signed-rank tests on 30 seeds, discarding zero differences. Holm correction is applied within one family per block and indicator. Reports also contain bootstrap intervals, the matched-pairs rank-biserial correlation and the Vargha-Delaney effect size. In the follow-up DTLZ3 ablation the paper compares each configuration with the core without local search under the same duplicate handling, with Holm correction over the six comparisons of each indicator.

## License

The code is released under the MIT License (`LICENSE`), except the ports of PlatEMO code in `src/hldbea/recent_algorithms.py` and `src/hldbea/imop.py`, which remain subject to the PlatEMO terms of use. The raw runs in `data/`, the reports in `results/` and the documentation in `docs/` are released under the Creative Commons Attribution 4.0 International License (`LICENSE-DATA`).
