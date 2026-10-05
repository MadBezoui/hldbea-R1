# Mechanism and failure summary

Manifest: `reviewer-geometry-ablation-validation-v3`. All rows passed the analysis validation gate.

| Algorithm | n | HV>0 | Rate | Median HV | Median IGD+ | Solver calls | Solver FE | Accepted | Accepted HV gain | Restart triggers | Runs restarted | Replacements | Median wall s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hldbea:axis-sum | 120 | 120 | 1.000 | 0.537428 | 0.0804028 | 13377 | 1072580 | 2827 | 16.288 | 31 | 24 | 620 | 4.709 |
| hldbea:axis-union | 120 | 120 | 1.000 | 0.538375 | 0.0745096 | 13376 | 1072992 | 2872 | 16.566 | 28 | 25 | 560 | 3.989 |
| hldbea:box-union | 120 | 120 | 1.000 | 0.54162 | 0.0796884 | 13283 | 1081819 | 2742 | 16.6311 | 32 | 25 | 640 | 3.721 |
| hldbea:candidate-global-rank | 120 | 120 | 1.000 | 0.538909 | 0.0854186 | 12884 | 1120872 | 3910 | 31.376 | 24 | 22 | 480 | 4.108 |
| hldbea:candidate-random | 120 | 120 | 1.000 | 0.54232 | 0.0832771 | 12869 | 1123029 | 4030 | 32.0907 | 25 | 23 | 500 | 4.002 |
| hldbea:knn-union | 120 | 120 | 1.000 | 0.541846 | 0.0782881 | 13359 | 1074628 | 2827 | 16.9288 | 31 | 25 | 620 | 4.393 |
| hldbea:objective-adaptive | 120 | 120 | 1.000 | 0.528192 | 0.0788553 | 12910 | 1117323 | 862 | 5.12884 | 17 | 14 | 340 | 4.135 |
| hldbea:selection-global | 120 | 120 | 1.000 | 0.538689 | 0.0905652 | 13436 | 1066306 | 2715 | 17.078 | 37 | 29 | 740 | 4.688 |
