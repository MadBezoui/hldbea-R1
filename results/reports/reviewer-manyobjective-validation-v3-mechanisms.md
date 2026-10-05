# Mechanism and failure summary

Manifest: `reviewer-manyobjective-validation-v3`. All rows passed the analysis validation gate.

| Algorithm | n | HV>0 | Rate | Median HV | Median IGD+ | Solver calls | Solver FE | Accepted | Accepted HV gain | Restart triggers | Runs restarted | Replacements | Median wall s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fdsea:upstream-default | 120 | 120 | 1.000 | 1.33952 | 0.383754 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2.630 |
| hldbea:cone-fixed-0p1 | 120 | 117 | 0.975 | 0.565462 | 0.681716 | 5778 | 626628 | 2089 | 2.025e+09 | 0 | 0 | 0 | 5.493 |
| hldbea:cone-linear | 120 | 119 | 0.992 | 0.520225 | 0.692521 | 5788 | 625760 | 2071 | 2.0965e+09 | 0 | 0 | 0 | 5.503 |
| hldbea:cone-saturating | 120 | 116 | 0.967 | 0.503483 | 0.686211 | 5788 | 625732 | 2052 | 1.93674e+09 | 1 | 1 | 20 | 5.541 |
| hldbea:cone-strict | 120 | 120 | 1.000 | 0.691233 | 0.615045 | 6047 | 599642 | 839 | 1.13816e+09 | 2 | 2 | 40 | 5.142 |
| maoea_hap:upstream-default | 120 | 120 | 1.000 | 2.44664 | 0.208441 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4.228 |
| nsga3:standard | 120 | 120 | 1.000 | 2.4667 | 0.171699 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1.775 |
| rvea:standard | 120 | 120 | 1.000 | 2.32864 | 0.191943 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.672 |
