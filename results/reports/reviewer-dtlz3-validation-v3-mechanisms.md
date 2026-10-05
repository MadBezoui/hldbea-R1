# Mechanism and failure summary

Manifest: `reviewer-dtlz3-validation-v3`. All rows passed the analysis validation gate.

| Algorithm | n | HV>0 | Rate | Median HV | Median IGD+ | Solver calls | Solver FE | Accepted | Accepted HV gain | Restart triggers | Runs restarted | Replacements | Median wall s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hldbea:augmented-no-ls | 30 | 23 | 0.767 | 0.600014 | 0.0788134 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 7.259 |
| hldbea:augmented-restart-no-ls | 30 | 28 | 0.933 | 0.602439 | 0.0791973 | 0 | 0 | 0 | 0 | 336 | 30 | 6048 | 7.887 |
| hldbea:augmented-with-ls | 30 | 13 | 0.433 | 0 | 1.016 | 9115 | 549814 | 2939 | 1.15158 | 0 | 0 | 0 | 5.631 |
| hldbea:core-no-ls | 30 | 20 | 0.667 | 0.609903 | 0.0747997 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 7.195 |
| hldbea:core-with-ls | 30 | 7 | 0.233 | 0 | 1.10675 | 9584 | 506944 | 4986 | 0.202033 | 0 | 0 | 0 | 5.901 |
| hldbea:full | 30 | 12 | 0.400 | 0 | 1.03942 | 9071 | 548143 | 3176 | 1.22908 | 314 | 30 | 5652 | 5.889 |
| nsga3:standard | 30 | 24 | 0.800 | 0.672933 | 0.0493388 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2.779 |
