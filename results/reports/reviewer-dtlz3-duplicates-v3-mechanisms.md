# Mechanism and failure summary

Manifest: `reviewer-dtlz3-duplicates-v3`. All rows passed the analysis validation gate.

| Algorithm | n | HV>0 | Rate | Median HV | Median IGD+ | Solver calls | Solver FE | Accepted | Accepted HV gain | Restart triggers | Runs restarted | Replacements | Median wall s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hldbea:core-no-ls | 30 | 20 | 0.667 | 0.609903 | 0.0747997 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 9.469 |
| hldbea:core-no-ls-drop | 30 | 22 | 0.733 | 0.595519 | 0.0757065 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 8.401 |
| hldbea:core-no-ls-eliminate | 30 | 25 | 0.833 | 0.649433 | 0.0583609 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 11.886 |
| hldbea:core-with-ls | 30 | 7 | 0.233 | 0 | 1.10675 | 9584 | 506944 | 4986 | 0.202033 | 0 | 0 | 0 | 7.807 |
| hldbea:core-with-ls-drop | 30 | 9 | 0.300 | 0 | 1.12995 | 9549 | 505510 | 5003 | 0.265031 | 0 | 0 | 0 | 7.577 |
| hldbea:core-with-ls-eliminate | 30 | 13 | 0.433 | 0 | 1.02994 | 9636 | 502175 | 5085 | 0.39081 | 0 | 0 | 0 | 8.602 |
| hldbea:full | 30 | 12 | 0.400 | 0 | 1.03942 | 9071 | 548143 | 3176 | 1.22908 | 314 | 30 | 5652 | 7.359 |
| hldbea:full-drop | 30 | 10 | 0.333 | 0 | 1.05367 | 9094 | 540739 | 2690 | 1.42133 | 254 | 30 | 4572 | 7.335 |
| hldbea:full-eliminate | 30 | 12 | 0.400 | 0 | 1.02243 | 9124 | 544079 | 2595 | 1.21 | 261 | 30 | 4698 | 7.420 |
