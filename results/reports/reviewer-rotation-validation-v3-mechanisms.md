# Mechanism and failure summary

Manifest: `reviewer-rotation-validation-v3`. All rows passed the analysis validation gate.

| Algorithm | n | HV>0 | Rate | Median HV | Median IGD+ | Solver calls | Solver FE | Accepted | Accepted HV gain | Restart triggers | Runs restarted | Replacements | Median wall s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hldbea:full | 150 | 150 | 1.000 | 0.587587 | 0.0464972 | 9596 | 549265 | 40 | 7.20266 | 7 | 7 | 140 | 1.518 |
| nsga2:standard | 150 | 150 | 1.000 | 0.603291 | 0.0436155 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.520 |
| nsga3:standard | 150 | 150 | 1.000 | 0.621157 | 0.0326219 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.642 |
