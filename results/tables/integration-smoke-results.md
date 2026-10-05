# Validated comparison results

`+` means the reference is significantly better after Holm correction; `-` means significantly worse; `=` means no corrected difference.

| Problem | Metric | Reference n, median [IQR]; 95% CI | Comparator n, median [IQR]; 95% CI | p raw | p Holm | A12 | Sign |
|---|---|---|---|---:|---:|---:|:---:|
| dtlz2-m3 | hv | hldbea:full (n=2): 0.004123 [0.002588, 0.005658]; CI [0.001053, 0.007193] | nsga2:standard (n=2): **0.01237 [0.008776, 0.01596]; CI [0.005185, 0.01955]** | 1 | 1 | 0.250 | = |
| dtlz2-m3 | igd_plus | hldbea:full (n=2): 0.6498 [0.6444, 0.6553]; CI [0.6389, 0.6607] | nsga2:standard (n=2): **0.4972 [0.4775, 0.5169]; CI [0.4579, 0.5365]** | 0.5 | 1 | 0.000 | = |
| dtlz2-m5 | hv | hldbea:full (n=2): **0.01067 [0.005333, 0.016]; CI [0, 0.02133]** | nsga2:standard (n=2): 0 [0, 0]; CI [0, 0] | 1 | 1 | 0.750 | = |
| dtlz2-m5 | igd_plus | hldbea:full (n=2): 0.8195 [0.8077, 0.8313]; CI [0.7958, 0.8432] | nsga2:standard (n=2): **0.765 [0.7497, 0.7803]; CI [0.7343, 0.7956]** | 0.5 | 1 | 0.000 | = |
| wfg2-m3 | hv | hldbea:full (n=2): 0.3948 [0.3802, 0.4094]; CI [0.3656, 0.4241] | nsga2:standard (n=2): **0.5556 [0.5424, 0.5689]; CI [0.5292, 0.5821]** | 0.5 | 1 | 0.000 | = |
| wfg2-m3 | igd_plus | hldbea:full (n=2): 0.3886 [0.3764, 0.4008]; CI [0.3642, 0.4129] | nsga2:standard (n=2): **0.2896 [0.2833, 0.2958]; CI [0.2771, 0.302]** | 0.5 | 1 | 0.000 | = |
| wfg2-m5 | hv | hldbea:full (n=2): 0.4916 [0.4681, 0.5151]; CI [0.4445, 0.5387] | nsga2:standard (n=2): **0.6603 [0.6554, 0.6651]; CI [0.6506, 0.6699]** | 0.5 | 1 | 0.000 | = |
| wfg2-m5 | igd_plus | hldbea:full (n=2): 0.4744 [0.4477, 0.5012]; CI [0.4209, 0.5279] | nsga2:standard (n=2): **0.3763 [0.3652, 0.3874]; CI [0.3541, 0.3986]** | 0.5 | 1 | 0.000 | = |

Reference totals: 0 wins / 8 ties / 0 losses.
