# Validated comparison results

`+` means the reference is significantly better after Holm correction; `-` means significantly worse; `=` means no corrected difference.

| Problem | Metric | Reference n, median [IQR]; 95% CI | Comparator n, median [IQR]; 95% CI | p raw | p Holm | A12 | Sign |
|---|---|---|---|---:|---:|---:|:---:|
| validation-rlinear-m3-a0 | hv | hldbea:full (n=30): 1.073 [1.066, 1.078]; CI [1.069, 1.075] | nsga2:standard (n=30): 1.083 [1.08, 1.088]; CI [1.081, 1.086] | 4.657e-08 | 1.863e-07 | 0.113 | - |
| validation-rlinear-m3-a0 | hv | hldbea:full (n=30): 1.073 [1.066, 1.078]; CI [1.069, 1.075] | nsga3:standard (n=30): **1.115 [1.114, 1.116]; CI [1.114, 1.116]** | 1.863e-09 | 1.863e-08 | 0.000 | - |
| validation-rlinear-m3-a0 | igd_plus | hldbea:full (n=30): 0.04662 [0.04467, 0.04874]; CI [0.04572, 0.04828] | nsga2:standard (n=30): 0.04285 [0.04165, 0.04427]; CI [0.04211, 0.04357] | 1.824e-05 | 7.298e-05 | 0.148 | - |
| validation-rlinear-m3-a0 | igd_plus | hldbea:full (n=30): 0.04662 [0.04467, 0.04874]; CI [0.04572, 0.04828] | nsga3:standard (n=30): **0.02212 [0.02122, 0.02261]; CI [0.02155, 0.02247]** | 1.863e-09 | 1.863e-08 | 0.000 | - |
| validation-rlinear-m3-a0 | phi_nz | hldbea:full (n=30): **0 [0, 0]; CI [0, 0]** | nsga2:standard (n=30): **0 [0, 0]; CI [0, 0]** | 1 | 1 | 0.500 | = |
| validation-rlinear-m3-a0 | phi_nz | hldbea:full (n=30): **0 [0, 0]; CI [0, 0]** | nsga3:standard (n=30): **0 [0, 0]; CI [0, 0]** | 1 | 1 | 0.500 | = |
| validation-rlinear-m3-a0 | r_fz | hldbea:full (n=30): **0 [0, 0]; CI [0, 0]** | nsga2:standard (n=30): **0 [0, 0]; CI [0, 0]** | 1 | 1 | 0.500 | = |
| validation-rlinear-m3-a0 | r_fz | hldbea:full (n=30): **0 [0, 0]; CI [0, 0]** | nsga3:standard (n=30): **0 [0, 0]; CI [0, 0]** | 1 | 1 | 0.500 | = |
| validation-rlinear-m3-a15 | hv | hldbea:full (n=30): 0.7763 [0.77, 0.7837]; CI [0.7721, 0.7809] | nsga2:standard (n=30): 0.8031 [0.7996, 0.8049]; CI [0.8013, 0.8041] | 1.863e-09 | 1.863e-08 | 0.000 | - |
| validation-rlinear-m3-a15 | hv | hldbea:full (n=30): 0.7763 [0.77, 0.7837]; CI [0.7721, 0.7809] | nsga3:standard (n=30): **0.8212 [0.8193, 0.8227]; CI [0.82, 0.8224]** | 1.863e-09 | 1.863e-08 | 0.000 | - |
| validation-rlinear-m3-a15 | igd_plus | hldbea:full (n=30): 0.0447 [0.04344, 0.04575]; CI [0.04427, 0.04554] | nsga2:standard (n=30): 0.0423 [0.04128, 0.04345]; CI [0.04186, 0.0432] | 0.0002563 | 0.0007689 | 0.209 | - |
| validation-rlinear-m3-a15 | igd_plus | hldbea:full (n=30): 0.0447 [0.04344, 0.04575]; CI [0.04427, 0.04554] | nsga3:standard (n=30): **0.0303 [0.02976, 0.03101]; CI [0.03012, 0.03075]** | 1.863e-09 | 1.863e-08 | 0.000 | - |
| validation-rlinear-m3-a15 | phi_nz | hldbea:full (n=30): **0 [0, 0]; CI [0, 0]** | nsga2:standard (n=30): **0 [0, 0]; CI [0, 0]** | 1 | 1 | 0.500 | = |
| validation-rlinear-m3-a15 | phi_nz | hldbea:full (n=30): **0 [0, 0]; CI [0, 0]** | nsga3:standard (n=30): **0 [0, 0]; CI [0, 0]** | 1 | 1 | 0.500 | = |
| validation-rlinear-m3-a15 | r_fz | hldbea:full (n=30): **0 [0, 0]; CI [0, 0]** | nsga2:standard (n=30): **0 [0, 0]; CI [0, 0]** | 1 | 1 | 0.500 | = |
| validation-rlinear-m3-a15 | r_fz | hldbea:full (n=30): **0 [0, 0]; CI [0, 0]** | nsga3:standard (n=30): **0 [0, 0]; CI [0, 0]** | 1 | 1 | 0.500 | = |
| validation-rlinear-m3-a30 | hv | hldbea:full (n=30): 0.5876 [0.5827, 0.5907]; CI [0.5838, 0.5897] | nsga2:standard (n=30): 0.6033 [0.5982, 0.6083]; CI [0.5996, 0.6049] | 4.657e-08 | 1.863e-07 | 0.054 | - |
| validation-rlinear-m3-a30 | hv | hldbea:full (n=30): 0.5876 [0.5827, 0.5907]; CI [0.5838, 0.5897] | nsga3:standard (n=30): **0.6212 [0.619, 0.6225]; CI [0.6206, 0.6217]** | 1.863e-09 | 1.863e-08 | 0.000 | - |
| validation-rlinear-m3-a30 | igd_plus | hldbea:full (n=30): 0.0463 [0.04403, 0.04724]; CI [0.04452, 0.04719] | nsga2:standard (n=30): 0.04576 [0.04316, 0.04776]; CI [0.04422, 0.047] | 0.428 | 0.428 | 0.444 | = |
| validation-rlinear-m3-a30 | igd_plus | hldbea:full (n=30): 0.0463 [0.04403, 0.04724]; CI [0.04452, 0.04719] | nsga3:standard (n=30): **0.03262 [0.03174, 0.03307]; CI [0.03211, 0.03277]** | 1.863e-09 | 1.863e-08 | 0.000 | - |
| validation-rlinear-m3-a30 | phi_nz | hldbea:full (n=30): **0 [0, 0]; CI [0, 0]** | nsga2:standard (n=30): **0 [0, 0]; CI [0, 0]** | 1 | 1 | 0.500 | = |
| validation-rlinear-m3-a30 | phi_nz | hldbea:full (n=30): **0 [0, 0]; CI [0, 0]** | nsga3:standard (n=30): **0 [0, 0]; CI [0, 0]** | 1 | 1 | 0.500 | = |
| validation-rlinear-m3-a30 | r_fz | hldbea:full (n=30): **0 [0, 0]; CI [0, 0]** | nsga2:standard (n=30): **0 [0, 0]; CI [0, 0]** | 1 | 1 | 0.500 | = |
| validation-rlinear-m3-a30 | r_fz | hldbea:full (n=30): **0 [0, 0]; CI [0, 0]** | nsga3:standard (n=30): **0 [0, 0]; CI [0, 0]** | 1 | 1 | 0.500 | = |
| validation-rlinear-m3-a45 | hv | hldbea:full (n=30): 0.4249 [0.4214, 0.4313]; CI [0.423, 0.4289] | nsga2:standard (n=30): 0.4395 [0.4366, 0.4431]; CI [0.4377, 0.4418] | 3.856e-07 | 3.856e-07 | 0.080 | - |
| validation-rlinear-m3-a45 | hv | hldbea:full (n=30): 0.4249 [0.4214, 0.4313]; CI [0.423, 0.4289] | nsga3:standard (n=30): **0.4513 [0.4503, 0.4527]; CI [0.4505, 0.4516]** | 1.863e-09 | 1.863e-08 | 0.000 | - |
| validation-rlinear-m3-a45 | igd_plus | hldbea:full (n=30): 0.04646 [0.04435, 0.04823]; CI [0.04483, 0.04714] | nsga2:standard (n=30): 0.04386 [0.04166, 0.04532]; CI [0.04304, 0.04464] | 0.004338 | 0.008676 | 0.238 | - |
| validation-rlinear-m3-a45 | igd_plus | hldbea:full (n=30): 0.04646 [0.04435, 0.04823]; CI [0.04483, 0.04714] | nsga3:standard (n=30): **0.03592 [0.03509, 0.03629]; CI [0.03537, 0.0362]** | 1.863e-09 | 1.863e-08 | 0.000 | - |
| validation-rlinear-m3-a45 | phi_nz | hldbea:full (n=30): **0 [0, 0]; CI [0, 0]** | nsga2:standard (n=30): **0 [0, 0]; CI [0, 0]** | 1 | 1 | 0.517 | = |
| validation-rlinear-m3-a45 | phi_nz | hldbea:full (n=30): **0 [0, 0]; CI [0, 0]** | nsga3:standard (n=30): **0 [0, 0]; CI [0, 0]** | 1 | 1 | 0.517 | = |
| validation-rlinear-m3-a45 | r_fz | hldbea:full (n=30): **0 [0, 0]; CI [0, 0]** | nsga2:standard (n=30): **0 [0, 0]; CI [0, 0]** | 1 | 1 | 0.483 | = |
| validation-rlinear-m3-a45 | r_fz | hldbea:full (n=30): **0 [0, 0]; CI [0, 0]** | nsga3:standard (n=30): **0 [0, 0]; CI [0, 0]** | 1 | 1 | 0.483 | = |
| validation-rlinear-m3-a60 | hv | hldbea:full (n=30): 0.2633 [0.2621, 0.2661]; CI [0.2625, 0.2653] | nsga2:standard (n=30): 0.2695 [0.2679, 0.2707]; CI [0.2681, 0.2704] | 1.639e-07 | 3.278e-07 | 0.090 | - |
| validation-rlinear-m3-a60 | hv | hldbea:full (n=30): 0.2633 [0.2621, 0.2661]; CI [0.2625, 0.2653] | nsga3:standard (n=30): **0.2742 [0.2731, 0.2755]; CI [0.2734, 0.275]** | 1.863e-09 | 1.863e-08 | 0.000 | - |
| validation-rlinear-m3-a60 | igd_plus | hldbea:full (n=30): 0.04945 [0.04794, 0.05282]; CI [0.04844, 0.05134] | nsga2:standard (n=30): 0.04463 [0.04346, 0.04523]; CI [0.04371, 0.0451] | 9.313e-09 | 4.657e-08 | 0.033 | - |
| validation-rlinear-m3-a60 | igd_plus | hldbea:full (n=30): 0.04945 [0.04794, 0.05282]; CI [0.04844, 0.05134] | nsga3:standard (n=30): **0.04047 [0.03895, 0.0413]; CI [0.03949, 0.04114]** | 1.863e-09 | 1.863e-08 | 0.000 | - |
| validation-rlinear-m3-a60 | phi_nz | hldbea:full (n=30): **0 [0, 0]; CI [0, 0]** | nsga2:standard (n=30): **0 [0, 0]; CI [0, 0]** | 1 | 1 | 0.500 | = |
| validation-rlinear-m3-a60 | phi_nz | hldbea:full (n=30): **0 [0, 0]; CI [0, 0]** | nsga3:standard (n=30): **0 [0, 0]; CI [0, 0]** | 1 | 1 | 0.500 | = |
| validation-rlinear-m3-a60 | r_fz | hldbea:full (n=30): **0 [0, 0]; CI [0, 0]** | nsga2:standard (n=30): **0 [0, 0]; CI [0, 0]** | 1 | 1 | 0.500 | = |
| validation-rlinear-m3-a60 | r_fz | hldbea:full (n=30): **0 [0, 0]; CI [0, 0]** | nsga3:standard (n=30): **0 [0, 0]; CI [0, 0]** | 1 | 1 | 0.500 | = |

Reference totals: 0 wins / 21 ties / 19 losses.
