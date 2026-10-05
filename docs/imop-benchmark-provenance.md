# IMOP benchmark provenance

Reviewer 3 requested newer and more structurally diverse problems than the
classical ZDT/DTLZ/WFG core. The validation suite therefore adds three members
of the 2019 irregular multi-objective problem (IMOP) suite:

- IMOP3: oscillatory disconnected bi-objective front;
- IMOP4: irregular degenerate three-objective curve;
- IMOP7: disconnected regions on a spherical three-objective front.

The problems were introduced by Y. Tian, R. Cheng, X. Zhang, M. Li, and
Y. Jin, “Diversity Assessment of Multi-Objective Evolutionary Algorithms:
Performance Metric and Benchmark Problems,” *IEEE Computational Intelligence
Magazine*, 14(3):61–74, 2019, DOI `10.1109/MCI.2019.2919398`.

## Frozen source

- Repository: `https://github.com/BIMK/PlatEMO`
- Commit: `d25e65d1ffba58dbf4d7e1b5259786187d12968a`
- Local port: `src/hldbea/imop.py`

| Upstream file | SHA-256 |
|---|---|
| `IMOP3.m` | `979f361ab1f79c26fa829527f046e12fbddb6dbef65f716c9865d07f3b7b7419` |
| `IMOP4.m` | `a3ae14e1ba8e95a939c862ce64e43eb1a680d36aa7c8ffe30f3d21e8597b4531` |
| `IMOP7.m` | `d58b9fd6c80c085ca70dbd649780ed5f654d6272d43c9366dc9cf7076d9de95d` |

The objective formulas and default parameters (`a1=0.05`, `a2=10`, `K=5`)
follow those files. Deterministic Pareto reference sets are generated from the
analytical front definitions and are hashed with the rest of each experiment's
reference geometry. PlatEMO is acknowledged and cited separately as required
by its source header.
