# Post-2024 comparator provenance

This record defines the exact upstream material used for the two post-2024
comparators requested by Reviewer 1. It is an implementation audit, not a
performance claim.

## Frozen upstream source

- Repository: `https://github.com/BIMK/PlatEMO`
- Commit: `d25e65d1ffba58dbf4d7e1b5259786187d12968a`
- Scope: real-valued, unconstrained benchmark problems supported by this study
- Local adapters: `src/hldbea/recent_algorithms.py`
- Shared protocol: the same seed, population, exact function-evaluation ledger,
  problem implementation, metric code, and artifact validator as every other
  comparator

The adapters are source-faithful Python ports of the audited MATLAB routines.
They are not claimed to be bitwise MATLAB reproductions: MATLAB/Octave is not
available in the execution environment, so cross-language numerical parity has
not been established. Helper-level formula tests, deterministic registry tests,
and end-to-end exact-budget smoke tests are mandatory before either comparator
is admitted to a confirmatory manifest.

## MaOEA-HAP

- Paper: X. Yue, W. Wen, Y. Jiang, Y. Tian, and H. Peng, “A hyper-curvature
  balanced indicator and adaptive phase exploration co-driven evolutionary
  algorithm for many-objective optimization,” *Swarm and Evolutionary
  Computation*, vol. 102, article 102313, 2026.
- DOI: `10.1016/j.swevo.2026.102313`
- Upstream directory:
  `PlatEMO/Algorithms/Multi-objective optimization/MaOEA-HAP`

| Upstream file | SHA-256 |
|---|---|
| `MaOEAHAP.m` | `ab4ef4dd9b5cfbb7e02250f055ef9601e8d95e5da3bf654d6ad9043de909b493` |
| `MatingSelection.m` | `a58dac60c8d5a462450e920be95d2397b075dc7a19fd6c2710b0c2da0180b8dc` |
| `NDQSort.m` | `1f5d213b2f951bcd2983bee3db31706c3744b5390944e2266d7a1258c66b3af5` |
| `UpdateCA.m` | `625aaf616ab86e1e53b265423ef0035244c7b801a56646498f55e00384b5b14f` |
| `CrowdDistance.m` | `c03ae2b736e6159a8ec50342526368763d2f608ec6b33d8c6711dc42552dd3e4` |
| `Shape_Estimate.m` | `8c33f63802b88a6f4c66bbe258beca9cd4684e9580a2776c2821f54d86e787e6` |
| `flog.m` | `01055193cb489aa678c2a45b94416a26f898ee412359e4bf8a0a396a0e33d519` |
| `UpdateDA.m` | `4130c7bc9ed84b90d6c5308c50db426243f7b9d10def18e6b475e089a55a65bd` |
| `AssociateAngle.m` | `2c8c1cac29046d43ae67a778524b93fb56b3aba34a9d8e2259788a32372e72e4` |

## FDSEA

- Paper: W. Wang, Z. Tan, Y. Wang, and W. Zhang, “Enhancing the scalability
  of large-scale multi-objective evolutionary algorithm through frequency
  domain search,” *Swarm and Evolutionary Computation*, vol. 106, article
  102423, 2026.
- DOI: `10.1016/j.swevo.2026.102423`
- Upstream directory:
  `PlatEMO/Algorithms/Multi-objective optimization/FDSEA`

| Upstream file | SHA-256 |
|---|---|
| `FDSEA.m` | `38e8c11b04a412640e884a5fe0d1f21c5f36bf0afe649853cf4637277da73be9` |
| `EnvironmentalSelection.m` | `b705228fac8f93d30fb7faad6f82514173069aba17d6dd04afc8e4e21b511f43` |
| `FRGA.m` | `d6cc24ea3b39be18c0c31380014953c720e4103801d614574f59d1fcf087f87f` |
| `AutoUpdate.m` | `09c9f1b1b42667bb2be73bcdbb1a4f7ed10053c4ed2f3440c4b7bbf10f45c114` |
| `Exchange.m` | `91a21e00f4c9b95f7e2268e9a0d0bee725ecc02d5994765dffe5a554321e5276` |
| `OperatorHybrid.m` | `433ce5ae31465babbd9d166c96e66ce9b03091e10798b17435bac8e6e2de3e1f` |
| `FRDE.m` | `0fb7c8df27612dd83147410d1ceb17c6847d1e206e736137761d488ce3dd67e2` |
| `Cal_Dec.m` | `ac1a94ac1b121bb4571e2ebe0c2f11d871ba519f1a8713b2bdc7de52a48a3f14` |
| `Cal_MP.m` | `3deeb360a5af596fbfae9a2e91820a28c65a0ff2e2ec9bf1735d104326db131b` |

## Required acknowledgement

PlatEMO's source header permits research use and requires acknowledgement of
the platform and citation of Y. Tian, R. Cheng, X. Zhang, and Y. Jin,
“PlatEMO: A MATLAB platform for evolutionary multi-objective optimization,”
*IEEE Computational Intelligence Magazine*, 12(4):73–87, 2017.
