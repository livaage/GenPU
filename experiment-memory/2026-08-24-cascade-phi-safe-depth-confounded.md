# 2026-08-24 — cascade architecture: phi-averaged conditioning is SAFE; depth-dependence is real but confounded by production position

- **Commit / branch**: `e069764` / `flow-response`
- **Job**: 12879196 (`cascade_depth_characterize.py`, shard 0, 8.55M nodes)
- Measurement only. Answers two architecture questions before any cascade model is built.

## Result

**(1) PHI SYMMETRY — safe, concern closed.** Interaction rate and production radius across 12 phi
bins, particles produced inside the tracker volume (vr < 1100 mm):

| species | n | interact-rate relative spread | radius relative spread |
|---|---|---|---|
| gamma | 1,176,581 | **0.0262** | 0.0210 |
| pi+ | 605,613 | 0.0233 | 0.0185 |
| pi- | 614,840 | 0.0222 | 0.0180 |

All ~2%, far below the 5% threshold. ODD's material budget is phi-symmetric enough that
`build_conversion_slice.py`'s `[log_E, eta, vr, vz]` conditioning is NOT missing a variable.
No change needed. (This closes a concern raised earlier the same day.)

**(2) SELF-SIMILARITY — depth-dependent at matched energy, but CONFOUNDED.** Mean daughters per
parent, parents with >=1 daughter:

| species | depth 0 | depth 1 | depth 2 | depth 3 |
|---|---|---|---|---|
| gamma | 1.708 | 1.197 | 1.396 | 1.265 |
| e- | 2.744 | 2.372 | 1.628 | 1.792 |
| pi+ | 4.046 | 2.142 | 1.685 | 1.546 |
| pi- | 4.066 | 2.107 | 1.662 | 1.558 |

It persists within a fixed energy bin (pi+ highest-E quartile: 6.27 at depth 0 vs 3.63 at depth 3).

**The confound**: depth correlates with production position, and `vr`/`vz` are ALREADY conditioning
variables. So this shows depth predicts multiplicity GIVEN ENERGY — not that depth adds information
beyond the existing inputs. The displacement column points at position doing the work: depth-1
particles sit a median **2,660 mm** from their parent (beamline -> first interaction) while depth-2
and 3 sit at **348 mm** and **144 mm** (local shower development). Two physically distinct regimes
that vr/vz already separate.

## Verdict — phi CLOSED; self-similarity UNRESOLVED (test was not sufficient)

Do not conclude "each level needs its own model" from this. The measurement does not control for
production position, which is the obvious mediator.

## Next

- **Re-bin by (E, vr) instead of E alone.** If the depth dependence vanishes, ONE recursive model
  conditioned on `[log_E, eta, vr, vz, pdg]` serves all levels. If it survives, add an explicit
  depth input — still far cheaper than per-level models. Cheap, no training.
- Note 91.5% of nodes have a resolvable parent; the remainder are primaries and a small unresolved
  tail. Depth histogram (shard 0): 2.24M / 2.43M / 2.18M / 1.03M / 0.46M / 0.15M / 0.06M.
