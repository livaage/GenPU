# 2026-08-24 — depth is REDUNDANT given (E, vr): one recursive cascade model serves every level

- **Commit / branch**: `aedfe35` / `flow-response`
- **Job**: 12879589 (`cascade_depth_evr.py`, shard 0)
- Resolves the question left open by
  [J4a](2026-08-24-cascade-phi-safe-depth-confounded.md), which found depth-dependent multiplicity
  at matched ENERGY but could not separate it from production POSITION.

## Result

**(1) Among depths 1-3, depth is redundant once (E, vr) is controlled.** Median spread of mean
daughter multiplicity across depths, inside each bin:

| species | parents | E only (what J4a measured) | **E x vr** | p90 | verdict |
|---|---|---|---|---|---|
| gamma | 740,779 | 1.036 | **1.024** | 1.048 | redundant |
| e- | 520,444 | 1.204 | **1.173** | 1.301 | redundant |
| e+ | 465,203 | 1.227 | **1.162** | 1.339 | redundant |
| pi+ | 346,739 | 1.241 | **1.169** | 1.266 | redundant |
| pi- | 347,245 | 1.231 | **1.157** | 1.235 | redundant |

**(2) Depth 0 needed no test — `vr` separates it by construction.** Production radius by depth:

| depth | n | vr p5 | p50 | p95 | frac vr < 1 mm |
|---|---|---|---|---|---|
| 0 | 2,244,523 | 0.0 | 0.0 | 9.9 | **0.929** |
| 1 | 2,433,070 | 24.0 | 404.4 | 1280.5 | 0.002 |
| 2 | 2,175,183 | 28.6 | 438.9 | 1276.0 | 0.001 |
| 3 | 1,700,572 | 116.4 | 428.4 | 1259.4 | 0.000 |

Primaries sit at the beamline and everything deeper at vr ~ 400 mm — nearly disjoint, so the vr
input already tells the model which regime it is in. This also explains J4a's large depth-0 -> 1
multiplicity drop (pion 4.05 -> 2.14): it is the primary/secondary regime change, which vr encodes.

## Verdict — ARCHITECTURE SETTLED for the cascade generator

**One recursive model conditioned on `[log_E, eta, vr, vz, pdg]`, applied ~3-4 times, serves every
level.** No depth input, no per-level models. Combined with the earlier findings this closes the
cascade architecture questions:
- depth is shallow (3 levels = 96.5% of tracker hits, 4 = 99%)
- each level is fully batchable, so sequential depth ~4 regardless of event size
- phi-averaged conditioning is safe (~2% material modulation across phi)
- depth needs no explicit input

## Next

- The remaining cascade unknown is **error compounding** across levels when a generated particle
  becomes the next level's conditioning — the same failure mode the tracker head hit (v3 needed
  scheduled sampling). That needs a trained generic parent->daughters model, i.e. J4b.
- Caveat on scope: multiplicity is the only target tested for redundancy. Daughter KINEMATICS
  (energy sharing, opening angle) were not, and should be checked the same way before the model is
  trained on pooled levels.
