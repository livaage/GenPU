# A-prime: cross-attention reproduces hadron coherence and NN spacing, but collides 4x MORE; gate barely moves

**Date** 2026-09-28 (rerun of 2026-09-16 job 13978193, reproduces it) · **Commit** `2604c3f` · **Branch** `flow-response`
**Jobs** train 13977754 (`jobs/calo_aprime_train.sh`, 2 arms, 60k steps, seed 0) · eval 14615599 (baseline) +
14616716 (cross, selfonly) via `jobs/calo_aprime_eval.sh` · held-out shard 5 · all 17 classes · `--max_cells 4096 --snap_cells`
**Ckpts** `multispecies_v2_s0` (baseline), `ms_aprime_cross_s0`, `ms_aprime_selfonly_s0`, all `checkpoint_060000`
Follows [NN-spacing entry](2026-09-16-calo-nn-spacing-generated-is-SPARSER.md).

## Hypothesis (pre-registered in the eval script header, written before any A-prime model existed)
Three measurements said a shower's cells are generated independently (no two-point coherence, collisions
0.036 vs 0.0003, NN spacing 2.1x real). If that is the cause, letting cells attend to each other should:
collisions self-only ~0.036 / cross -> 0.0003; NN ratio self-only ~2.1 / cross -> 1; hadron coherence
self-only flat / cross -> real; copula half cross < self-only; zero-suppressed < 1% on both new arms.

## Change
`PointCFM(attn_layers=3)` set attention over each shower's cells + `CaloFlow(joint_energy=True)` (log-E as a
4th flow coordinate, sub-threshold cells dropped, no EnergyHead). Two arms differing ONLY in the attention
mask: **cross** (cells see each other) vs **self-only** (mask on the diagonal, matched control). Baseline is
the old per-cell model. NB baseline -> self-only changes TWO things (joint energy + whole-shower batching),
so only self-only -> cross isolates attention.

## Result

| | baseline | selfonly | cross | real |
|---|---|---|---|---|
| `event_gate_auc` (10 feat, snap) | 0.9451 | **0.9045** | 0.9361 | 0.5 |
| decomposition: marginals only | 0.865 | 0.877 | **0.963** | |
| decomposition: copula only | 0.787 | 0.612 | **0.586** | |
| collision rate (200k showers) | 0.0361 | 0.0268 | **0.1302** | 0.0003 |
| collision, core r/width < 0.25 | 0.117 | 0.095 | **0.319** | |
| merged under `--snap_cells` | 3.6% | 2.6% | **13.1%** | |
| NN spacing gen/real | 2.06 | 2.30 | **1.08** | 1 |
| pion centroid pull / null (0.186) | 0.87 | 1.29 | **1.99** | 1.92 |
| pion depth pull | +0.001 | +0.286 | **+0.406** | +0.364 |
| e± depth pull | -0.000 | **-0.037** | -0.020 | -0.035 |
| zero-suppressed cells | n/a | 242k (0.58%) | 175k (0.42%) | |
| µs / particle (50 ODE steps) | 35.6 | 209 | 205 | |

Worst gate features: **cross** `logE_p90` 0.844, `logE_mean` 0.838, `cells_per_src` 0.695 (LOO: `cells_per_src`
+0.013). **selfonly** `frac_near_floor` 0.780 (near-floor band gen 0.0236 vs real 0.0295), `width_std` 0.673.
**baseline** `logE_std` 0.760, `logE_p90` 0.716.

Reproducibility: the 2026-09-16 run gave 0.9455 / 0.9361 / 0.9045 and collisions 0.0354 / 0.1306 / 0.0270.

Training: val flow loss flat from ~20k (cross 0.757 @30k -> 0.727 @60k, noisy); val gnll still improving
(-5.24 -> -5.43). Not a gate-level convergence check.

## Pre-registration scorecard
- NN spacing cross -> 1: **PASS** (1.08).
- Hadron coherence cross -> real: **PASS** (1.99 vs 1.92; depth pull 0.41 vs 0.36). First generator ever to show it.
- Copula cross < self-only: **PASS, marginally** (0.586 vs 0.612).
- Zero-suppressed < 1%: **PASS** on both. Joint energy is not broken.
- Collisions cross -> 0.0003: **FAILED, in the wrong direction** (0.036 -> 0.130).
- Self-only coherence flat: **FAILED partly** (1.29, not ~1.0). Cells that cannot see each other still pick
  up correlation. Candidates: shared per-shower flow time t, joint energy/position within a cell. Unexplained.
- Speed 2-5x: **slightly over** (~5.8x).

## Reading
Attention fixed WHERE cells go (spacing, coherence, joint structure) but not that they OVERLAP: cells now
cluster at real spacing with no exclusion, so they land on the same channel 4x more than before. The cross
gate's losses are exactly what heavy merging would produce: fewer cells per source, more energy per cell
(`cells_per_src`, `logE_mean`, `logE_p90`). **This is a hypothesis, not yet tested.** The defect has moved
from the copula (joint) to the marginals. Self-only is the best composite gate, but through a different
defect (the floor band, from zero-suppression).

## Verdict
**KEPT (opt-in), not promoted.** Attention delivers the mechanism it was built for, but the composite gate
does not beat self-only, and the collision prediction failed. Single seed per arm. The ±0.002 in the
decomposition is CLASSIFIER-seed spread only; MODEL-seed spread has been as large as 0.11 on this gate
(`--floor_n_buckets` replicate, 0.72 -> 0.83), so self-only vs cross (0.905 vs 0.936) is NOT established
until a second model seed exists.

## Next
1. Diagnostic: gate **without** `--snap_cells` on all three arms (existing checkpoints). If cross's marginal
   AUCs (`logE_p90`, `logE_mean`, `cells_per_src`) collapse toward self-only's, the merge explains cross's
   gate. Per the 2026-09-15 open thread this is a DIAGNOSTIC, not a fix: merging is correct, the error is
   placement.
2. If confirmed: an exclusion mechanism (repulsion term, or generating on the cell lattice).
3. Explain self-only's 1.29 coherence before crediting all of cross's coherence to attention.
4. Second seed of self-only vs cross.
