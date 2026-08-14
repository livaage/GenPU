# 2026-08-13 — width-in-global: FAILED, and it relocated the defect to the shower CORE

- **Date:** 2026-08-13 · **commit:** 9e0bbc2 (working tree) · **branch:** flow-response
- Jobs: 12326562 (e±), 12326563 (pion). Follows
  [capacity + width gate](2026-08-13-calo-capacity-test-and-width-gate.md).

## Hypothesis
`width_std` was a 0.90-0.96 single-feature discriminator and generated widths were compressed to
0.79x the real spread, against a real q10-q90 range of ~50x. Cells are i.i.d. given (cond, global)
and the particle features cannot predict compactness, so: store points in units of each shower's own
RMS width, put `log_width` in the GlobalHead (dim 4, quantile-normalised), and scale the sampled
cloud by the sampled width. The direct analogue of the per-shower CORE fix that already worked.

## Change
`build_calo_slice.py --width_norm`; `CaloFlow.width_norm` buffer + rescale in `sample_showers`;
optional `--width_renorm` (project each sampled cloud to exactly unit RMS first). New slices
`electron_wn.npz` (400k/class, width spread 46.9x) and `pion_wn.npz` (102.2x); checkpoints
`electron_wn` (60k steps) / `pion_wn` (40k). Controlled: same data + steps as `electron_v1` and
`pion_qtd_v2`, width_norm the only change. Slice sanity: normalised cloud RMS = 1.0000 exactly.

## Result — worse on every axis, and width did not move AT ALL
| run | GATE10 | gate8 | width_mean AUC | width_std AUC | frac_floor AUC | width W/σ | cell_logE W/σ |
|---|---|---|---|---|---|---|---|
| e- before (`electron_v1`) | 0.971 | **0.773** | 0.762 | 0.908 | **0.616** | 0.171 | **0.017** |
| e- width_norm | 0.990 | 0.969 | 0.763 | 0.903 | 0.853 | 0.169 | 0.081 |
| e+ before | 0.968 | **0.760** | 0.753 | 0.903 | **0.616** | 0.169 | **0.017** |
| e+ width_norm | 0.987 | 0.969 | 0.761 | 0.899 | 0.844 | 0.169 | 0.081 |
| pion before (`pion_qtd_v2`) | 0.993 | **0.809** | 0.945 | 0.963 | **0.666** | 0.320 | **0.020** |
| pion width_norm | 0.993 | 0.838 | 0.943 | 0.962 | 0.693 | 0.317 | 0.035 |

`--width_renorm` on/off was identical to 3 decimals, so the unit-RMS projection is irrelevant here.

## WHY it failed — the metric's "width" is the CORE, not the intrinsic size
The slice normalises the CORE-RELATIVE cloud size; `calo_metrics` measures RMS about the PARTICLE,
which is `sqrt(core² + intrinsic²)`. Measured on the slices (multi-cell showers):

| | median \|core\| | median intrinsic w | core/intrinsic | core's share of width variance |
|---|---|---|---|---|
| e± | 0.302 | 0.026 | **9.1x** | **83%** |
| pion | 1.541 | 0.069 | **20.6x** | **93%** |

So the fix targeted the 6-17% of the variance that was never the problem. Nothing about width could
have improved, and the flat width AUCs confirm it exactly.

The core offset is physical: shower centroids are displaced from the particle direction by magnetic
bending. Charge-antisymmetric shift is clean for pions (core_phi median **+0.365** for q<0 vs
**-0.331/-0.397** for q>0) but is swamped by an IQR of **0.73-1.87** (low-pT charged hadrons curl and
deposit almost anywhere in phi). `corr(|core|, log_pt)` = **-0.382** (pion) — real conditional
structure. Linear `corr(core_phi, charge/pT)` is ~0, i.e. the dependence is NOT linear in charge/pT.
e± show NO charge asymmetry in core_phi (median ~0.00, IQR 0.24) and `corr(|core|,log_pt)` = +0.115.

## Second finding — the three heads SHARE one conditioning MLP, so head changes are not isolated
`width_norm` only changed the GlobalHead's task (4 -> 5 dims, including a std-2.56 `log_width`), yet
the ENERGY marginal degraded badly (cell_logE 0.017 -> 0.081, frac_near_floor AUC 0.616 -> 0.853).
The energy head's own inputs and loss were untouched; the only coupling is `ParticleConditioning`,
shared by all three heads. Hypothesis (untested): a harder global task drags the shared embedding.
This is a plausible contributor to the earlier 2x2 degradations too, and it means **any per-head
experiment is confounded until the trunks are separated** — a cheap ablation worth doing before the
next architecture attempt.

## Verdict
**ABANDONED.** `--width_norm` stays opt-in and unused (default path unchanged; no revert needed —
`electron_wn`/`pion_wn` are separate checkpoints). The references remain `electron_v1` (e- 0.773) and
`pion_qtd_v2` (0.809).

## Next
1. **Target the CORE, not the width.** It is 83-93% of the width variance and the physics is known
   (magnetic bending). Cheapest informative step first: a DIAGNOSTIC comparing real vs generated
   core distributions in (charge x pT) bins — median and IQR — to see whether the GlobalHead misses
   the charge-antisymmetric shift, the pT-dependent spread, or both. Do this BEFORE another retrain.
2. If the conditional is the problem, the fix with precedent in this project is a **physics anchor**:
   predict the expected bend from (charge, pT, calo radius) analytically and model the core RESIDUAL
   — the same move that fixed tracker coherence in v4.
3. **Separate the per-head conditioning trunks** (or weight the global NLL) so head-level experiments
   stop confounding each other.
4. Still open, unchanged: floor-fraction global dim; multi-species capacity (~0.12 for e±);
   incidence head.
