# 2026-08-27 — two-point coherence is a HADRON-only defect; EM needs nothing beyond the profile

- **Commit / branch**: `32d0cab` / `flow-response`
- **Job**: 13033389 (`jobs/calo_coherence.sh`, diagnostic only — no training)
- **New**: `scripts/calo_shower_coherence.py`
- **Ckpts scored**: `electron_v2_s0/checkpoint_060000`, `multispecies_v2_s0/checkpoint_060000`
- **Artifacts**: `plots/calo/metrics/coherence_{ele_dedicated,ms_ele,ms_pion,ms_photon,survey}.{json,png}`
- Follows [the one-point entry](2026-08-27-calo-energy-position-coupling-and-the-floor-that-isnt.md)

## Hypothesis

The one-point run showed real showers carry an energy/position gradient the model emits none of, and
that conditioning `EnergyHead` on each cell's own position would reproduce it. But per-cell
conditioning gives two cells at the same radius i.i.d. draws — the correct average falloff with a
salt-and-pepper realisation, where a real shower has a contiguous hot core. That is the TWO-POINT
function, and it decides how big the fix has to be: concat position into the existing head (cheap)
versus the cells seeing each other (attention / set model, and ultimately gap #1's cell projection,
since "neighbouring cell" is undefined for a continuous point cloud).

## Change

`scripts/calo_shower_coherence.py`. Per species, real vs null vs generated:
- **xi(separation)** two-point energy correlation, log-E standardised WITHIN shower, random pairs
  binned by separation — reported RAW and with each side's own one-point profile removed
- **`<z>` vs rank of distance to the hottest cell** — the plain-language question asked directly
- **centroid pull** |energy-weighted centroid − unweighted| / width, and its depth analogue

## Two tooling corrections, both caught by synthetic arms before real data

1. **Raw xi cannot answer the question.** An arm whose energy depends only on each cell's OWN radius
   — i.e. exactly what per-cell conditioning produces — reproduced xi almost perfectly (+0.603 ->
   -0.471) against a coherent blob's (+0.565 -> -0.385). Nearby cells have similar radii, so a
   radial gradient ALONE manufactures a separation-dependent correlation. Fixed by subtracting each
   side's one-point profile E[z | own radius rank, own depth rank] first.
2. **The profile-preserving null sits on a different baseline.** Shuffling values among cells sharing
   a (radius-rank, depth-rank) bin ACROSS showers destroys the within-shower sum-to-zero constraint
   that standardisation imposes, so it reads ~0.000 where real sits on the -1/(n-1) baseline. It is
   printed for reference but every number below is quoted against the WITHIN-shower null, which is
   baseline-matched. Recalibrated synthetic references: coherent **+0.415**, one-point-only
   **+0.117**, i.e. +0.117 is this estimator's FLOOR, not zero.

## Result — a clean split, replicated on both charges and both checkpoints

| class | pull/null | xi_perp excess | xi_depth excess | verdict |
|---|---|---|---|---|
| mu- / mu+ | **3.38 / 3.33** | +0.079 / +0.075 | +0.088 / +0.085 | track-like, see caveat |
| p | **2.22** | +0.096 | **+0.161** | coherent |
| K+ | **2.02** | +0.066 | **+0.124** | coherent |
| pi+ / pi- | **1.95 / 1.89** | +0.067 / +0.065 | **+0.119 / +0.108** | coherent |
| K- / pbar | 1.65 / 1.54 | +0.055 / +0.049 | +0.103 / +0.106 | coherent |
| n / nbar / K0L | 1.27-1.29 | +0.101 / +0.038 / +0.069 | +0.112 / +0.050 / +0.085 | marginal |
| **pi0 / e- / e+ / gamma** | **1.00 / 0.99 / 0.98 / 0.97** | +0.024 .. +0.035 | +0.020 .. +0.049 | **one-point only** |
| ALL pooled | 1.33 | +0.061 | +0.056 | mixture |

Reference: coherent synthetic **2.10**, one-point synthetic **0.90**; xi floor **+0.117**.

- **EM showers show no coherence beyond the one-point profile.** gamma, e±, pi0 all sit at pull ~1.0
  with xi excess below the floor. Per-cell position conditioning suffices for them.
- **Charged hadrons do.** p, K±, pi±, pbar combine pull 1.5-2.2 with xi_depth excess 0.10-0.16; for
  p and K+ the xi evidence clears the floor INDEPENDENTLY of the centroid pull, which was the
  two-witness condition set before the run.
- **Generated is flat everywhere.** e± dedicated: real xi_res -0.040 vs gen -0.069 vs null -0.071;
  p(floor) and `<z>` profiles identical to the null. The model has no coherence of any kind, as the
  architecture requires.
- **depth pull** is the sharpest real-vs-gen gap: e± real **-0.0349 ± 0.0005** vs null -0.0002 and
  gen +0.0007 (70 sigma); pi± real **+0.364**; p real **+0.403**. Transverse centroid pull for e± is
  real 0.1652 vs null 0.1679 — nothing.

**CAVEAT on muons.** They top the pull ranking but their xi excess stays below the floor. Centroid
pull conflates "energy is spatially coherent" with "the position cloud is anisotropic", and a muon's
cells lie along a track. Their 3.4x is mostly geometry. The hadrons are where both statistics agree.

## Verdict — KEPT; it changes the scope of the fix, not its direction

Coherence is a **continuum** (gamma 0.97 -> mu± 3.38), not a binary, so a hard EM/hadron ROUTER is
the wrong architecture — K0S (1.00) and n (1.27) sit in the gap. A single **PDG-conditioned attention
head** subsumes both: it can learn near-uniform attention for gamma and strong locality for protons,
and with showers capped at 128 cells the n^2 cost is negligible. PDG is truth conditioning, available
before any cell is drawn.

**A coverage argument made here earlier was wrong and is retracted**: "EM is 65% of showers so the
cheap fix covers most of it". Measured on `multispecies_v2_h5`: EM is 65.3% of showers but only
**46.6% of CELLS** (19.41M of 41.65M) because hadron showers are twice the size (pi± 40.9 cells,
K± ~62, pbar 97.8 vs gamma 15.2). Hadrons are 33.0% of showers and **51.6% of cells** — the majority.
The gate features are cell-level aggregates, so cells are the right denominator.

## Next

Sequence the work for ATTRIBUTION, not coverage. The floor fix is species-independent and corrects a
falsified premise; the coherence fix is species-graded and adds capability. Bundling them would be
unreadable — this project's own history (`ctx_norm`, `width_norm`, anchor_cond each helping one
species and detonating another; STATUS.md "every per-head experiment is confounded") is the argument.
Floor + fractions first, decompose, then position + attention, decompose again.
