# 2026-08-13 — core context normalisation: pion WIN (small), e± LOSS, mechanism unchanged for both

- **Date:** 2026-08-13 · **commit:** 9e0bbc2 (working tree) · **branch:** flow-response · **job:** 12371088
- Follows [width-in-global FAILED](2026-08-13-calo-width-in-global-FAILED.md).

## Hypothesis
The core carries 83-93% of the "shower width" variance. Its marginal is matched by construction
(pooled quantile transform) but per (charge x pT) bin the model generates only 0.58x (e±) / 0.94x
(pion) of the real core spread — the pooled transform leaves each conditional slice heavy-tailed and
differently scaled. Normalising each context to zero-median / unit-IQR BEFORE the pooled transform
should remove the under-dispersion. Context is TRUTH conditioning (charge, log_pt), exactly known at
generation — the structural difference from the sampled-global conditioning that failed earlier.

## Change
`--ctx_norm` / `--ctx_pt_bins` (default 8 -> 24 contexts = 3 charge signs x 8 pT bins).
`CaloFlow` gains `ctx_pt_edges` / `ctx_loc` / `ctx_scale` / `ctx_mask` buffers, `context_of()`, and
context-aware `std_glob`/`unstd_glob`. The context block runs BEFORE the quantile grids are fit (they
must see context-normalised values). Round-trip verified exact for 99.5% of showers (median error 0,
p99 2e-7); the 0.5% that saturate are the intended bounded-inverse tails. Added
`CaloFlow.from_checkpoint()` since buffers are now shape-dependent (a size mismatch raises even under
strict=False). Checkpoints `electron_ctx` (60k), `pion_ctx` (40k) — same slices/steps as references.

## Result — SPLIT, and the intended mechanism did not move in either case
| run | gate10 | gate8 | width_std AUC | frac_floor AUC | cell_logE W/σ | core spread vs real |
|---|---|---|---|---|---|---|
| pion before (`pion_qtd_v2`) | 0.993 | 0.809 | 0.963 | 0.666 | 0.020 | 0.94x |
| **pion ctx_norm** | **0.980** | **0.797** | **0.837** | **0.630** | **0.017** | 0.93x |
| e- before (`electron_v1`) | 0.971 | **0.773** | 0.908 | **0.616** | **0.017** | 0.58x |
| e- ctx_norm | 0.980 | 0.967 | 0.850 | 0.858 | 0.081 | 0.56x |

- **Pion: the first genuine improvement of the session** — every axis better, including the width gate
  (0.993 -> 0.980) and `width_std` (0.963 -> 0.837, the metric nothing had moved before), with NO
  collateral (cell_logE actually improved 0.020 -> 0.017). Modest, but real and clean.
- **e±: a loss** — gate8 0.773 -> 0.967 with the by-now familiar collateral signature
  (cell_logE -> 0.081, frac_floor -> 0.858), IDENTICAL to the `width_norm` e± run's numbers.
- **The intended mechanism did not move for either**: core spread 0.94 -> 0.93 (pion),
  0.58 -> **0.56** (e±). So the pion's gain came from a better conditional SHAPE, not from fixing
  dispersion, and the e± under-dispersion — the single biggest defect, on the biggest species —
  is untouched by conditioning. **A Gaussian mixture under-disperses heavy-tailed conditionals
  regardless of how its inputs are normalised.** This was the 4th patch to that component's
  inputs/representation; all 4 left the mixture itself in place.

## Second finding, now twice with an identical signature — the shared trunk
Both `width_norm` and `ctx_norm` changed ONLY the GlobalHead's task, and both produced e±
cell_logE **0.081** and frac_floor **0.858** (baseline 0.017 / 0.616) with the energy head's inputs
and loss untouched. The only coupling is the shared `ParticleConditioning` MLP. The pion ctx run
(40k steps, 2.5M showers) did NOT show it while both e± runs (60k steps, 800k showers) did, so the
effect is config-dependent as well. **Every per-head experiment is confounded until the trunks are
separated** — this must be fixed before any further architecture conclusion is trustworthy.

## Verdict
- **KEPT for pions** (`pion_ctx`, gate10 0.980 vs 0.993) — small, clean, the only forward motion today.
- **NOT kept for e±**; `electron_v1` remains the e± reference.
- `--ctx_norm` stays opt-in and species-dependent, which is itself a sign the fix is a patch and not
  the cure.

## Next (reordered by what the evidence now supports)
1. **Separate the per-head conditioning trunks** — removes a confound that has now corrupted two
   experiments with a measurable, repeatable signature. Cheap, and a precondition for reading anything.
2. **Replace GlobalHead's Gaussian mixture with a normalizing flow.** Literature: EVERY CaloClouds
   version (1, 2, 3) uses a normalizing flow "ShowerFlow" for exactly these per-shower globals —
   CaloClouds1's explicitly includes the shower CENTRE OF GRAVITY in x and y, i.e. our core.
   CaloClouds3 slimmed it to 12 affine + 2 spline couplings, so it is cheap. Our 8-component GMM is
   the "lite" in CaloClouds-lite and is the component that has now resisted four patches.
3. **Anchor the core on a helix extrapolation to the calo face** (`helix.py:helix_at_r` already
   exists). The core is a 1.5 m magnetic-bending displacement (pion median |core| 1.54 rad) that we
   are asking a mixture to predict from vertex kinematics. Making it a RESIDUAL from the physics
   prediction is the same frame-change that gave v3 surface-local (0.9996 -> 0.80) and the v4 helix
   anchor their wins — and it uses truth conditioning only, so no exposure-bias risk.
   Cheap decisive pre-check, no training: how much of the core variance does the helix explain?
4. Then the genuinely novel step: condition the calo on the GENERATED TRACK's outer state (captures
   scattering / energy loss / early stopping that a helix cannot), evaluated with real vs generated
   tracks to separate information gain from exposure bias.
