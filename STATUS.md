# STATUS — GenPU tracker

Rolling state. Append-only detail lives in `experiment-memory/`. See also `TRACKER_GATE_FINDINGS.md`.

_Last updated: 2026-08-14_

## Current best
- **Tracker: v3 surface-local + scheduled sampling BEATS the v1 deliverable.** Honest full-event
  gate, all charged species [e±,π±,p,μ±], shards 0-2 (like-for-like vs the 0.77): **AUC 0.61** vs v1
  **0.77**. Ckpt `checkpoints/tracker/surface_ms_v3_ss/checkpoint_050000`; count
  `count_head_d0_selfnorm.pt`. See
  [2026-08-10 all-species](experiment-memory/2026-08-10-all-species-v3-ss-beats-deliverable.md).
- Pion-only honest gate **0.596** (surface_pion_v3_ss). Progression (honest): v1 0.77 → v3-ms 0.61.
- **Count head**: self-normalizing fix (`count_head_d0_selfnorm.pt`); "undercount" was a norm mismatch.
- **Documented v1 deliverable** (separate setup): multispecies honest full-event gate **0.77** (base-8),
  ckpt `multispecies_vertex_512bin/checkpoint_070000`. Not yet re-contested by v3 (v3 is pion-only so far).

## Active hypothesis
The v1 outward drift is a representation artifact (unbounded ±60 mm layer residual); surface-local
(v3) fixed the marginals. The **residual module-selection drift** in v3 is now **confirmed exposure
bias** (teacher-force test 2026-08-10: teacher-forced ≡ real to <1mm at all steps; free-running
compounds outward). Fix under test: **scheduled sampling** (train on own history). Seed is fine —
**vertex dropped as a lever** (the "too-inner seed" was a population artifact).

## v4 helix anchor — CONCEPT VALIDATED (coherence fixed), marginal trade-off
- **Coherence FIXED**: v4 gen `phi_r_resid`/`z_r_resid` now spike at 0 matching real (`plots/tracker/
  v4_eval/track_feats.png`); v3 was ~10x broader. The helix anchor reproduces per-track coherence.
- **Marginal gate regressed**: v4 honest pion gate **0.78** vs v3-SS 0.596, from the 18.5% clipped
  tail (low-pT curly outer hits) inflating r_std. So: v3-SS wins the marginal gate, v4 wins track
  realism. Trade-off, not a failure.
- Ckpt `helix_pion_v4/checkpoint_050000`. See
  [2026-08-10 helix](experiment-memory/2026-08-10-helix-anchor-prebuild.md).

## Calo (audits: [2026-08-12](experiment-memory/2026-08-12-calo-audit-and-metrics.md),
## [census+energy](experiment-memory/2026-08-13-calo-species-census-and-energy-audit.md),
## [2x2+multispecies](experiment-memory/2026-08-13-calo-energy-fix-2x2-and-multispecies.md))
- **CaloClouds-lite flow model** (`src/genpu/flow/calo_flow.py`): global mixture + point flow-matching
  (50-step ODE) + energy mixture. Conditions on the 7 CONT_FEATURES + PDG embedding.
- **BEST CALO HEADS ARE NOW THE PHASE 1 ANCHORED ONES** (`pion_anchor`, `electron_anchor`, table
  below): pion width gate **0.9833**, e− **0.9639** / e+ **0.9578**, and the e± core mechanism fixed.
  The numbers in the rest of this section are the pre-anchor references they beat.
- **Photon** (`photon_qtd_v1`): held-out gate **0.557**, energy faithful, lateral `shower_width` the
  weak spot — **not yet re-run with the anchor** (Phase 0b says 2.82x, pure vertex displacement).
  **Pion** (`pion_qtd_v2`): **0.809**; `pion_ctx` (core context normalisation) was a small
  clean win — width gate 0.993→**0.980**, `width_std` AUC 0.963→0.837, no collateral — but
  species-dependent (LOST on e±), so a patch, not the cure; superseded by the anchor. See
  [ctx-norm split result](experiment-memory/2026-08-13-calo-ctx-norm-split-result.md).
- **Physical bounds now on by default in `sample_showers`** (per-cell `logE_max`, per-species
  `logE_max_pdg`, and `E_reco <= E_true`): worst ⟨E_reco/E_true⟩ 240x (photon) / 657701x (pion) →
  **1.00**, gate unchanged (0.558 / 0.809), ~1e-5 of showers touched. Real showers never exceed E_true.
- **SPECIES CENSUS — e± dominate the calo: 48% of deposited energy, 57% of cells** (photon is 2.3%).
  π± 29%/27%, p 13%/7%, rest ~8%. e±+π±+p+γ = 92% of energy.
- **Multi-species head** (`multispecies_v1`, one PDG-conditioned model, all 17 classes, 60k steps):
  per-species held-out gates γ 0.681, π 0.870, e- 0.888, e+ 0.887, p 0.749, rest 0.626. Covers
  everything; shared capacity costs per-species fidelity (γ 0.557→0.681) — capacity is the open question.

## Calo metric CHANGED 2026-08-13 — the gate now has a width variant (read this first)
`calo_metrics.py` reports **two** AUCs: `event_gate_auc` (the historical 8 energy/multiplicity
features, kept for comparability) and `event_gate_auc_width` (+ per-shower `width_mean`/`width_std`).
The 8-feature gate is **blind to lateral shape and therefore flattering**: photon 0.558 → **0.805**,
pion 0.809 → **0.993**, e± 0.89 → **0.98**. Quote the width gate for anything forward-looking.

## Calo Phase 0 DONE — helix-core gate is GO on every species (2026-08-13)
Calo geometry derived (`calo_geometry.json`): **front face barrel r=1259 mm, endcap |z|=3212 mm,
eta transition 1.666; the ENDCAPS carry 83% of deposited energy** (so the anchor extrapolates to a
plane for most showers). Core measured against a truth-helix extrapolation to that face:

| species | \|core\| med particle→helix | **phi tightening** |
|---|---|---|
| proton | 1.803 → 0.119 | **25.5x** |
| e± | 0.297 → 0.043 | **11.2x** |
| pion | 1.543 → 0.377 | **4.6x** |
| photon (neutral control) | → 0.009 | 2.9x |

Gain scales as 1/pT exactly as bending predicts. **The e± "no charge asymmetry" puzzle is resolved**
— soft e± bend through >pi and `wrap_pi` symmetrised the median; the brem hypothesis is NOT needed.
Caveat: the helix fixes the BULK not the tail (e± 11.2x robust vs 2.2x on plain std) — the anchored
core is sharply peaked with outliers, so keep the bounded quantile inverse.
See [Phase 0 entry](experiment-memory/2026-08-13-phase0-calo-geometry-and-helix-core-GO.md).

> **SUPERSEDED 2026-08-14 — the table above was measured through a SIGN BUG.**
> `helix.py:helix_at_z` swept the endcap the wrong way (`-sign*sign(q)` vs `helix_at_r`'s
> `sign*sign(q)`); against 80k real endcap tracker hits, median |Δφ| to the true hit was **0.633 rad**
> with the old sign and **0.016 rad** fixed. The endcaps carry 83% of the energy, so every number
> above is wrong. `helix_at_z` is used only by the calo anchor, so no tracker result is affected.
> See [2026-08-14 entry](experiment-memory/2026-08-14-helix-endcap-sign-bug-and-phase1-anchor.md).

### Corrected Phase 0b (job 12374055, 400k showers/species) — φ tightening
The old validity test silently accepted arc-clipped curlers; with that closed, the face is reached
by only ~11-14% (barrel) + 12-26% (endcap) of showers and the REST are curlers anchored at the helix
turning point. So report per branch — a pooled number mixes two different physics claims:

| species | pooled | barrel | endcap | **turning pt** | branch mix (bar/end/turn) |
|---|---|---|---|---|---|
| pion | 3.67x | **18.84x** | **15.17x** | 2.50x | 0.11 / 0.26 / 0.63 |
| proton | 2.80x | 9.66x | **23.65x** | 2.45x | 0.14 / 0.12 / 0.74 |
| e± | **1.86x** | 2.80x | 7.46x | 1.98x | 0.12 / 0.16 / 0.72 |
| photon (neutral) | 2.82x | 2.82x | — | — | 1.00 / — / — |

Where the particle actually reaches the calo the anchor is decisive (15-24x). The curler branch is
worth a steady ~2.5x on every charged species. η tightening is larger than φ throughout
(pion 5.59x, photon 7.61x).

**e± is the outlier and now reads MARGINAL (1.86x)** — and its gain DEGRADES with pT
(4.52x → 2.14 → 1.36 → 1.08 → 1.04 → **0.43x**, i.e. the anchor actively HURTS the stiffest bin).
This is the plan's e± caveat resolving toward branch (ii), **bremsstrahlung**: the deposit comes from
photons radiated early that travel STRAIGHT from the radiation point, so the electron's own bent
trajectory is the wrong predictor and gets wronger as the helix bends further from the initial
direction. If so the e± core needs the **tracker** (Phase 4 — which knows where the electron went and
where it stopped), not a truth helix. Note this contradicts the 2026-08-13 reading that the wrap_pi
explanation had resolved the puzzle; that was measured through the sign bug.

## Calo Phase 1 DONE — truth-helix anchored core is KEPT (2026-08-14)
[Result entry](experiment-memory/2026-08-14-phase1-helix-anchored-core-RESULT.md) · ckpts
`pion_anchor/checkpoint_040000`, `electron_anchor/checkpoint_060000` · shared trunk, recipes and step
counts identical to the references so the core frame is the only variable.

| | e− ref | **e− anch** | e+ ref | **e+ anch** | pion ref | **pion anch** |
|---|---|---|---|---|---|---|
| gate (8-feat) | 0.7729 | **0.7456** | 0.7597 | **0.7488** | 0.8092 | **0.8055** |
| gate (width) | 0.9713 | **0.9639** | 0.9675 | **0.9578** | 0.9931 | **0.9833** |
| `width_std` | 0.9085 | **0.8952** | 0.9029 | **0.8905** | 0.9627 | **0.9425** |
| `width_mean` | 0.7617 | **0.7330** | 0.7528 | **0.7270** | 0.9449 | **0.9081** |

Every metric improves, **no energy-head collateral** (`cell_logE` W/σ 0.015 e− / 0.022 pion,
`frac_near_floor` 0.60 / 0.62). That is the contrast with the earlier patches: `ctx_norm` bought a
better pion `width_std` (0.837) but detonated the e± 8-feature gate (0.773→**0.967**). The anchor is
the first core change that helps every species at once.

**MECHANISM — read it in the RESIDUAL frame** (what the mixture fits; the physical frame's anchor
part is exact by construction, and `calo_core_diag.py` now prints both):
- **e± FIXED: 0.585 → 0.99** per (charge × pT) bin (min 0.97, max 1.01). The defect four patches
  failed to move is gone.
- **Pion NOT fixed: 0.93** (baseline 0.935, ctx 0.931). A 3.6x tighter target left the *relative*
  under-dispersion identical — scale-free, so it is the **density model** (Phase 2 ShowerFlow), not
  the frame.
- **NEW defect, both species — the GlobalHead is blind to the anchor BRANCH.** Real residual scale
  differs 10x (pion 0.057 face vs 0.598 curler) / 15x (e± 0.010 vs 0.132), and the conditioning
  cannot express a hard threshold in (2R vs r_calo). So face showers come out too wide
  (pion 1.13, e± 1.17-1.48) and curlers too narrow (0.92 / 0.85).
- **e± physical frame over-disperses (1.29) and that is the ANCHOR's fault, not the model's**
  (mixture is at 0.99). Overshoot is monotone in pT — 0.96 → **1.93** — matching Phase 0b's
  4.52x → **0.43x** gain inversion. See the brem note above.

## Calo Phase 1 — build notes (2026-08-14)
Core stored as a **residual from the helix prediction**; generation adds the anchor back from truth
conditioning (`src/genpu/calo_geom.py` is the single implementation, shared by the slice builder,
`calo_metrics.py` and the probe, so the training and generation frames cannot drift apart).
`build_calo_slice.py --core_anchor helix`; the model carries a `core_anchored` buffer and **raises**
if an anchored checkpoint is sampled without an anchor.
- **Three anchor branches**, reported per run: barrel / endcap face crossing, else the helix
  **turning point**. The face is honestly reached by only **37% of pion / 28% of e±** showers — the
  rest are soft secondaries (median pT 0.27 GeV, born at vr ≈ 420 mm) that curl back first with their
  energy booked to the parent. (The old code's endcap fallback silently accepted arc-clipped curlers,
  e.g. anchoring a 0.2 GeV track at η+2.5.) Turning point on that population: σ(core_φ) **2.4x**
  tighter (pion/proton), 2.0x (e±); σ(core_η) 4.6x.
- Pion slice, all branches: median |core| **1.64 → 0.44**, σ(φ) 3.63x, σ(η) 5.54x; per branch
  barrel **18.4x**, endcap **14.5x**, turning point 2.58x.
- Bonus: the anchored frame fixes a latent ±π-straddling bug in the old core (3.4% of showers, where
  `mean(wrap(cell − particle))` is ill-defined) — point-cloud `d_phi` std 0.306 → 0.273.
- Ran **shared trunk**, so the comparison against the shared-trunk references `pion_qtd_v2` /
  `electron_v1` is like-for-like. This deliberately does not take Phase 0c's "separate trunks for
  pion/e±" recommendation — that is a second, independent lever and mixing them would break
  attribution. **Combining them is now an open run.**

## Calo Phase 0c DONE — separate per-head trunks (`--separate_trunks`)
Gradient isolation verified (glob loss → 0.0000 gradient in the point/energy trunks vs 1.4993 into
the shared one); +10.6k params (~4%). Effect tracks DATA VOLUME: pion (39M cells) gate8 0.809→**0.797**,
e- (8.5M) 0.773→**0.739**, e+ 0.760→**0.741**, but photon (4.2M) 0.558→0.679 **regressed**.
Step-matched 120k test settled it: `cell_logE` recovered (0.027→0.016) but the gate did NOT
(0.679→0.680) at essentially identical likelihood — **the photon regression is real, not
undertraining**. Residual is `frac_near_floor` (0.610 vs 0.521). Mechanism: floor fraction depends on
MULTIPLICITY (the GlobalHead's variable), and the shared trunk let the energy head see it implicitly;
species ordering matches cells/shower exactly (photon ~4 hurt, e± ~9 mild, pion ~15 helped).
**Use separate trunks for pion / e± / multi-species; keep the photon shared.** Untested idea from the
mechanism: with separate trunks, condition the energy head on `log_n` ONLY (supplies multiplicity
explicitly). See [0c entry](experiment-memory/2026-08-13-phase0c-separate-trunks.md).

## Calo next steps (evidence-ordered)
1. ~~Separate the per-head conditioning trunks~~ — DONE (above), photon step-test pending. Changing ONLY the GlobalHead task degraded the
   energy head twice with an identical signature (e± cell_logE 0.017→**0.081**, frac_floor
   0.616→**0.858**), via the shared `ParticleConditioning` MLP. Every per-head experiment is
   confounded until this is fixed.
1b. ~~Feed the anchor to the GlobalHead~~ / ~~straight-line & pT-split anchor for e±~~ — BOTH FAILED
   the gate on a shared trunk, see "Ruled out". Anchor conditioning fixed its mechanism target
   (1.003) but leaked into the energy head. **Live follow-up: anchor_cond + `--separate_trunks`.**
2. **GlobalHead mixture → normalizing flow.** Now has a sharp isolated target: the PION residual is
   under-dispersed 0.93 in a frame where e± sits at 0.99, so it is the pion's heavy-tailed
   conditional the mixture cannot fit, not the location. (e± core spread is no longer stuck —
   Phase 1 fixed it.) Literature: every
   CaloClouds version uses a flow ("ShowerFlow") for exactly these globals, CaloClouds1's including
   the shower centre of gravity = our core; CaloClouds3 slims it to 12 affine + 2 spline couplings.
3. ~~**Helix-anchor the core**~~ — BUILT 2026-08-14, training in flight (Phase 1 section above).
   Residual from the physics prediction at the calo face rather than a mixture prediction from
   vertex kinematics; same frame-change class as v3 surface-local and the v4 helix anchor. Truth
   conditioning only. Pre-check answered: 16.5x (pion) φ tightening on the face-reaching population,
   2.4x on the curler population via the turning-point branch.
4. **Then the novel step — condition the calo on the GENERATED TRACK's outer state** (scattering,
   energy loss, early stopping a helix can't know). Evaluate with real vs generated tracks to
   separate information gain from exposure bias. Not found in the calo literature (which generates
   single-particle showers conditioned on truth); would also make the incidence problem physical
   (whether a particle deposits ≈ whether its track survives the tracker).

## Active calo hypothesis — SUPERSEDED for e± by Phase 1 (2026-08-14)
The core-is-the-defect diagnosis below stands, and the helix anchor acted on it: **e± per-bin core
spread 0.585 → 0.99 (fixed)**, pion still 0.93. What remains is (a) the pion's heavy-tailed
conditional → Phase 2, (b) branch-scale blindness → feed the anchor to the GlobalHead, (c) stiff-e±
brem → straight-line anchor / Phase 4.

## Prior calo hypothesis — the "width" defect is really the shower CORE (magnetic bending)
`width_std` is a near-perfect discriminator (pion **0.963**, e± **0.90**, photon 0.689), but the
metric's width is RMS about the PARTICLE = `sqrt(core² + intrinsic²)`, and the core dominates:
median |core|/intrinsic **9.1x** (e±) / **20.6x** (pion), i.e. the core is **83% / 93%** of the width
variance. Normalising the intrinsic size therefore did nothing (width AUCs flat to 3 decimals) —
see [width-in-global FAILED](experiment-memory/2026-08-13-calo-width-in-global-FAILED.md).
The core is physical (bending): pion core_phi median **+0.365** (q<0) vs **-0.331** (q>0), but with
IQR **0.73-1.87**, and `corr(|core|,log_pt) = -0.382`; the dependence is NOT linear in charge/pT.
**Next: diagnose real-vs-generated core in (charge x pT) bins BEFORE retraining**, then a physics
anchor on the expected bend (the move that fixed tracker coherence in v4).
**Also: the three heads share ONE conditioning MLP** — width_norm changed only the GlobalHead task
yet degraded the energy marginal (cell_logE 0.017→0.081). Separate the trunks before trusting any
further per-head experiment.

## Secondary calo hypothesis — floor structure
`frac_near_floor` is the top single-feature discriminator for **4 of the 5 failing cases**, and is
benign for the one that works: single-species photon **0.510** (gate 0.558) vs pion **0.666** (0.809),
ms-electron **0.702** (0.888), ms-pion **0.650** (0.870), ms-gamma **0.622** (0.681). Second is
`logE_max`/`logE_std`/`logE_mean` (≤0.61). Clamping the energy tail by 5 orders of magnitude moved the
gate 0.000, and per-species cell clamps moved ms-gamma by 0.0002 — so neither is the lever.
**EXCEPTION — proton** (ms 0.749): led by `logE_p90` **0.570**, with `frac_near_floor` only 0.528. It
deposits a median 3 cells (vs 9 for e±/π±), too few for the floor fraction to dominate a per-event
statistic. So the floor fix should move e±/π±/γ and NOT the proton — its defect is the upper energy
percentiles, and the proton also has the worst `shower_width` (0.332), which the gate cannot see.
Next: model the per-shower floor-cell fraction as a GlobalHead dim and drive the floor Bernoulli from
the sampled value — a shower-level latent that does NOT make continuous energy a sharp function of a
noisy input (the failure mode that killed the 2x2 variants).
Also: the gate's 8 features are all energy/multiplicity — **it is blind to lateral width**, the
pion/proton weak spot (W/σ 0.32/0.33 vs photon 0.11). Needs a width feature to be trustworthy.

## Publication readiness (fast-sim framing; NO Stage-1 needed — condition on real/Pythia particles)
- Metric suite (`tracker_metrics.py`, `calo_metrics.py`) on HELD-OUT shard 5: gate + per-observable
  Wasserstein + (calo) energy response + speed. Results (all held-out): tracker 0.617, calo-photon
  0.557, calo-pion 0.809. Speeds: tracker 191 µs, calo-photon 14.5 µs, calo-pion 27.8 µs/particle.
- DONE: pion calo retrained (`pion_qtd_v2`) + gated. TODO: fix calo energy-head tail (pion) + core on
  photon; median-based response metric; memorization (NN) both; reco-level (ACTS) — the big lift.
- Reproducibility fix needed: `preprocessing.py` emits 6-col tracker hits vs 5-col training shards.
- **NOT committed to git** — all session work (v3/v4/helix/calo-metrics scripts+models, docs) is in
  the working tree only. Needs a commit before it's truly safe/reproducible.

## Open threads
- **Turning-point anchor is a proxy, not the physics**: 63% of pion / 72% of e± calo showers are
  energy booked to a parent that never reached the calo (soft, born deep in the tracker). That is
  exactly the population **Phase 4** (track-conditioned calo) should own — the generated track knows
  where the particle actually stopped. Revisit the branch there.
- **Fix v4's 18.5% tail** (low-pT curly outer hits) to get coherence AND marginals: hybrid reference
  (layer-mean fallback where the arc-length march is unreliable) or nonlinear deviation bins.
- **Reco-level eval**: does coherence actually matter downstream? Decides v4 (coherent) vs v3-ms
  (0.61 gate, beats 0.77) as the shippable tracker.
- **v3-ms training extension**: cheap probe showed more steps help (25k 0.66 → 50k 0.61); a longer
  run would nudge the marginal gate a little (near its floor). Lower priority than the coherence gap.
- **Drift tail** (steps 13–15, longest tracks): minor residual after SS; higher ss_prob or the helix
  anchor could close it. Low priority.
- **Helix anchor** (optional): physics-guaranteed no compounding + reco-fidelity bonus; SS already
  did most of the work, so this is now a refinement, not a necessity.
- **Calo incidence head MISSING** — every calo slice/model/gate is conditioned on `n_cells >= 1`, i.e.
  P(shower | deposits), never P(deposits | particle); only 68.7% of particles deposit and the rate is
  species-dependent (e- 50%, e+ 95%, n 35%, K0L 100%). Logistic probe on the standard contract already
  gets AUC 0.897 (`scripts/calo_incidence_probe.py`). Blocks an honest full-event calo gate; the calo
  analogue of the tracker count head.
- **Multi-species calo capacity — PARTIAL lever, quantified**: a dedicated e± head (2 classes, same
  data/steps) gives e- **0.773** / e+ **0.760** vs the shared head's 0.888/0.887, so capacity is worth
  ~0.12 — but it does NOT reach photon-like 0.56, i.e. e± are intrinsically harder than photons
  (9 cells/shower vs 4, wide production-radius spread). Ckpt `electron_v1`, slice `electron_v1.npz`.
- **Count head is wrong for pions**: multispecies count head undercounts (gen median 1 vs real 10) →
  honest count-driven pion gate meaningless until a per-species/pion count head exists. Orthogonal to tracker.
- **Generalize v3**: all-species surface slice + per-species count head → honest full-event gate vs the 0.77.
- **Stale checkpoints**: old `pion_v1`/`pion_v2_long` are 128-bin/pre-abspos, unloadable by current code.

## CALO STATE OF PLAY after 2026-08-14 (read this before proposing anything calo)
**Best config: plain Phase 1 helix** (`pion_anchor`, `electron_anchor`) — unbeaten across SIX
variants tried on 2026-08-14: anchor_cond, `line`/`auto` anchors, anchor_cond+separate trunks,
energy-head-on-`log_n`, and the combinations.
- **The core is solved.** e± per-bin core spread 0.585 → **0.99** (residual frame); with anchor
  conditioning the physical frame reaches **1.00** and branch ratios ~0.8-1.0. Pion residual 0.93.
- **The binding constraint moved to the ENERGY head.** `frac_near_floor` has been the top or
  near-top discriminator in every run since 2026-08-13, and no per-head patch has moved it.
- **The recurring pattern: every change improved its own mechanism target and lost on the gate**,
  because 8 of the gate's 10 features are energy/multiplicity, downstream of the head being
  perturbed. A change that halves the core error and nudges `logE_p90` reads as a regression.
  The width gate was added for exactly this reason; the core may need the same treatment.
- **Separate trunks isolate exactly** (val_gnll bit-identical across runs differing only in the
  energy head) and buy a much better GlobalHead (val_gnll -3.395 vs -2.427) for a worse energy head
  (val_ehl 0.2655 vs 0.2083). The energy head's loss under isolation is a **multi-task
  representation benefit**, NOT a missing multiplicity signal — `log_n` conditioning recovers
  essentially none of it (0.2655 → 0.2633).
See [energy-head log_n entry](experiment-memory/2026-08-14-energy-head-logn-PARTIAL.md).

## Partly-right calo diagnosis (2026-08-14) — the energy head is under-informed about MULTIPLICITY
`frac_near_floor` is a function of multiplicity, which is the **GlobalHead's** variable (`log_n`).
Every run uses `--no_energy_glob`, so the energy head is never told multiplicity and must infer it
**through the shared trunk**. That one fact explains five separate failures:

| change | effect on that implicit signal | `frac_near_floor` |
|---|---|---|
| width_norm | changes the GlobalHead task | 0.616 → 0.858 |
| ctx_norm | changes the GlobalHead task | 0.616 → 0.858 |
| anchor_cond | GlobalHead gets a direct route, leans on the trunk less | 0.601 → 0.660 |
| `auto` anchor | same + a discontinuous target | (cell_logE 0.082) |
| **anchor_cond + separate trunks** | **removes the shared route entirely** | **0.601 → 0.828** |

Confirmed by likelihood, not just the gate: e± **val_ehl 0.2655 (separate) vs 0.2141 (shared)** at
60k, with val_cfm/val_gnll identical. It also *predicts* Phase 0c's species ordering (separate trunks
hurt photon 4.4 cells/shower, mild on e± 10.6, helped pion 15.6) — fewer cells, more the floor
fraction matters per event, more the energy head depends on knowing multiplicity.
**TESTED (`--energy_glob_idx 1`), and it is REAL BUT SMALL**: floor fraction improves on every
species (pion 0.618→**0.574**, e− 0.601→**0.583**, e+ 0.592→**0.564**), per-shower energy improves
2.4x (`logEreco` W/σ 0.0399→**0.0165**, the best on record), val_ehl improves in both trunk settings
— but the upper energy percentiles get worse (`logE_p90` 0.548→0.589) so gate8 nets out slightly
worse, and it does NOT rescue the separate-trunks damage. Flag stays opt-in.
See [falsification entry](experiment-memory/2026-08-14-anchor-cond-separate-trunks-FALSIFIED.md)
and [the log_n result](experiment-memory/2026-08-14-energy-head-logn-PARTIAL.md).

**Anchor conditioning is confirmed robust and trunk-independent on its own target**: e± physical core
mechanism 1.275 → **1.00**, pion 0.929 → **1.02**, branch ratios → ~0.8-1.0, at identical
val_cfm/val_gnll. The only thing between it and a shipped win is the floor fraction above.

## Ruled out
- **energy head on `log_n`** (`--energy_glob_idx 1`) as a GATE lever — improves its target on every
  species and is the best per-shower-energy setting on record, but loses the upper energy
  percentiles and nets slightly worse; recovers none of the separate-trunks damage.
  [2026-08-14](experiment-memory/2026-08-14-energy-head-logn-PARTIAL.md).
- **anchor_cond + `--separate_trunks`** — PREDICTION FALSIFIED (predicted gate recovery; got
  gate8 0.886→**0.965**, cell_logE 0.019→**0.080**). Killed the trunk-interference diagnosis and
  produced the unifying one above.
  [2026-08-14](experiment-memory/2026-08-14-anchor-cond-separate-trunks-FALSIFIED.md).
- **Anchor conditioning on a SHARED trunk** (`--anchor_cond`) and the **`auto` pT-split anchor**
  (`--core_anchor auto`), [2026-08-14](experiment-memory/2026-08-14-anchor-cond-and-auto-anchor-FAILED.md).
  Anchor conditioning *worked* on its target — physical mechanism 1.275→**1.003**, `d_eta` W/σ
  0.027→0.007 — but degraded the untouched ENERGY head (`frac_near_floor` 0.601→0.660, gate8
  0.746→0.886), the same fingerprint as width_norm/ctx_norm. Injecting after the trunk blocks
  forward contamination but **not backward** (the gradient the GlobalHead returns changes what the
  energy head sees). `auto` was worse still (gate8 0.968, `cell_logE` 0.082): its pT split is a
  **discontinuity** in the anchor rule that the smooth `log_pt` conditioning cannot express, so a
  2.69x tighter frame produced the worst residual spread of any run (0.646). **Lesson: measuring a
  frame in isolation is not sufficient — a tighter target that is discontinuous in the conditioning
  can be harder to model than a looser smooth one.** Flags stay opt-in. Retest of anchor_cond with
  `--separate_trunks` is the indicated follow-up.
- **Calo width-in-global (`--width_norm`)** — targeted the intrinsic cloud size, which is only 6-17%
  of the measured width variance (the core is 83-93%). Width AUCs flat; energy marginal degraded via
  the shared trunk. e- gate8 0.773→0.969, pion 0.809→0.838.
  [2026-08-13](experiment-memory/2026-08-13-calo-width-in-global-FAILED.md). Flag stays opt-in/unused.
- **Calo energy head conditioned on the sampled global, and partitioning cells to the sampled total**
  (2x2 on both species, [2026-08-13](experiment-memory/2026-08-13-calo-energy-fix-2x2-and-multispecies.md)):
  photon 0.557 → 0.820 (conditioning) / 0.944 (partition) / 0.808 (both); pion 0.809 → 0.935/0.998/0.977.
  Both degrade the cell-log-E marginal the gate is built from. Flags `--no_energy_glob`/`--no_partition`.
- v1 drift levers (all failed to beat 0.77; see TRACKER_GATE_FINDINGS.md): lower sampling
  temperature; count-routed single-hit placement; vertex anchoring (2 retrains); input jitter;
  momentum-z input feature (moved drift only ~10%).
- **v3 vertex/seed lever** — teacher-force test showed the v3 seed is already correct on multi-hit
  tracks; the "too-inner seed" was a population artifact. See
  [2026-08-10](experiment-memory/2026-08-10-v3-exposure-bias-scheduled-sampling.md).
