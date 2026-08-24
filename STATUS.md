# STATUS — GenPU tracker

Rolling state. Append-only detail lives in `experiment-memory/`. See also `TRACKER_GATE_FINDINGS.md`.

_Last updated: 2026-08-24_

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

## SUBSYSTEM VISIBILITY (2026-08-24) — read before any incidence / cross-subsystem work
[Census entry](experiment-memory/2026-08-24-subsystem-visibility-census.md) ·
`scripts/subsystem_visibility.py` · real data, no model.
- **The tracker has the SAME missing-incidence hole as the calo, and it is bigger.** Both slice
  builders filter to `n >= 1` (`build_count_slice_stage2.py:24`, `build_tracker_slice.py --min_hits`)
  and `CountHead` is a categorical over **1..48**, so it structurally cannot emit zero. Tracker
  zero-trace **59.6%** vs calo **31.3%** (shard 0, among particles visible somewhere —
  `preprocessing.py:223` already dropped the fully invisible).
- **Only 9.2% of particles are seen by BOTH subsystems** (trk-only 0.313 / calo-only 0.596 /
  both 0.092 / neither 0.000). The population-level reason the tracker->calo edge came out
  anti-complementary: the two subsystems see **near-disjoint populations**.
- **The outcome is ~94% deterministic** (depth-8 tree, baseline 0.595; depth-16 0.961), driven by
  **log_E 0.53** > |vz| 0.17 > pdg_class 0.14 > vr 0.08. Species alone is a WEAK predictor
  (per-species purity ~0.5); what separates outcomes within a species is energy and production
  vertex — charged born vr > 1100mm is calo-only at **0.986**.
- **The residual ~5% is the conversion/interaction coin flip**, i.e. physics a generator should
  SAMPLE, not predict. Photons: P(trk | charged daughter) **0.185** vs P(trk | none) **0.001**
  (factor ~200); pi0/K0L/K0S/nbar flat **0.000**. Neutrons invert it (0.576 with NO daughter) —
  unresolved. Photon hits are real isolated deposits (median **1 hit**), 75% distinct from the
  daughters' hits, not double-booking.
- **No decay head.** `pileup_generator_plan.md:359-379` already ruled it out: genuine decays are a
  minority, secondary production is material-dominated, and injection is a GENERATION-TIME concern —
  training conditions on truth particles that already contain every conversion daughter. The
  conversion is an INPUT. The useful object is `P(leaves trace | E, eta, vr, pdg, has-chg-daughter)`.
  **Scope caveat**: that holds for TRUTH-conditioned response only. Sampling novel particles needs
  the cascade generator (conversion/brem/nuclear, ~96% of secondary production) — already spiked
  and validated, see the Cascade section above. Decays specifically stay a ~4% analytic add-on.
- **BLOCKER**: stage2 stores `parent_id` but NOT the particle's own id, so the parent->child graph
  cannot be rebuilt downstream (`preprocessing.py:233` has `particle_ids`, never writes it to the
  npz). One-line fix; unlocks the strongest neutral feature.
- **Methodology**: the first version of this probe joined by row index (agreement **0.000**) and
  produced a flat, physically impossible ~12% tracker rate for pi0. The script now prints a
  wrong-event NULL column so the failure is visible. `preprocessing.py` joins correctly.

## CASCADE / SECONDARY GENERATION (Jul 8, RECOVERED into STATUS 2026-08-24)
[scoping_secondary_cascade.md](scoping_secondary_cascade.md) · commits 9290678..f164d05 · spike
artifacts in `/scratch/gpfs/IOJALVO/lv7805/genpu_data/cascade/`. **This line was validated and then
fell out of the rolling state**; it is the answer to "can we generate NEW particles, not just
respond to truth ones".
- **The gap it names**: 91.6% of particles are secondaries, and **61.4% of tracker hits + 84.4% of
  calo hits/energy come from secondaries** (material effects, r=20-1100mm; decays only ~4%). Truth-
  conditioned input hands over most of the response — fine for the current deliverable, but it does
  NOT support sampling a novel particle, whose daughters would be missing.
- **Option C is FORCED by data, not chosen.** `any_daughter_frac == conv_frac` in every energy bin
  and **90% of high-E photons have NO daughter particles** — their shower is recorded as calo HITS,
  not a particle tree. So: **calo absorbs its cascade inclusively** (per-incident particle; there are
  no daughters to generate), **tracker needs an explicit secondary-track model** (clean particle-
  generation problem; the crux and highest-risk piece), analytic decays a cheap ~4% add-on.
- **Both hard cases SPIKED AND MATCHED** with cond-only models. Photon conversion: convert_frac
  0.6153 → **0.6148**, log_conv_r mean/std 6.073/1.030 → **6.072/1.027**, and the non-monotone
  convert-vs-E curve including the high-E collapse (0.104 → 0.100); `esplit` 0.711 → 0.678 is soft.
  Pion nuclear: interact_frac 0.4384 → **0.4391**, multiplicity matched bin-by-bin (mean 3.073 →
  3.085), log_int_r 5.707 → 5.733.
- **Loop CLOSED for conversions**: generated e± vs truth e± through the SAME tracker head give
  layer_class 20.4 vs 19.2, frac_inner 21.6% vs 23.8%, hit radius 304 vs 285 mm — the generated
  cascade reproduces truth HIT placement, not just particle statistics. Caveats: fixed n_hits (tests
  placement not count), gen emits 2 e± vs truth's ~1.5 recorded, collinear-direction approximation.
- **Sequencing (unchanged, still correct)**: do NOT build the cascade generator until the per-detector
  heads are solid and M3 assembly works on truth input. M3 has no section here yet, so cascade stays
  deferred — the spikes de-risked it, they did not promote it to a component.
- **Consistent with the 2026-08-24 census**: calo-only is 59.6% of particles and only 9.2% are seen
  by both subsystems, i.e. most calo energy arrives from particles the tracker never saw — the calo
  cascade really does live in hits, not in the particle list.

### DIRECTION SET 2026-08-24 — primaries-in with a LEARNED cascade
[Entry](experiment-memory/2026-08-24-cascade-depth-and-primaries-in-scoping.md).
- **The cascade is SHALLOW**: depth 0/1/2/3 carry 40.5 / 35.7 / 13.7 / 6.6% of tracker hits, so
  **3 levels = 96.5%, 4 = 99%**; 3.3 particles per primary. Each level is **fully batchable**
  (all depth-k particles in one call), so sequential depth is ~4 regardless of event size.
- **But the SPEED case for learning the tracker cascade is weak** — Geant4's cost is dominated by
  CALO shower development, which the ML calo head already replaces. The tracker cascade is the cheap
  part. Fatras is not "slightly faster Geant4": simplified surface geometry, parameterized material
  effects at surface crossings, and **no calo showers at all**. **The case for primaries-in is
  ARCHITECTURAL** (self-contained generator, no external geometry dependency), not throughput —
  know this before it goes in a paper. Fatras retained as a **validation target**, not discarded.
- **Both spikes only validated ONE level** (direct daughters = 76% of hits). Depth 2-4 and **error
  compounding across levels** are untested — the same failure mode the tracker head already has
  (v3 needed scheduled sampling), as the scoping doc predicted.
- **Conditioning gaps**: `build_conversion_slice.py:54` uses `[log_E, eta, vr, vz]` — **no phi** (the
  material map is learned phi-averaged; untested for ODD), and for CHARGED parents the material
  integral is a helix-path quantity that a production-point proxy does not capture.
- **PRIMARY FRACTION — RESOLVED, M0's number was wrong**: **~27% primaries** (`primary` flag), not
  8.4%. **Every** particle carries a `parent_id` (range 0..6295; `parent_id <= 0` is 0.000), so M0's
  "has a parent" secondary test is TRUE for ~100% and does not measure what it claims. The cascade
  must generate ~73% of the list, a **3x SMALLER** job than "91.6% of particles are secondaries"
  implied — a claim that appears in `pileup_generator_plan.md:363` and should be re-read.

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

## CALO SHOWERS ARE OVER-SPLIT (2026-08-24) — candidate mechanism for the standing e± problem
[Entry](experiment-memory/2026-08-24-calo-shower-oversplitting.md) · measurement only, **hypothesis
UNTESTED** (no retrain run).
- `build_calo_slice.py` groups by `calo_offsets` = **per direct depositing particle**. But **64.8% of
  calo depositors were born INSIDE the calorimeter** (past the front face r>1259mm / |z|>3212mm),
  carrying **54.2% of all calo cells**. Those are shower FRAGMENTS trained as showers, with
  "incident kinematics" that are really mid-shower kinematics.
- **The species ordering tracks the calo difficulty ordering**: γ **0.1%** split (gate **0.557**, best
  on record) · π± ~56% (0.809) · e± **74.2%** (0.888 ms / 0.773 dedicated) · p 87.3% (0.749).
- This supplies a mechanism for a line elsewhere in this file that blames intrinsic difficulty —
  "e± ... wide production-radius spread" **IS** the over-splitting (e± born at varying calo depths).
- **Counter-evidence, stated plainly**: proton is 87.3% split yet gates better than e±. It deposits a
  median 3 cells so plausibly a different regime, but the correlation is NOT clean and this is not proof.
- **CORRECTED 2026-08-24 (job 12879425)**: cells have 1.21 contributors on average (89.3% single),
  so the merged count above was summed rather than deduplicated — **e± cells/shower is ~17.9, not
  20.72** (1.154x inflation). Direction and magnitude unchanged; any implementation must DEDUPLICATE.
  Within-shower sharing 7.2% vs true superposition 3.5% — and that 3.5% is a **PU0 floor**, far
  higher at M3's mu = 30-200. [entry](experiment-memory/2026-08-24-calo-cell-sharing-corrects-j3a.md)
- **The fix is half of Option C** (calo-inclusive) and is needed for Pythia-input generation anyway:
  re-attribute each cell to its **calo-incident ancestor**, rebuild slices, retrain, gate — **2 seeds
  minimum**. BLOCKED on `preprocessing.py` storing `particle_ids`.
- **If it moves the gate**, every calo result since 2026-08-13 was measured through a confounded
  training target and the `frac_near_floor` / n=1 thread must be re-read in that light.

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
4. ~~**Condition the calo on the GENERATED TRACK's outer state**~~ — **FALSIFIED 2026-08-16 on real
   data, before any training.** 81% of pion / 93% of e± curlers leave **zero tracker hits**, and e±
   have essentially none in any branch (86-98% empty). Tracks exist exactly where the helix already
   works (face branches, 15-24x) and are absent where it is weak (curlers, 2.5x) — anti-complementary.
   See "Ruled out" and the [entry](experiment-memory/2026-08-16-curler-tracker-coverage-PHASE4-FALSIFIED.md).
   Surviving scope is narrow: a last-hit **position + direction** extrapolation for face-reaching
   **pions** only (endcap 76% coverage, median 9 hits), where the helix already gets 15-24x. No
   longer the project's novelty claim.

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
- **Turning-point anchor is a proxy, and 2026-08-16 showed there is NO better one available**: 63% of
  pion / 72% of e± showers are energy booked to a parent that never reached the calo, and those
  particles leave zero tracker hits 81% / 93% of the time — so Phase 4 cannot own that population.
  Its ~2.5x is the ceiling unless something other than the tracker supplies the stopping point.
  (Also measured: the vacuum turning point's φ is **uncorrelated** with the real outermost hit's φ —
  |Δφ| median 1.567 rad ≈ π/2, the uniform value — so its 2.4x gain is not φ agreement with the
  actual track.)
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
- **Incidence head MISSING on BOTH subsystems** (upgraded 2026-08-24 from "calo only") — calo is
  conditioned on `n_cells >= 1` (68.7% deposit; e- 50%, e+ 95%, n 35%, K0L 100%) and the TRACKER is
  conditioned on `n_hits >= 1` (only **40.4%** leave a hit). Blocks an honest full-event gate on both.
  Now the cheapest open item: the 3-way outcome is **94% predictable** from truth kinematics and the
  residual ~5% is the conversion coin flip, which the generator should sample. Existing calo logistic
  probe already gets AUC 0.897 (`scripts/calo_incidence_probe.py`).
  See [census](experiment-memory/2026-08-24-subsystem-visibility-census.md).
- **Stage2 cannot rebuild the parent->child graph** — `particle_aux` carries `parent_id` but the
  particle's own id is never written (`preprocessing.py:233`). One-line fix; unlocks
  "has charged daughter", which moves photon P(tracker hit) 0.001 -> 0.185.
- **Neutron tracker visibility unexplained** — 48% have tracker hits, and NOT via recorded charged
  daughters (P(trk | daughter) 0.070 vs P(trk | none) 0.576). Probably sub-threshold nuclear recoil.
- **The tracker's modelled population is majority stubs** — 44.7% of charged particles with >= 1 hit
  have exactly ONE (median 2), plus photon single deposits (median 1 hit, real not double-booked).
  Whether isolated deposits belong in the tracker slice at all has never been asked.
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
- ~~**The core is solved.**~~ **CORRECTED 2026-08-24 (job 12879155)** — the "0.99" is carried by the
  74% of the slice born INSIDE the calo, which receive the turning-point FALLBACK anchor. Split:
  born-inside **1.05**, born-outside (real showers) **1.86** (worst bin 2.71). In the frame the
  mixture fits, real showers are badly over-dispersed. Physical-frame ordering REVERSES (all 1.29 /
  inside 1.21 / outside **1.15**) because the anchor is exact and carries ~82% of the physical core,
  so net impact is moderate — but the ANCHOR is better than documented (**5.6x** phi tightening on
  real showers vs 1.9x pooled) and the MODEL is worse. Pion residual 0.93 unaudited.
  [entry](experiment-memory/2026-08-24-phase1-headline-carried-by-fragments.md)
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

## THE ENERGY-HEAD TARGET IS NOW PRECISE (2026-08-16) — p(floor | n) at small n
[Entry](experiment-memory/2026-08-16-floor-iid-test-FALSIFIED-and-n-dependence.md) · job 12470841 ·
`scripts/calo_floor_dispersion.py`. The Aug-14 multiplicity diagnosis was right in MECHANISM and
wrong in TARGET. It is not "tell the energy head n" (`--energy_glob_idx 1`, worth 0.618 → 0.574);
it is that **p(floor) must be an explicit, strongly NON-LINEAR function of n**:
- **e±: the entire error is at n = 1.** Real p(floor) **0.011** (e−) / 0.009 (e+), generated
  **0.061 / 0.062** — a **6x excess** over ~8% of showers each. By n ≥ 9 real and gen agree to 0.001.
  Physically obvious: a one-cell shower's cell carries the ENTIRE shower energy, so it cannot be a
  faint fringe cell. A single `log_n` scalar into an MLP dominated by large-n showers under-fits
  exactly that corner.
- **Pion: no n-dependence at all.** Real p climbs 0.018 → 0.032 with n; generated is flat ~0.018.
  Its defect is a MISSING SLOPE, not a small-n corner — same fix, different reason.
- Cells-per-shower itself is fine (pion 15.57 real / 15.72 gen; e− 10.50 / 10.52), so `log_n` is not
  the problem — what the head does with it is.
- **Free first probe: the `partition` A/B.** All logged runs use `--no_partition`; with partition ON
  cells renormalise to the sampled total, which for n = 1 *forces* the cell to carry `total_logE` and
  would fix the e± defect by construction. Partition was ruled out 2026-08-13 on the POOLED
  `cell_logE` marginal — never checked at small n.

### The floor-Bernoulli-on-n fix FAILED its own target (2026-08-16, job 12473556)
`--floor_n_buckets 32` (at-floor logit gets its own net + an embedding of the shower cell count)
moved p(band | n=1) by **0.003** (e− 0.061 → 0.058 vs real 0.011). **The attribution was wrong.**
`EnergyHead`'s Bernoulli models only the NARROW pile (`|logE−log_floor| < 0.05`), while
`frac_near_floor` scores a WIDE band (`logE < log_floor + 0.5`) that is *also* fed by every low
**mixture** draw — including the sub-floor tail `EnergyHead.loss` deliberately leaves to the mixture.
If the excess is mixture-tail, no Bernoulli change can reach it (and that is why `partition`, which
rescales cell energies, could). Decomposition running as job 12526752.
See [the entry](experiment-memory/2026-08-16-floor-n-buckets-PARTIAL-mechanism-untouched.md).

### SETTLED 2026-08-24 (job 12526752) — mixture tail, and the gate gain was SEED NOISE
[Entry](experiment-memory/2026-08-24-floor-band-decomposition-and-seed-replicate.md).
- **The n=1 excess is the MIXTURE's sub-floor tail, not the Bernoulli.** e− at n=1, real/gen:
  sub-floor **0.0019 / 0.0368** (19x, ~90% of the excess), pile 0.0007 / 0.0054, just-above
  0.0086 / 0.0184. Matched to ~0.001 by n >= 5. So no Bernoulli change could ever have reached it,
  and `--floor_n_buckets` is **ABANDONED**.
- **The e± gate gain did not replicate.** Arm B seed 1: e− gate8 **0.8272**, e+ **0.8133** — worse
  than the Phase 1 baseline (0.7456 / 0.7488) and 0.11 away from seed 0 (0.7204 / 0.7015).
- **SEED SPREAD ON e± gate8 IS ~0.11.** Every single-run gate delta on record smaller than that is
  uninterpretable — which covers most of the 2026-08-14 sequence. **Two seeds minimum from here.**
- **Exact remaining target**: the Gaussian mixture's lower tail at **n = 1 for e±**. At n=1 the cell
  carries the ENTIRE shower energy, so the correct constraint is a BOUND (`cell_logE == total_logE`),
  not a bucket. `partition` enforced that globally — right at n=1, catastrophic elsewhere. **An
  n=1-only constraint is the untried version.**

**Anchor conditioning is confirmed robust and trunk-independent on its own target**: e± physical core
mechanism 1.275 → **1.00**, pion 0.929 → **1.02**, branch ratios → ~0.8-1.0, at identical
val_cfm/val_gnll. The only thing between it and a shipped win is the floor fraction above.

## Ruled out
- **`--floor_n_buckets` (floor Bernoulli on a cell-count embedding)** — mechanism untouched (0.003)
  and its apparent e± gate gain did not survive a `--seed 1` replicate (0.7204 -> **0.8272**).
  Flag stays opt-in at 0. Job 12526752,
  [2026-08-24](experiment-memory/2026-08-24-floor-band-decomposition-and-seed-replicate.md).
- **A DECAY head** — not indicated in the current truth-conditioned scope: conversion daughters are
  already in the input particle list, genuine decays are a minority of secondary production, and
  injection is a generation-time concern (`pileup_generator_plan.md:359-379`). The incidence head
  is the object that is actually missing.
  [2026-08-24](experiment-memory/2026-08-24-subsystem-visibility-census.md).
- ~~**Track-endpoint anchor / Phase 4 for the CURLER population**~~ — **REOPENED 2026-08-24, the
  premise was contaminated.** The logged "81% (pion) / 93% (e±) of curlers have zero tracker hits"
  pooled two unrelated populations: the turning-point branch is **81-92% particles born INSIDE the
  calorimeter** (endcap shower fragments at vr ~ 420 mm but |vz| > 3212 mm), which have no tracker
  hits by construction. For **genuine tracker-born curlers, 95.5% of pions and ~77% of e± DO leave
  tracker hits**. This does NOT show Phase 4 works — Q3 has never been measured on the clean
  population — only that the reason for dismissing it was invalid, and the door it reopens is
  narrower (true curlers are 18.7% of the pion branch, ~8% of e±). Jobs 12874732 / 12874782,
  [2026-08-24](experiment-memory/2026-08-24-turning-branch-contamination-phase4-REOPENED.md).
  **RESETTLED same day (job 12878141)** — Q3 measured on the clean population at last: the
  track endpoint buys only **1.16x (pion) / 1.20x (e±) in eta and NOTHING in phi**, vs the
  helix's 15-24x on face branches — and that is an UPPER BOUND, using the real truth endpoint
  rather than a generated one. **Stays unbuilt**, now for a supported reason. Two facts worth
  keeping: e± coverage is **complementary**, not anti-complementary (face branches 86-96%
  EMPTY, curler branch 77% tracked — the reverse of what was logged), and pion curlers are
  95.6% tracked with a median 11 hits.
  [Q3 clean](experiment-memory/2026-08-24-q3-clean-phase4-resettled.md).
- **`partition` as the n=1 floor fix** — MECHANISM CONFIRMED, DELIVERY REJECTED. Forcing the single
  cell to carry the total lands e− n=1 exactly on truth (0.061 → **0.011** vs real 0.011), proving
  the physical reading — but it rescales every cell's continuous energy: n ≥ 2 gets worse on every
  species (pion n=4 0.016 → 0.053 vs real 0.022), the generated floor count becomes OVER-dispersed
  (D 1.03 → **3.81** pion), and pooled `cell_logE` 0.0223 → **0.309**, gate8 0.8055 → **0.9977**.
  Job 12471293, [2026-08-16](experiment-memory/2026-08-16-partition-AB-mechanism-confirmed-delivery-rejected.md).
- **"Cells need to know about each other" as the `frac_near_floor` lever** (set model / correlation
  latent) — real showers are barely over-dispersed at all: `D_floor` real **1.035** (e±) / 1.199
  (pion) vs an i.i.d. floor of 1.000, so there is nothing for cell coupling to buy. The positive
  control inverted too (real AND gen width over-dispersed 3-16x, gen *more* than real). Job 12470841,
  [2026-08-16](experiment-memory/2026-08-16-floor-iid-test-FALSIFIED-and-n-dependence.md).
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
