# STATUS — GenPU tracker

Rolling state. Append-only detail lives in `experiment-memory/`. See also `TRACKER_GATE_FINDINGS.md`.

_Last updated: 2026-09-15_

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
- **The gap it names** (numbers CORRECTED 2026-09-07, see PIPELINE.md §7 4a): **73.8%** of particles
  are non-primary (`primary == 0`; the old 91.6% used the "has a parent" test, true for ~100%), and
  they deposit **59.7% of tracker hits** and **83.5% of raw-attributed calo energy** — but 58.4% of
  calo energy comes from particles born INSIDE the calorimeter, which the v2 slice re-attributes to
  the calo-incident ancestor, leaving **45.7%** conditioned on a secondary. The old 61.4% / 84.4%
  pair is superseded. Material effects dominate (median secondary vr 424 mm); decays only ~4%. Truth-
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

**MOVED FORWARD 2026-09-07**: v3's frame change is only HALF done. The representation is
per-module standardised but NOT rotated, so the plane constraint the module encodes is thrown away
again by the independent position heads (section below). The next tracker hypothesis is that a
ROTATED two-coordinate in-plane frame is to v3 what v3 was to v1.

## TRACKER v3's "SURFACE-LOCAL" FRAME IS NOT ROTATED (2026-09-07) — generated hits are off the silicon
[Entry](experiment-memory/2026-09-07-tracker-v3-local-frame-is-not-rotated.md) · real data, no
model, no job (`surface_pion_proto.npz`, 809,750 hits).

`ModuleGeometry.local_residual` (`module_geometry.py:52`) is `(physical - mean)/std` on **global**
Cartesian axes — centred and scaled, NOT module-frame. And `generate`
(`tracker_module_ar.py:265-271`) samples `x_head/y_head/z_head/time_head` from the SAME hidden
state, independently, **not conditioned on the module just drawn** (the module enters only via
denormalisation). Layer→surface IS hierarchical; the positions are not.

A module is a PLANE, so the three residuals are linearly dependent:
- **barrel `|corr(x,y)| = 1.000` exactly** (vol 17); barrel is **52.1%** of hits.
- **endcap local z std = 0.05 mm** — a disc at fixed z, so the 512-way `z_head` predicts a constant
  for the **47.9%** of hits there.

Off-plane error (fit each module's plane; permute each standardised column within the module, which
destroys the joint and preserves every marginal — exactly what independent categoricals permit):

| volume | hits | real off-plane | independent-column draw |
|---|---|---|---|
| 17 | 25.9% | **0.0000 mm** (p99 0.0000) | **2.35 mm** |
| 24 | 20.6% | **0.0000 mm** (p99 0.0000) | **6.54 mm** |

Real hits are EXACTLY planar (silicon smears in-plane only). Invisible to the gate — every feature
is a marginal in (r,phi,z) and layer spacing is tens of mm — and **fatal for ACTS**, which is the
open item that decides v3 vs v4. **Caveat**: no model was sampled; the permutation test bounds what
the factorisation permits, it does not measure what the trained model does.

**Fix**: rotate into a real module frame, predict **two in-plane coordinates, drop the normal**.
Exact by construction, and deletes the wasted endcap z head. Needs a `module_geometry.npz` rebuild
(per-module rotation, versioned), surface-slice rebuild, retrain.

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

## CALO v2 (2026-08-24) — 3D + re-attributed; `frac_near_floor` COLLAPSED TO CHANCE
[Entry](experiment-memory/2026-08-24-calo-v2-first-3d-model.md) · ckpts `electron_v2_s0/s1`
(60k, 2 seeds) · slices `electron_v2.npz` / `electron_v2_h5.npz` · jobs 12899065, 12900324.

**NOT COMPARABLE TO ANY v1 NUMBER** — v1 gated 2D fragment-showers (10.5 cells) against a v1
reference; v2 gates 3D re-attributed showers (19.8 cells) against a v2 reference. Reading
0.7456 → 0.81 as a regression is exactly the error the metric change prevents.

- **`frac_near_floor` = 0.5045, i.e. CHANCE.** It was the top or near-top discriminator in every run
  since 2026-08-13 (0.59-0.60) and survived ~10 experiments. The August energy-head thread was
  chasing a data-contract artifact, not a modelling failure.
- **The evidence was already in this file.** v1 `frac_near_floor` tracks each species' FRAGMENT
  fraction monotonically: γ **0.1% split → 0.510** (already chance), π± ~56% → 0.666,
  e± 74% → 0.702. **Testable prediction: pion v2 should fall from 0.666 toward chance; photon,
  never broken, should barely move.**
- **Depth is modelled** — energy-weighted mean depth real **131.0** vs gen **137.7 / 138.7** mm,
  bulk profile bins within 3%, the 0-25 mm bin exact to 4 dp. First time the plan's longitudinal
  acceptance metric has been computable at all.
- **Seed spread ~10x tighter**: gate8 0.8105 / 0.8074 (**0.003**) vs v1's **0.11**. Likely because
  v2 is a homogeneous population where v1 mixed fragments with real showers.
- **New leading discriminator: `logE_max` 0.5758** (upper energy tail). Composite gate 0.81 with no
  single feature above 0.58, so per-feature AUCs will NOT say what to fix next — a different
  diagnostic is needed from the one used all August.
- **Confounded by design**: dedup + re-attribution + depth landed together. `--no_reattribute` exists
  if attribution needs isolating.
- **e± ONLY.** No pion / photon / multispecies v2 slice yet.

## CALO IS TWO DETECTORS AND THE DEPTH AXIS CANNOT SAY WHICH (2026-08-27)
[Entry](experiment-memory/2026-08-27-calo-ecal-hcal-step-vs-shower-physics.md) ·
`scripts/calo_section_split.py` · job 13033802 · real data, no model.

ECAL (dets 9,10,11) = **71.5%** of energy at 5.050 mm layer pitch; HCAL (12,13,14) = **28.5%** at
51.000 mm, with a PHYSICAL GAP between them (barrel 228->388 mm, endcap 227->435 mm). Mean cell
energy **steps 3.06x** (+1.12 in logE) across it — an HCAL cell integrates ~10x more material.

**`build_calo_slice_v2` measures depth from ONE front face for every detector** (`calo_geom.py:32`),
so the axis is AMBIGUOUS: barrel and endcap overlap on it but transition at different depths (388 vs
435), and at depth 150-250 mm the population is 12% barrel / 88% endcap. **`point_layer` is ambiguous
too** — per-detector index with no offset, so layer 20 is EM endcap OR hadronic endcap. The model must
reproduce a 3x step whose location depends on a variable it is never given.

**But the gradient is mostly PHYSICS, so the 2026-08-27 headline stands.** rho(logE,depth) with the
step removed: p **+0.542 -> +0.519** (holds within ECAL alone, +0.518), mu± +0.46 -> +0.38, pi±
+0.27 -> +0.20, gamma -0.101 -> **-0.110**, e± -0.013 -> **-0.034**. For EM the raw number was TOO
SMALL — a few hot HCAL cells masked a genuine negative ECAL gradient. The POOLED number was the
misleading one (53% geometry). And p vs pbar resolves: proton +0.518 within ECAL, **antiproton
+0.001** — antimatter annihilates promptly, matter cascades.

**FIX ORDER — the slice change now PRECEDES the head work.** Store `(section token,
depth-within-section)`; offset `point_layer` per detector. Same class as the tracker v1->v3
surface-local change (matched pion gate 0.9996 -> 0.80). The `calo_rz` sidecar already has (r,z).

## CALO COHERENCE IS HADRON-ONLY (2026-08-27)
[Entry](experiment-memory/2026-08-27-calo-coherence-is-hadron-only.md) ·
`scripts/calo_shower_coherence.py` · job 13033389 · both v2 checkpoints.

Two-point energy correlation, each side's one-point profile removed, vs a within-shower null.
Synthetic references: coherent **2.10**, one-point-only **0.90**; xi estimator FLOOR **+0.117**.

| centroid pull / null | classes |
|---|---|
| 0.97 - 1.00 (**no coherence**) | gamma, e±, pi0, K0S |
| 1.27 - 1.29 (marginal) | n, nbar, K0L |
| 1.54 - 2.22 (**coherent**) | pbar, K-, pi±, K+, **p** |
| 3.3 (track-like, see caveat) | mu± |

p and K+ clear the xi floor independently of the pull, which was the two-witness condition. Generated
is FLAT everywhere. `depth pull` is the sharpest real-vs-gen gap (e± real -0.0349 vs gen +0.0007, 70σ).

Coherence is a **continuum, not a binary**, so a hard EM/hadron ROUTER is wrong — K0S (1.00) and n
(1.27) sit in the gap. One **PDG-conditioned attention head** subsumes both (PDG is truth
conditioning). ~~showers cap at 128 cells so n^2 is free~~ — **WRONG, corrected 2026-09-15**: 128 was
a SAMPLER clamp, not a property of showers. Real showers run to **1,685** cells (pbar); n^2 attention
must be sized for that, or use inducing points / local attention. See
[cap entry](experiment-memory/2026-09-15-calo-cell-cap-128-was-real-and-gate-flat.md).

**RETRACTED**: "EM is 65% of showers so the cheap fix covers most of it". EM is 65.3% of showers but
only **46.6% of CELLS**; hadrons are 33.0% of showers and **51.6% of cells** — the majority, because
their showers are twice the size. Gate features are cell-level aggregates, so cells are the denominator.

## CALO FLOOR ATOM DELETED (2026-09-07) — not load-bearing, but the mixture smears the edge
[Entry](experiment-memory/2026-09-07-calo-floor-atom-removed-mixture-smears.md) · job 13557402 ·
ckpt `ele_noatom_s0` · **1 seed** · control `electron_v2_s0/s1` (job 12930682).

Measured first, on 13,054,567 held-out e± cells: **0.000000** sit at `log(5e-5)`, **0.387%** below,
and the threshold is a **x19 density step**. So `--no_floor_atom`: no Bernoulli, no BCE, mixture
trained on ALL cells (the old `cont` mask cut a hole the atom then filled back in). Output width
unchanged, so old checkpoints still load — with the atom ON.

| | real | control | **no atom** |
|---|---|---|---|
| `sub_floor_gen` | 0.00387 | not measured | **0.01032 (x2.7)** |
| `near_floor_gen` | 0.03793 | not measured | **0.03934 (x1.04)** |
| `cell_logE` W/sig | — | 0.0280 / 0.0262 | **0.0244 (best on record)** |
| gate8 | — | 0.8105 / 0.8073 | **0.8169** |
| `frac_near_floor` AUC | — | 0.5045 / 0.5461 | **0.5394** |

- **The atom is NOT load-bearing.** Gate flat (+0.006, ~2x the 0.003 seed spread, one seed),
  per-cell energy fit IMPROVED, `frac_near_floor` AUC inside the control's own seed range.
- **The predicted smearing is real**: 2.7x too many cells below a hard cut, with `near_floor`
  nearly exact — the error sits precisely at the edge.

**Verdict: the atom stays deleted.** The indicated replacement was TRUNCATION (normalise on
`[log_floor, inf)`) — **DECIDED 2026-09-07 NOT TO BUILD IT YET**: the 0.65 pp sub-threshold excess
is not worth the compute against the joint/copula defect (~0.70, the bigger half of the gate).
Recorded as PIPELINE gap 4b with its revisit conditions (cell projection landing; or
`frac_near_floor` resurfacing — it is still 0.73-0.74 on the POOLED model, never measured there).
**Caveat**: the x2.7 is against REAL, not against the atom-on model — `floor_edge` is a new
diagnostic and the control was never measured with it. ~7 min to close if wanted.

## CALO `--energy_pos`: MECHANISM CONFIRMED, GATE STILL OPEN (2026-09-07)
[Entry](experiment-memory/2026-09-07-calo-energy-pos-mechanism-CONFIRMED.md) · job 13556379 ·
controls job 13032162 · four `--energy_pos` checkpoints (train 13038484, eval 13045670).

Per-cell position + a 4-way ECAL/HCAL x barrel/endcap token into `EnergyHead`. The gate was measured
in August; **the mechanism target was not**, which left "learned it and paid elsewhere" and "never
learned it" both alive. Now measured — the pre-registered prediction is **CONFIRMED on every arm**:

| arm | rho(logE,r) real / **gen** | rho(logE,depth) real / **gen** | p(floor) core->fringe real / **gen** |
|---|---|---|---|
| e± dedicated s0/s1 | -0.098 / **-0.100 / -0.087** | -0.011 / **-0.017 / -0.024** | x3.55 / **x2.30 / x2.97** |
| e± pooled s0/s1 | -0.098 / **-0.093 / -0.104** | -0.011 / **-0.014 / -0.021** | x3.55 / **x2.98 / x3.48** |
| pi± pooled s0 | — | +0.249 / **+0.155** | x3.83 / **x2.65** |
| gamma pooled s0 | — | -0.100 / **-0.047** | x3.90 / **x3.04** |

**Every control `gen` number was +0.000 / x1.00.** e± is essentially exact laterally; pi± recovers
62%, gamma 47%, and the SIGNS are right on both — the PDG interaction the design argued for is real.

**So the dedicated-e± gate regression (0.811/0.807 -> 0.941/0.876) is NOT a failure to learn.**
Candidate reconciliation from this run's numbers: real p(floor) 0.022->0.078, dedicated-epos gen
**0.032**->0.073 — gradient present, **core LEVEL ~45% too high**; pooled 0.025->0.075, much closer.
Matches `frac_near_floor` marginal AUC 0.5045 -> **0.6967** on the dedicated arm while pooled
improved. The head bought the gradient and paid with the level. **UNVERIFIED** — one cheap
no-retrain check (overall floor rate, not the ratio) would settle it, and it lands squarely on the
floor redesign already ranked item 0.

**Do not revert `--energy_pos` on the gate alone** — it is the only change on record that moved the
joint mechanism, and the copula half is the bigger half.

## CALO: THE FLOOR PILE DOES NOT EXIST, AND ENERGY IS BLIND TO POSITION (2026-08-27)
[Entry](experiment-memory/2026-08-27-calo-energy-position-coupling-and-the-floor-that-isnt.md) ·
`scripts/calo_energy_position_coupling.py` · job 13032162 · real data + both v2 checkpoints.

**READ BEFORE ANY FURTHER `frac_near_floor` WORK.** Two facts overturn the premise of the whole
August energy thread:

1. **There is no zero-suppression pile.** Raw `calo_hits.total_energy`: min>0 = 5.0001e-05,
   frac below 5e-5 = 0.00000, 1.49M distinct values in 1.53M cells. The 50 keV cutoff is real and
   applies to the CELL TOTAL, but it is a **hard truncation, not an atom**. **Zero** cells sit at
   log(5e-5) in either the v1 or the v2 slice. `EnergyHead` pins its Bernoulli-fired cells to exactly
   that value (`calo_flow.py:229`), so the model puts ~0.56% of cells where the data has none.
2. **The gate feature and the head's mechanism are 12x apart in population.** Gate band
   (`logE < LF+0.5`) = **3.79%** of e± cells; head band (`|logE-LF|<0.05`) = **0.30%**. So **92% of
   what `frac_near_floor` measures is drawn by the MIXTURE, not the floor logit** — and
   `--floor_n_buckets` only ever touched the logit (`calo_flow.py:205`). Ten experiments tuned the
   rate of an atom that should not exist, on a branch governing 0.3% of cells.

**Energy is independent of position within a shower BY CONSTRUCTION** — every cell of a shower gets a
bit-identical input vector (`calo_flow.py:192`, all inputs `[src]`-gathered). Measured rho(logE, depth),
real vs generated: p **+0.542**, mu± +0.46/+0.44, pi± +0.27/+0.23, K+ +0.29, n +0.13, e± -0.01,
gamma **-0.101** — generated is **+0.000** on every species tested. p(floor) rises x2.4-4.5
core->fringe in real and is dead flat x1.00 generated, while the overall rate matches to four
decimals. **Signs flip by species and cancel when pooled** (-0.042 / +0.067), so the position input
must interact with PDG.

**The calorimeter is NOT phi-symmetric** (the `cont` contract's stated justification for omitting phi):
endcap occupancy modulates rms/mean 0.153 with a clean **n=80** harmonic; barrel median r swings
**22 mm** with phi (stave polygon). Harmless today — the model produces no cell-level geometry — but
binding once the gap-#1 cell projection exists.

**Two self-corrections in that script**: part C does not isolate cell-level coupling (particle
conditioning uncontrolled; the model already reproduces most of those partials and has 3x too much on
the floor pairs), and the cross-shower null was index-aligned and contaminated parts A/B (fixed;
part C unaffected). `gen` was the valid A/B null throughout.

Indicated redesign: drop the point mass for a mixture **truncated** at log(5e-5); generate energies as
**fractions on the simplex** so the sum is exact by construction and `partition` (which would smear the
edge by `s_rest`) disappears; put position into `self.net`, absolute depth first.

## CALO GATE DECOMPOSED (2026-08-25) — the defect is DEPENDENCE, and `frac_near_floor` is NOT fixed
[Entry](experiment-memory/2026-08-25-calo-gate-joint-structure-decomposed.md) ·
`scripts/calo_gate_diagnose.py` · `calo_metrics.py --dump_features` · job 12930682, both e± v2 seeds.

The 2026-08-24 line "per-feature AUCs alone will not say what to fix" is now resolved, and it
reverses the entry above it.

| held-out AUC | seed 0 | seed 1 |
|---|---|---|
| full | **0.8253** | **0.8329** |
| marginals only (joint destroyed) | 0.6007 | 0.6166 |
| **copula only (marginals matched)** | **0.7004** | **0.7242** |
| copula, quadratic logistic (= correlation matrix) | 0.6620 | 0.6666 |
| copula, linear logistic [control, must be 0.5] | 0.4845 | 0.4827 |
| *pure-aggregation prediction from the 12 marginals* | *0.6269* | *0.6269* |

- **The marginal half is pure aggregation.** 0.60-0.62 against a first-principles independent
  prediction of **0.627**. No hidden marginal defect exists; chasing any single marginal is worth
  ~0.01.
- **The joint half is the BIGGER half** (0.70-0.72 > 0.60-0.62). The dominant calo defect is
  structure, not calibration. Most of it (0.66) is second-order, i.e. the correlation matrix.
- **`frac_near_floor` was never fixed — only made marginally invisible.** It is in **7 of the top 12
  |Δρ| pairs on both seeds**, mean Δρ over its 11 pairs **−0.077 / −0.082** vs an all-pair mean of
  −0.025 / −0.033. `logE_mean × frac_near_floor` real −0.112 → gen −0.351; `logE_p90 ×
  frac_near_floor` real **+0.120 → gen −0.079** (the sign inverts). Its paired per-event scatter is
  **1.63x the real spread** — per event it is noise.
- Generated correlations are **systematically too weak** (negative in 39/66 and 41/66 pairs): the
  generator makes event-level quantities too nearly independent.
- **`n_cells` has a conditional bias the marginal cannot see**: r(gen−real, log n_src) = **−0.26 /
  −0.32** while its single-feature AUC is 0.5052. Matches the `n_cells × log_totE` pairwise excess
  of +0.077 (both singles ~0.50). Separate, smaller thread — a GlobalHead `log_n` question.
- Method validated on synthetic arms with known answers first (pure marginal shift → copula 0.489;
  pure correlation change → marginals 0.487, copula 0.711).

## CALO MULTISPECIES v2 (2026-08-25) — pooling costs +0.09 on e±, ALL of it marginal
[Entry](experiment-memory/2026-08-25-multispecies-v2-pooling-rebreaks-the-floor.md) · ckpts
`multispecies_v2_s0/s1` (60k, 2 seeds) · slices `multispecies_v2.npz` (5,025,302 showers / 125.0M
cells) / `multispecies_v2_h5.npz` · jobs 12953664, 12953673, 12953682, 12953717.

Clean A/B: `ms_ele` and `electron_v2` score the **same 660,330 showers over 7,186 events**.

| held-out gate8 | dedicated e± | **pooled** |
|---|---|---|
| seed 0 | 0.8105 | **0.9079** |
| seed 1 | 0.8073 | **0.8852** |

- **The whole regression is MARGINAL; the joint defect is untouched.** Copula-only 0.700/0.724 →
  0.705/0.696 (flat); marginals-only 0.601/0.617 → **0.727/0.723**, with the aggregation prediction
  tracking it (0.627 → 0.769). The two defects are separable and respond to different things.
- **`frac_near_floor` is the responsible feature, and pooling put it back WORSE than v1**:
  v1 0.702 → dedicated v2 **0.5045/0.5461** → pooled v2 **0.7335/0.7448**. In the pooled model it is
  the first greedy pick and worth **0.758 alone**.
- **The 2026-08-24 prediction is CONFIRMED for π and γ**: π± 0.666 → **0.526/0.557**, γ 0.510 →
  **0.512/0.574**. The fragment-fraction explanation is right for those species; e± is where it does
  not survive pooling.
- **Full-calorimeter number, first time measurable**: all 17 classes, gate8 **0.9354/0.8891**, with
  depth **0.9673/0.9539**. Marginal-dominated; greedy order `logE_std` → `depth_mean` →
  `frac_near_floor` → `depth_std`. `depth_mean` entering second is new — it is species-dependent, so
  only a shared model exposes it.
- **Seed spread blew up ~10x**: 0.003 (dedicated e±) → 0.023 (pooled e±) → **0.046 (pooled all)**.
  "v2 needs fewer seeds" holds for single-species only; on the mixture a 0.02 effect is unreadable.
- **CAVEAT — per-species gates are NOT comparable across species.** Features are pooled over an
  event's showers of the selected class, so p (9.9/event) and n (7.5/event) give noisier features
  and weaker gates than e± (91.9/event) for that reason alone. p 0.748/0.701 and n 0.749/0.706 look
  best and almost certainly are not. Within-species comparisons are sound.
- **`n_src` now reports the truth**: 2.67 distinct depositors merged per shower, **60.4% unmerged**.
  Earlier builds printed 24.87, which was the cell count — re-attribution merges materially less
  than that implied.

## CALO ALL-SPECIES v2 CENSUS (2026-08-25) — re-attribution makes γ the LARGEST class
Job 12930683, shard 0, uncapped: **1,668,788 showers / 41.4M cells / 24.83 per shower** (median 11);
depth p95 **1098 mm** vs 146 mm for e± alone — the deep hadronic population the 3D model has never
been trained on.
- γ **410,578** · e− 338,662 · e+ 321,059 · π− 186,480 · π+ 183,381 · p 66,021 · n 48,539 ·
  π0 22,019 · K0L 16,068 · K+ 14,950 · K− 14,080 · μ+ 12,479 · μ− 11,920 · p̄ 9,038 · n̄ 8,982 ·
  other 4,071 · K0S 461.
- The v1 depositor census had γ at **2.3% of energy**; under v2 it is the biggest class, because the
  e± fragments that used to be their own "showers" now book to their photon ancestor.
  **No v1 per-species number is comparable to a v2 one.**
- Three shards uncapped ≈ 5.0M showers / 124M cells ≈ **11 GB** GPU resident at ~84 bytes/point,
  which is why the production slice is built without `--max_per_class`.

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
0. **Floor redesign — atom -> truncated edge, and energies as simplex FRACTIONS** (2026-08-27, above).
   Outranks everything below it: a falsified premise, not a missing refinement. Blocks any further
   `frac_near_floor` work.
0a. **SLICE REBUILD FIRST** (2026-08-27): `(section token, depth-within-section)` + per-detector
   `point_layer` offset. The depth axis cannot currently express the ECAL/HCAL boundary at all.
0b. **Position into the energy mixture** (depth-within-section, interacting with PDG), then a
   PDG-conditioned ATTENTION head — coherence is hadron-only but graded, so route by conditioning,
   never by a hard EM/hadron branch.
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

## CALO CELL-COUNT CAP (2026-09-15) — a real defect; BOTH gate halves moved, the composite did not
[Entry](experiment-memory/2026-09-15-calo-cell-cap-128-was-real-and-gate-flat.md) · jobs 13920087,
13921623 · ckpt `multispecies_v2_s0` · 1 seed (correct: paired A/B, same checkpoint).

`sample_showers` clamped generated cells at a bare `max_cells=128` literal. On `multispecies_v2_h5`
that truncates **2.10% of showers / 17.7% of cells and destroys 6.87% of ALL cells**, hadron-selective
(**pbar 25.3% of showers, 21.3% of cells**; mu±/pi0 exactly 0.00%; e± only 2.9% — which is why every
earlier look, all on the dedicated e± model, found it harmless). `partition=True` then repacks the
full sampled energy into the survivors, so it inflates every per-cell energy feature.

Fix: `n_max_pdg` buffer (same contract as `logE_max_pdg`), bound stays FINITE because `log_n` is an
unbounded mixture. Backward compatible (absent -> inf -> old behaviour). No retrain needed.

| | cap 128 | cap 4096 |
|---|---|---|
| `logE_mean` / `logE_p90` AUC | 0.629 / 0.710 | **0.524 / 0.610** |
| `frac_near_floor` / `cells_per_src` | 0.561 / 0.570 | **0.508 / 0.512** |
| W/sigma `cells_per_shower` | 0.0489 | **0.0104** (4.7x) |
| **marginals only** | 0.8421 | **0.7432** |
| **copula only** | 0.8453 | **0.7760** |
| composite gate8 / depth | 0.9328 / 0.9650 | 0.9356 / 0.9586 |

- **The pre-registered prediction "copula stays FLAT" was FALSIFIED** — it fell 0.069. Truncation acts
  selectively on LARGE showers, so it injects `n_cells` x energy correlation; a defect on a
  subpopulation cannot be purely marginal. Do not re-derive this.
- **The composite gate is a poor instrument near saturation**: signal removed from both halves, gate
  flat, because at 0.96 it has redundant paths. Quote the decomposition, not `event_gate_auc`.
- Marginal half is STILL pure aggregation in both arms (0.844 vs 0.842; 0.749 vs 0.743).
- Fix restored the expected ordering: copula (0.776) > marginals (0.743), which the pooled model had lost.
- **Open discrepancy**: `frac_near_floor` reads 0.5605 here where this file records 0.7335/0.7448 for
  pooled v2. Composite control matched (0.9328 vs 0.9354) so it is not drift. Likely greedy-selection
  vs single-feature AUC — **NOT verified**.

## CALO CELL GRID (2026-09-15) — ECAL endcap SOLVED and analytic; HCAL open
[Entry](experiment-memory/2026-09-15-calo-cell-grid-ecal-solved-hcal-open.md) ·
`scripts/calo_cell_grid_derive.py` · job 13921623 · 5 shards x 7k events = 190M cell-hits · real data.

**Cell ids are genuinely absent, not dropped by us** — verified against our shard, the README schema,
the SOURCE parquet via HF datasets-server, and two other calo configs; the HF repo holds no geometry
file. EDM4HEP `SimCalorimeterHit` has four members (`cellID`, `energy`, `position`, `contributions`)
and ColliderML published three: **`cellID` is the one dropped in conversion**. Public ODD ships the
tracker only. Contrast: `tracker_hits` DOES carry `volume_id/layer_id/surface_id`.

> **GEOMETRY IS NOW KNOWN, NOT INFERRED — the ODD calorimeter XML exists** (2026-09-15):
> `github.com/OpenDataDetector/OpenDataDetector`, `xml/detectors/Calorimeter{ECal,HCal}.xml` +
> `xml/OpenDataDetectorEnvelopes.xml`. **Every derived number confirmed**: `ECal_cell_size = 5.1*mm`,
> `HCal_cell_size = 30*mm`, `repeat="48"` / `repeat="36"`, slice stacks summing to **5.05** / **51.0 mm**,
> `*_symmetry = 16` for all four, and envelope `rotation z = 90 - 180/16 = 78.75 deg` = **11.25 deg**
> mod the 22.5 deg face spacing — the exact offset the phase scan found.
> **The slides' "30 HCAL layers" was WRONG and our measured 36 was right** (XML: `repeat="36"`).
> The dropped cellID is `system:8,barrel:3,module:4,stave:1,layer:6,slice:5,x:32:-16,z:-16` —
> `module:4` = 16 modules, corroborating the symmetry.
> **Earlier claim in this file that the calo geometry is unpublished was WRONG** — I searched
> `acts-project/OpenDataDetector` and `OpenDataDetector/ColliderML`, never
> `OpenDataDetector/OpenDataDetector`. Build `snap_to_cell` from the XML constants and keep the fit
> as the acceptance test.
> [entry](experiment-memory/2026-09-15-calo-geometry-CONFIRMED-from-ODD-xml.md)

> **EXTERNALLY VALIDATED + SYMMETRY CORRECTED 2026-09-15 (ODD slides, user-supplied):** ECAL
> **5.1 mm square cells** and **48 sampling layers** confirmed exactly against our 5.09998 / 48 —
> the first external check this derivation has had, and it retires the 5.0900 mode for good.
> **But the symmetry is 16-FOLD, not 32**: the slides' "hexadecagon" prompted a sector PHASE-OFFSET
> scan, which spikes at **11.25 deg** (face boundaries at `11.25 + k*22.5`) — ECAL N=16 on-grid
> **0.9999**, HCAL **0.9344** (better than 32's 0.8768). Our scan swept N and never swept the phase,
> so it could only ever find MULTIPLES of the truth and reported them at R=1.0000. **Every "32
> sectors" in this file and in `calo_cell_grid_v*.json` is an alias.** HCAL 30 mm cells confirmed;
> HCAL layer count 36 (measured, unambiguous) vs the slides' 30 is an OPEN mismatch.
> [entry](experiment-memory/2026-09-15-calo-grid-validated-against-ODD-slides-16-fold.md)

> **CONFIRMED AT SCALE (job 13926563, v3):** pitch **5.09998 mm, IQR 0.00000** at every one of the
> 12 ECAL layer-fits, **on-grid 0.9993-1.0000 with radial banding OFF**, and an angle pattern
> identical at every layer (`-45x4, -22.5x8, 0x8, +22.5x8, +45x4`, period-8 in sector index).
> **ECAL ENDCAP IS CLOSED** — 32 sectors, per-sector angle + origin, pitch 5.1 mm, depth on the
> 48-plane ladder, NO radial index. HCAL: instability is confounded with sparsity at depth, but
> layer 0 is STABLE (IQR 0.0056) and still only 0.88 on-grid, so the lattice model is genuinely
> incomplete there. See [v3](experiment-memory/2026-09-15-calo-grid-v3-ecal-closed-hcal-sparsity-confound.md).

> **SUPERSEDED THE SAME DAY — the pitch below is wrong. See
> [pitch correction](experiment-memory/2026-09-15-calo-ecal-pitch-is-5.1-exactly-supersedes.md).**
> ECAL pitch is **5.1 mm EXACTLY** (refined 5.09998), not 5.0900; `pitch_from_nn` quantised it to a
> 0.02 mm histogram bin. With the right constant, **one origin per sector needs NO radial index** and
> on-grid is **1.0000 with median residual 0.0011 mm** — as exact as the longitudinal axis. The
> `--r_band 200` in job 13921623 was silently absorbing a 0.0100 mm pitch error that slips a full
> pitch every ~510 cells.

**ECAL endcap (9/11): pitch 5.0900 mm, 32 phi sectors, CONSTANT across layers 0/4/12/20/30/40,
on-grid 0.9989-0.9997, median residual 0.080-0.088 mm.** Both endcaps agree to 4 dp.
So the projection is **ANALYTIC, not a LUT** — `floor(phi/(2pi/32))` -> rotate -> round to the pitch
-> snap depth to the 48-plane ladder. That overturns the implication of gap 2a's ~2e7 vocabulary.

**HCAL endcap (12/14): pitch 29.99 mm, 32 sectors (shared symmetry), but on-grid only 0.894-0.954
with p95 residual 0.43-0.79 mm** vs ECAL's 0.19. NOT a subsampling artifact (HCAL was never
subsampled). Global R 0.23 vs ECAL 0.044 is unexplained by a uniform rotated-square model.
**UPDATED same day**: projective towers FALSIFIED (pitch is 29.990 in every radial band from r=362
to 3047). Pitch was never the problem — refined it is 29.9930 vs the mode's 29.9900. The real finding
is that **module ORIENTATION varies by sector** — 0, 22.5 and 45 deg all appear, with 45 deg on
sectors 4/12/20 (= 4 mod 8) and 0 deg on 8/16 (= 0 mod 8), so the true symmetry is plausibly
**8-fold with alternating orientations and 32 is an alias**. Even with angle fitted and pitch
refined, R caps at ~0.97 and on-grid is sector-dependent (0.67-0.96): HCAL is NOT a single rotated
square lattice.

**METHOD WARNING 2 — `pitch_from_nn` takes a histogram MODE and so quantises the answer to the bin
width.** It produced two separate wrong results on 2026-09-15: the `5.09*sqrt2` diagonal (below), and
a 0.0100 mm slip that forced every ECAL fit to use radial bands. The mode is acceptable as a SEED;
the value must then be refined by maximising circular concentration. **Fix before reusing.**

**METHOD WARNING — random subsampling breaks lattice fitting.** `--max_pts` used `rng.choice` and
returned `5.09*sqrt2` then `5.09*sqrt5` as retention fell (1.00 -> 5.09, 0.50 -> 7.21, 0.25 -> 11.41);
job 13920087 lost 2 of 3 ECAL layers to it and layer 30 survived by luck. `annulus_subsample` keeps
whole radial annuli — annuli NOT phi wedges, since wedges would impose the very angular period the
sector scan measures. NN mass at the true pitch 0.234 -> 0.98.

## CALO CELL PROJECTION LANDED (2026-09-15) — gate 0.9357 -> 0.9458, and the cost is real
[Entry](experiment-memory/2026-09-15-calo-cell-projection-first-gate-and-a-preexisting-wrap-bug.md) ·
job 13929818 · `src/genpu/calo_cells.py` + `calo_metrics.py --snap_cells` · ckpt `multispecies_v2_s0`.

**PIPELINE gap #1 is closed for the projection half.** Generated points now land on real ODD cells:
42.1M points -> 40.6M cells (3.61% merged), energy conserved x1.000001. On real cells the map snaps
0.99698 within 10 um and recovers the same cell id 0.9925.

**`event_gate_auc` 0.9357 -> 0.9458** (the only comparable number: the decomposition's `full` is 12
features on the plain arm and 10 on snap, since depth is dropped — reading 0.9638 -> 0.9486 as a win
is a feature-count artifact). Cost is **entirely ENERGY** — `logE_p90` +0.106, `logE_mean` +0.078,
`logE_std` +0.053 — while width and position marginals are untouched, which is what a correct
projection should do. **Read it as the projection EXPOSING a defect the continuous representation
hid**: a cloud can put two points 0.1 mm apart and be scored as two cells where the detector reports
one. 0.9357 was flattering.

> **CORRECTED 2026-09-15 — the direction was BACKWARDS. The generator OVER-concentrates.**
> 7.2% was the wrong reference: it counts raw CONTRIBUTIONS sharing a cell, which v2 re-attribution
> already merges away, so the slice stores one row per DISTINCT cell and `log_n` targets distinct
> cells. Measured on 200k real showers / 5.0M rows through the identical snap path: **real collision
> 0.0003 vs generated 0.0361 — 120x.** The generator places 3.6% of its points close enough that the
> detector reads one channel.
> [correction](experiment-memory/2026-09-15-calo-generator-OVER-concentrates-corrects-under-merge.md)

~~**The generator UNDER-merges**: co-occupancy 3.09% against the 7.2% within-shower sharing.~~

> **METHOD WARNING — a PRE-EXISTING missing `wrap_pi` was corrupting every width/d_phi number.**
> `calo_metrics.py` differenced `gen_phi - p_phi` unwrapped while the real side has always wrapped
> (`build_calo_slice_v2.py:292`). Not a no-op, because the helix anchor displaces pion CORES by
> ~1.5 rad. Plain-arm corrections: `width_std` **0.5016 -> 0.6953** (it was never at chance),
> `width_mean` 0.5618 -> 0.5100, `shower_width` W 0.0344 -> 0.0505. **Any argument resting on
> `width_std` being indistinguishable must be re-read.** Under `--snap_cells` the same bug read as
> `width_std` 0.9565 and looked exactly like "projection destroys shower width"; the tell was that
> `d_eta` was untouched while `d_phi` blew up 12x — one of two symmetric coordinates breaking alone
> is a coordinate bug, not physics.

## Open threads
- **HCAL endcap cell geometry** (2026-09-15, v3) — pitch is settled (29.9894, IQR 0.0056 at layer 0)
  and projective towers are falsified, yet **layer 0 still leaves 12% of cells off-grid on adequate
  statistics**, concentrated in specific sectors. Work at LAYER 0 ONLY — deeper layers' instability
  is confounded with sparsity (distinct cells fall 13,463 -> 5,176 with depth) and cannot settle
  anything. 28.5% of calo energy.
- ~~Build `snap_to_cell` for the ECAL endcap~~ — **DONE 2026-09-15**, and for the WHOLE calorimeter:
  `src/genpu/calo_cells.py`, **99.698%** of 4.29M real cell-hits within 10 um, ECAL endcap ids
  bijective. **PIPELINE gap #2 (barrel unreachable) is RETRACTED** — radius was the wrong coordinate.
  [entry](experiment-memory/2026-09-15-calo-snap-to-cell-built-and-barrel-gap2-RETRACTED.md)
- ~~Wire `snap_cells` in~~ **DONE** (`--snap_cells`, job 13929818). Remaining: a cell-level metric
  proper (occupancy, energy-per-cell against real cells).
- **Collisions: core-localised, and NOT a density excess** (2026-09-16, job 13976736) — generated NN
  spacing is 2.14x REAL (0.00674 vs 0.00314), so the cloud is sparser yet collides 120x more:
  independent per-cell draws, not core density. Density tuning RULED OUT. Caveat: 2-D statistic,
  real cells stack in depth; per-layer 3-D NN would split that.
  [entry](experiment-memory/2026-09-16-calo-nn-spacing-generated-is-SPARSER.md)
- ~~**The generator OVER-concentrates**~~ (superseded above; kept for history) — probe where the
  collisions sit; the shower CORE is the candidate. Needs generated cells, so a GPU job. If it is the
  core, `PointCFM` controls that directly and it is testable by resampling, no retrain.
  Do NOT "fix" the merge: merging is the correct response to two deposits in one channel, the error
  is upstream in placing them there.
- **Boundary assignment** — the residual (HCAL endcap 2.5%, barrels 0.2-0.5%) is cells near a
  face/stave boundary assigned to the neighbour, not a lattice error (in-face residual is 1e-5 mm).
  Fix by testing both adjacent faces.
- **Fix the sector scan to sweep PHASE with order** (2026-09-15) — a symmetry search with a fixed
  origin finds only multiples of the truth and reports them with full confidence. Re-run afterwards;
  the `best_sectors` field of both stored grid JSONs is wrong.
- ~~Ask ColliderML for the calorimeter XML~~ — **RESOLVED 2026-09-15**, the XML is public at
  `OpenDataDetector/OpenDataDetector`. Asking them to restore `cellID` to the calo tables is still
  worth it (it is one uint64 they already have), but the GEOMETRY question is closed.
- ~~Revisit PIPELINE gap #2 (barrel cells unrecoverable)~~ — **RESOLVED 2026-09-15, gap #2 retracted.**
  The barrel snaps at 0.9982 (ECAL) / 0.9948 (HCAL) within 10 um.
- **The calo floor's COUPLING, not its marginal** (2026-08-25). `--floor_n_buckets` conditioned the
  floor Bernoulli on a cell-count embedding and left the mechanism untouched (0.003); the missing
  conditioning is on the shower's ENERGY scale, which is what `logE_mean` / `logE_p90` measure. The
  `EnergyHead` already receives `glob_std` — whether the floor branch actually sees it is the first
  thing to check. Note "cells need to know about each other" was falsified on a *dispersion*
  argument (D_floor real 1.035), which says nothing about cross-feature correlation, so that door is
  not closed by the 2026-08-16 result.
- **The shared calo head loses something the dedicated e± head had** (2026-08-25) — measured cost
  +0.08-0.10 gate8, entirely marginal, entirely `frac_near_floor`. Both use the same `EnergyHead`;
  what differs is that the shared trunk's gradients now come mostly from the deep hadronic classes.
  `--separate_trunks` failed on v1, but it has never been tested where the mixture is this
  heterogeneous — a different experiment from the one that failed. The narrower version:
  `logE_max_pdg` already exists as a per-class buffer, so per-class energy bounds are machinery that
  is present, and only the FLOOR branch is class-blind.
- **Per-species gates need a multiplicity-matched variant** — the current one confounds shower
  multiplicity with model quality, which makes p/n look like the best-modelled species.
- **Turning-point anchor is a proxy, and 2026-08-16 showed there is NO better one available**: 63% of
  pion / 72% of e± showers are energy booked to a parent that never reached the calo, and those
  particles leave zero tracker hits 81% / 93% of the time — so Phase 4 cannot own that population.
  Its ~2.5x is the ceiling unless something other than the tracker supplies the stopping point.
  (Also measured: the vacuum turning point's φ is **uncorrelated** with the real outermost hit's φ —
  |Δφ| median 1.567 rad ≈ π/2, the uniform value — so its 2.4x gain is not φ agreement with the
  actual track.)
- **Rotated module frame for v3** (2026-09-07, above) — two in-plane coordinates instead of three
  global ones. Highest-value tracker item now: it is exact-by-construction, removes a head that
  predicts a constant for 47.9% of hits, and is a precondition for the reco-level (ACTS) eval that
  decides v3 vs v4. Separate, untested sub-question: the position heads never see the module
  embedding at all.
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
