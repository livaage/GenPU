# Tracker head — full-event gate: what we tried

**Status:** honest full-event generator (all species, all fragments, **no truth hit-counts** — the
count head produces them). Two-sample event gate (MLP on per-event features, rank-AUC, 0.5 = perfect):
- **base-8 gate: 0.77** · rich/tail-aware gate: 0.86 · count head matches n_hits (median 2, mean 4.53).

## ★ NEW LEAD LEVER (2026-08): surface-local coordinates, not layer residuals

The drift lives in the **representation**, and it is fixable by changing what the AR predicts.
Re-read the reference paper (below): it does **not** predict residuals — each hit is an **absolute
local coordinate on a discretely-chosen detector module** (`geometry/module ID + local X,Y`), plus
carried momentum. We have been on `layer_class(48) + per-layer-standardized residual` since day 1
(git history confirms: `abspos` was only ever an *input*; `use_helix` z is still a deviation; v2's
"learned position" reuses the residual head verbatim). **We never adopted the paper's representation.**

**Data check on ColliderML (1 shard, 4000 events, 6.7M hits — [scripts/check_surface_granularity.py](scripts/check_surface_granularity.py)):**
`surface_id` is present in the source (dropped at preprocessing). The module key is the
`(volume,layer,surface)` composite (surface_id is reused across layers):

| discrete anchor | # units | within-unit `std_r` | within-unit in-plane spread |
|---|---|---|---|
| `(volume,layer)` — **current** | **48** | **60.7 mm** (p90 131) | 447 mm (p90 898) |
| `(volume,layer,surface)` — **paper module** | **~18k** | **2.1 mm** (p90 44) | 22 mm (p90 47) |

The radial coordinate our residual head must span drops from **±60 mm → ±2 mm** (~28×). The discrete
surface token pins the global position; the continuous part becomes a **small bounded local offset
that cannot compound into a hundred-mm global drift**. This is structurally why the paper doesn't
drift — global "where" is in the discrete module choice, not a continuous coordinate. **~366 hits/
module** → ample statistics to learn each local distribution.

**Caveats (honest):** (1) vocabulary grows 48 → ~18k classes (fixed geometry, bounded; the paper
handles thousands with a shared dictionary) — the drift *moves* into a discrete next-surface
prediction, but geometry constrains valid transitions so errors are "adjacent surface," not smooth
outward creep. (2) We have **no per-hit momentum** (source lacks it) — the paper credits momentum
for pinning the surface sequence, so this is "paper representation *minus* the momentum guide";
cost of that omission is unknown (but see the transition-entropy result below — geometry alone
already pins it hard).

**Pre-build tests — BOTH PASS ([scripts/tracker_surface_prebuild.py](scripts/tracker_surface_prebuild.py), 3000 events, 5M hits):**
- **Local coords are bounded everywhere.** Within-module spread broken down by volume: `std_r`
  0.6–45 mm, in-plane 5–50 mm; **0/18,824 modules exceed 100 mm.** No per-volume frame needed —
  raw `(x−x̄, y−ȳ, z−z̄)` local coords work detector-wide. (The ~500 mm p99 tail reported in an
  earlier draft was a **script bug**: an arithmetic module key `vol*1e6+lay*1000+surf` collided
  because layer_id/surface_id exceed 1000. Corrected with `np.unique` over rows. Do not re-litigate.)
- **Module sequence is strongly learnable.** Over 1.08M tracks / 3.9M transitions: `H(next)` = 13.54
  bits (~11.9k eff. modules) but **`H(next|current)` = 5.13 bits (~35 eff. modules)** — 8.4 bits of
  info gain from the current module alone; greedy top-1 = 20.6%. This is an **upper bound** on the
  AR's uncertainty (it also sees history + kinematics), so next-module prediction is easier than
  standard LM next-token and far better-posed than hitting a continuous ±60 mm residual. Minor note:
  7.3% self-transitions (two hits on one surface) → the module head must be allowed to repeat.

**Implementation sketch (representation change, NOT a from-scratch redesign — reuses the v1/v2
decoder, conditioning, count/stop machinery):**
1. **Preprocessing**: keep `surface_id` through to stage2; build a global module vocabulary
   `(volume,layer,surface) → module_index` (fixed detector geometry → a stable id table). Store
   per-module local frame: origin (mean x,y,z) + optionally 2 in-plane axes, so `local (u,v) =
   projection of (hit − origin)` is small & bounded. Fall back to `(x−x̄, z−z̄)` if axes are overkill.
2. **Model**: replace `layer_head (48)` with `module_head (|V|)`; replace the `r,φ` residual heads
   with `u,v` local-coordinate heads (bin the same 512-way, but now over a ±tens-of-mm range).
   `z` folds into the local frame. `abspos` input feeds the reconstructed global position as before.
3. **Generation**: sample module → look up frame → sample `(u,v)` → map back to global `(x,y,z)`.
   Everything downstream (count head, stop head, gate) is unchanged.
4. **First test (cheap, before full build)** — ✅ DONE, both pass (see pre-build tests above):
   local coords bounded ≤50 mm detector-wide; `H(next|current)`=5.13 bits (~35 eff. modules).

This is higher-value than every "untried lever" below and should be tried first. Pre-build
viability is confirmed.

**Preprocessing IMPLEMENTED + verified (prototype scale).** Self-contained, reads raw source
(stage2 already dropped `surface_id`+`x,y`), does NOT touch the shared stage2/calo/count pipeline:
- [scripts/build_module_geometry.py](scripts/build_module_geometry.py) → `module_geometry.npz`:
  the module vocabulary (18,824 modules) + per-module local frame (mean/std of x,y,z,time). The
  surface-granular analogue of `detector_geometry.LAYER_GEOMETRY`.
- [src/genpu/module_geometry.py](src/genpu/module_geometry.py): loader/lookup (`to_index`,
  `local_residual`, `to_physical`) — the module analogue of `detector_geometry.py`.
- [scripts/build_tracker_surface_slice.py](scripts/build_tracker_surface_slice.py): per-particle
  slice `[module_index, x_res, y_res, z_res, time_res]`, same 7-feature conditioning contract as
  `build_tracker_slice.py`, sorted inner→outer.
- **Verified**: local-frame round-trip max err 7.6e-6 mm; module↔layer_class consistent; local
  residuals ~N(0,1), p99 |x,y,z|<2 (no clipping); 0 unseen-module hits; each layer fans to 60–3360
  surfaces.
- **Full-scale artifacts BUILT** (job 12068460, 14 min CPU):
  `module_geometry.npz` (6 shards, 73M hits) and `tracker_slice/surface_pion.npz` (shards 0-2,
  1.62M pions, 16.4M hits). **Vocabulary saturates at 18,824 modules** (same as the 1500-event
  proto → detector fully covered, 0 modules <20 hits, 0 unseen-module hits in the slice). Local
  residual std x/y≈1.0, z=0.70, t=0.05 for pions; p99<2, no clipping.
- **v3 MODEL head DRAFTED + CPU-verified** ([src/genpu/models/tracker_module_ar.py](src/genpu/models/tracker_module_ar.py)):
  HIERARCHICAL module head — `layer_head(48)` + layer-conditioned **masked** `surface_head(3360)`
  (better-conditioned than a flat 18,824-way softmax, reuses the proven layer head, and is *cheaper*:
  ~5.5× fewer output params than flat) — plus local Cartesian `x/y/z`(512-bin)+`time`(64-bin) heads
  over the bounded module frame. Decoder/conditioning mirror v1. Two-stage generation (layer→mask→
  surface→local→`to_physical`). 1.38M params (v1 ~1.0M). Hierarchy/mask/index-maps from
  `ModuleGeometry.build_hierarchy()`. Test [scripts/test_tracker_module_ar.py](scripts/test_tracker_module_ar.py):
  init losses correct (layer CE≈ln48), surface masking verified, generation always a clean
  (layer,slot)→module round-trip. Design note: pion local time std≈0.05 → time near-constant per module.
- **Trained + gated (RESULT, 2026-08).** v3 wrapper [tracker_module_model.py](src/genpu/models/tracker_module_model.py),
  trainer [train_tracker_module.py](scripts/train_tracker_module.py), gate adapter
  [tracker_honest_gate_module.py](scripts/tracker_honest_gate_module.py). Trained 60k steps on
  `surface_pion.npz` (ckpt `surface_pion_v3/checkpoint_060000`); losses layer 3.67→0.35, surf 5.6→1.7,
  cont 5.24→3.2.
  - **MATCHED comparison (same pion slice, current code, 60k steps, TRUTH-count gate, both plain):**
    v1 layer-residual **AUC 0.9996** → v3 surface-local **AUC 0.80**. The representation eliminates the
    gross outward misplacement: r_mean |Δ|/σ **1.86→0.36**, layer_mean **1.84→0.23**, frac_inner
    **1.83→0.06**, z_std **1.45→0.06**. (Plain v1-pion truth-count was never measured before; 0.9996
    shows pion tracks are where the drift is worst — pions are the long coherent tracks.)
  - **Honest caveats**: (1) 0.80 is NOT below the best documented v1 0.77 — but that's apples-to-oranges
    (0.77 = multispecies + matched count head; this = pion-only, truth-count, plain v3, noisy surf head).
    (2) A **residual mild outward bias remains** (r_mean 325 vs 303, layer_mean 19.2 vs 18.6, |Δ|/σ~0.3) —
    what the 0.80 keys on; likely the *module-SELECTION sequence* drifting slightly outward (discrete-anchor
    version of the old drift). Next lever. (3) The multispecies **count head undercounts pions** (gen median
    1 vs real 10) → the HONEST (count-driven) pion gate is meaningless until a pion/per-species count head
    exists; orthogonal to the tracker.
  - **Next levers for v3**: vertex-seed the module sequence; more steps / bigger surf head; all-species
    slice + per-species count head; then the honest full-event gate vs the 0.77 deliverable.

## What actually distinguishes generated events from real

The classifier keys on **outer/forward spatial TAILS**, not the bulk. Single-feature AUCs:

| feature | 1-feat AUC | meaning |
|---|---|---|
| `absz_p90` | 0.715 | hits reach too far in \|z\| |
| `r_p90` | 0.672 | hits reach too far in radius |
| `z_std` | 0.618 | z-spread too wide |
| `r_mean` | 0.536 | (weak) |
| `frac_inner` | 0.524 | (weak) |

Root mechanism = **"stretched track"**: the AR generates hits inner→outer, one at a time, and the
**later hits drift systematically outward** (per-step `gen r_resid` climbs 0 → +0.22 by step 8 while
real stays ~flat; gen_std grows 0.99→1.07). Classic autoregressive error-compounding. The *seed*
being too inner is a real but **weak** contributor (it only moves `r_mean`/`frac_inner`).

## What we tried to move 0.77 — all failed

| attempt | targeted | result | why it failed |
|---|---|---|---|
| **lower sampling temperature** | tails | gate *worse* (0.857→0.865) | sharpening under-disperses the already-good features; doesn't touch the z-tail |
| **count-routed single-hit placement** (place debris at vertex / neutral flight instead of AR) | seed/debris hits | capped at baseline (0.888) | AR is jointly-consistent; bolt-on placement fixes marginals, breaks the joint. Confirmed count head is ~perfect; built correct neutral-flight physics |
| **vertex anchoring** — learnable vr→layer bias + seed-loss up-weight (2 retrains: all-position bias, then seed-only) | seed placed too inner | seed unchanged (layer-7 ~18% vs real 4.9%), gate unchanged | additive logit bias can't overpower the layer_head's entrenched "start at beampipe" default; **and the seed was never the gate lever anyway** |
| **input jitter** (±N bins on fed-back history, scheduled-sampling proxy) | the drift | drift unchanged (still climbs to +0.25), gate move was run-to-run noise | drift is a *systematic directional* bias; *zero-mean* jitter can't correct it |
| **momentum-estimate feature** (`use_mom_feat`: feed helix-expected z `z_guide(r)=vz+(r−vr)·sinh η` per hit as an INPUT feature, output unchanged) | the drift (z half) | gate 0.79 (within noise of 0.77); drift reduced **~10%** (step-8 +0.19 vs +0.21), variance growth smaller (1.046 vs 1.069); z_resid now perfect | **First lever to move the drift at all** — but only ~10%. We anchored **z only**; z_resid was already fine, and the gate-driving **r/radial drift** got no transverse anchor |

## Key facts learned (don't re-litigate)

- **The seed is NOT the gate lever** — the outward *drift of later hits* is. Fix the drift, not the seed.
- **The count head is excellent** — n_hits shape matches truth almost exactly. Not a lever.
- **The gate is noisy across from-scratch retrains** (~0.05: e.g. 0.77 vs 0.86 for near-equivalent
  models). Judge drift fixes with the **per-step `r_resid` drift table** (variance-free), not the gate.
- The neutral single-hit displacement physics is worked out (sample real (Δr,Δz) per species, clip to
  detector) — reusable when a dedicated neutral model is wanted.

## Reference paper (arXiv:2512.24254, Novak & Keršķevan — GPT tracker sim)

Read in full (the auto-summary hallucinated "scheduled sampling / teacher forcing" — those are
NOT in the paper). What they actually do, and why they don't have our drift:

- **They carry the particle's momentum through the sequence.** Each hit has 7 features incl. the
  **3 momentum components *after* that hit**; the sequence is seeded with a virtual start hit at the
  beamspot with the **initial momentum**. So the model tracks the *evolving kinematic state* — the
  momentum kinematically pins the trajectory, so it can't drift. We predict *positions only*, no
  kinematic memory → free to drift. This is the whole difference, and it's implicit (no training trick).
- **They grade on a more forgiving metric.** "Comparable to full simulation" = **ACTS reconstruction
  efficiency** (seeding + tracking) + 1-D marginal distributions — NOT a classifier two-sample gate.
  Reco is robust to small hit perturbations; they even report up to 5% fluctuations in global coords.
  So **our 0.77 two-sample gate is a strictly harder bar** than their headline. Their method isn't
  perfect either — for hadrons they admit "the momentum modelling is not sufficient... too large pT".

**Data check (decisive):** the ColliderML source `tracker_hits` schema is
`event_id, x, y, z, true_x/y/z, time, particle_id, detector, volume_id, layer_id, surface_id` —
**NO per-hit momentum.** So we cannot copy their method directly. We approximated it with the
initial-momentum helix (`use_mom_feat`, z half) — see the tried table.

**Smearing check (ruled out):** measured vs true `z_r_resid` ratio = **0.979** (per-hit smearing
median ~0.017mm transverse, ~0). The excess scatter is **real trajectory mismodeling, not measurement
smearing** — training on `true_x/y/z` + parametric smearing would change nothing.

## Untried levers (all attack the *drift*)

0. **★ Surface-local coordinates** — see the NEW LEAD LEVER section at the top. Replace
   `layer(48)+residual` with `module(~18k)+local(u,v)`; shrinks the drift-prone coordinate ±60mm→
   ±2mm. Highest-value, and a representation change not a redesign. **Try this first.**
1. **Transverse/bending momentum anchor** — the completion of `use_mom_feat`: also feed the helix-
   expected *radial/φ* crossing per layer (needs curvature = B-field × charge / pT). Targets the
   `r_p90` discriminator the z-only anchor couldn't reach. The faithful version of the paper's full
   momentum-carrying. Moderate rework, uncertain payoff (z-half was only ~10%).
2. **Proper (2-pass) scheduled sampling** — feed the model its *own* compounding predictions in
   training (not zero-mean noise). Faithful version of the failed jitter idea.
3. **Non-autoregressive / one-shot track generation** — generate all hits jointly, no sequential
   compounding (could reuse the calo flow-matching head).
4. **Reconstruction-level eval (ACTS)** — measure our generator on the *field-standard* metric. Large,
   deferred effort, but it would tell us whether the drift the two-sample gate obsesses over even
   matters physically. If reco efficiency is fine, 0.77-on-a-strict-gate was never the right bar.

## Orthogonal (does NOT address 0.77)

- **Material cascade** (generate secondaries from primaries for Mode-2 / raw-Pythia input): every
  particle still flows through the same AR with the same drift. The cascade adds a *capability*, not
  quality; the end-to-end Mode-2 gate will likely sit **at or slightly above** 0.77, never below.

## Where we are (bottom line)

- Tracker = honest full-event generator at **base-8 gate 0.77** / rich 0.86, count head ~perfect. A
  real, usable deliverable.
- Every cheap-to-moderate drift lever tried (temperature, placement, vertex anchoring, jitter, momentum-
  z feature). Only the momentum-z feature moved the drift, and only ~10%; the gate never went below the
  original 0.77 (all within run-to-run noise).
- The drift is **radial and systematic** and has proven robust to everything short of a full transverse
  trajectory anchor (untried lever #1) or a non-AR redesign (#3).
- **Recommendation:** bank the tracker at 0.77 and build the material cascade (higher-value, Mode-2
  capability). Revisit the drift only via lever #1 (transverse anchor) or #4 (reco-level eval to check
  it even matters) if the two-sample gate becomes the binding constraint.
- Model code: `use_mom_feat` / `use_helix` / `seed_weight` / `jitter` flags live in
  `src/genpu/models/tracker_ar.py` + `scripts/train_tracker.py`. Best honest checkpoint = the original
  `multispecies_vertex_512bin/checkpoint_070000` (0.77); `multispecies_momfeat` is equivalent (0.79) and
  physically cleaner (z_resid perfect).
