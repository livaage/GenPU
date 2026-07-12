# Tracker v2 — state-carrying autoregressive tracker (design doc)

## 1. Why (from our diagnostics)

The current tracker ([tracker_ar.py](src/genpu/models/tracker_ar.py)) predicts, per hit, a
**per-layer-standardized position residual** (binned), autoregressively, conditioned on truth
`n_hits`. We mapped its ceiling from every angle:

- **Coherence spikes undershot** (`r_mono=1`, `z_r_resid≈0`, `dr_std≈0`): gen tracks are slightly
  less coherent than real.
- **Quality persistence under-propagated**: real first-half↔second-half roughness correlation
  = **0.70**, gen = **0.58** (~7σ gap). The model *knows* p_T (conditioning) so it modulates
  scatter, but it also injects a track-*independent* per-hit noise floor that decorrelates quality.
- **Root cause**: the track's quality lives in a **track-level latent (its trajectory /
  momentum)**, and the current output (residual-from-layer-mean) encodes *nothing about this
  track*. The model only propagates the trajectory implicitly (attention + abspos), weakly.
- We cheat with **truth `n_hits`** — no existence/length model.

Two "structural" attempts confirmed the wall: helix-z (fixed analytic z) matched `z_r_resid` but
didn't move the aggregate + side effects; temperature made it worse. The remaining lever is an
**explicit, carried, updated trajectory state** — which is exactly the paper's idea.

## 2. The paper — Novak & Keršan, arXiv:2512.24254 (Dec 2025), ODD

GPT-like decoder-only (nanoGPT, 8×8, CE on tokens), fully generative "to ensure full correlations
between hits." Per hit = **7 tokenized features**: particle ID, geometry (module) ID, two *local*
surface coordinates, and **the 3-momentum (px,py,pz) after the hit**. Sequence = a virtual
**start hit** (initial momentum + beamspot) → hits IP-outward → a virtual **end token** when the
particle leaves the tracker. Continuous features rounded to 2 dp for a finite vocabulary.

The load-bearing ideas for us:
- **Momentum-after-hit = the updatable state.** It evolves via scattering/energy-loss; each hit's
  position is generated *consistent with the current momentum* → one trajectory → coherence, and
  quality (the momentum) persists by construction.
- **End token = stopping** → no truth-`n_hits` cheat (the count/existence frontier, for free).

**Caveat (important):** their result ("comparable to full sim") is on **80–85 GeV muons, narrow η,
no secondaries**. Ours is far harder — soft pions, wide η, multi-species, material secondaries.
Their success does not transfer for free.

## 3. Data reality (checked against the source schema)

`tracker_hits`: `event_id, x,y,z, true_x,true_y,true_z, time, particle_id, detector, volume_id,
layer_id, surface_id`.
`particles`: `…, px,py,pz, perigee_d0, perigee_z0, vx,vy,vz, …`.

Consequences:
- **No per-hit momentum.** The paper's core feature is NOT stored — we cannot supervise a physical
  momentum state directly.
- BUT we have three things the current tracker ignores:
  1. **`true_x/y/z`** — the un-smeared hit position. `measured − true` = the *physical* measurement
     smearing. This lets us **separate physical smearing (real, in the data) from the model's extra
     noise**, and gives a *coherent* target (the true trajectory).
  2. **`surface_id`** — module-level geometry (thousands), finer than our 48 `layer_id` (≈ the
     paper's geometry ID).
  3. **`perigee_d0, perigee_z0` + initial `p`** — the true initial track parameters, usable as the
     seed state / conditioning (we currently use only p_T,η,vr,vz).

## 4. Design options (increasing scope)

### Option A — decompose `measured = true trajectory + smearing` — RULED OUT (measured)
The smearing check ([tracker_smear_check.py](scripts/tracker_smear_check.py)) shows physical
smearing is **negligible** (median transverse 0.017, z 0.001 vs a 100s–1000s scale), and
z_r_resid(true)/z_r_resid(measured) = 0.98 — the measured hits ARE the true trajectory. There is
nothing to decompose. It also reframed z_r_resid: it's dominated by **trajectory curvature** (a
linear z-vs-r fit is a poor model), not scatter — so for v2 use a helix/quadratic-fit coherence
metric. And true r_mono = **1.0000** → the r_mono ties are 100% a binning artifact, confirmed at
the data level. NET: the coherence spikes are real (smearing≈0), so the ONLY reason gen misses them
is the model's per-hit sampling floor -> Option B is the right and only lever.

### Option B — state-carrying AR (the paper's idea, adapted to our data) — RECOMMENDED
Two changes together (they go hand in hand — do NOT keep the layer-mean residual):
1. **Carry an explicit trajectory state** per hit. Lacking per-hit momentum, derive a direction
   from `true_x/y/z`: `tangentᵢ = normalize(true_posᵢ − true_posᵢ₋₁)` (2 angles) — a supervisable
   physical proxy for "momentum direction after the hit". Predict it (tokenized) and **feed it
   back**.
2. **Replace the layer-mean residual** with a position parametrized off the state: the state points
   to where the trajectory crosses the next layer; predict the hit as a small **scattering residual
   from THAT point** (tiny, track-dependent), or as absolute **local coordinates**. Keep the
   binning/tokenization (sharp marginals) — only change WHAT is binned: not deviation-from-layer-
   average, but deviation-from-this-track's-trajectory (or a local coord).

Result: all hits share one evolving state → coherence + persistence by construction; the token is
physically meaningful; the track-level structure lives in the state, not inferred from residuals.
Then add an **END token** → learn stopping → drop truth-`n_hits`.

NOTE: the paper predicts local *module* surface coords + momentum — i.e. a local (relative)
position is fine; the substantive addition over our current model is the STATE, plus dropping the
layer-*mean* framing in favour of a track-relative / local one.

### Option C — full paper-style rebuild
Flat token stream, `surface_id` geometry, local coords, end token, state features, sliding-window
attention. Biggest change; defer until B validates the hypothesis.

## 4a. HARD CONSTRAINT — learned, not analytic (do not skip)

The position must be produced by a **learned head conditioned on the state**, never computed from
params via detector geometry (no circle–cylinder intersection math in the generation path). Analytic
derivation = helix-z rebuilt: it bakes in the ideal-helix assumption and cannot represent multiple
scattering, soft-p_T spirals, kinks, secondaries, or layer inefficiency — the parts that make this
hard. The derived `true_xyz` direction is a **training target / auxiliary supervision** for the
state, NOT a generator; at inference the state AND the position are both predicted (learned).

Note (from the smearing check): real hits ARE near-deterministic given the trajectory, so a *sharp*
position prediction for clean tracks is correct — the model must LEARN that sharpness (and learn to
be BROAD for scattering tracks), not have it hard-coded, and must still learn which layers fire,
the stopping point, per-track scatter amplitude, and non-ideal tracks. The position head predicts
the full tokenized distribution conditioned on the state — never "pick one of N analytic candidates".

**Verification gate before trusting v2:**
1. Beat a dumb **analytic-helix baseline** on NON-ideal tracks (soft-p_T / scattering / secondary).
   Only tying on clean tracks ⇒ we baked in the helix.
2. **Ablate the state** — coherence must degrade without it (proves the state does the work).
3. Position distribution **non-trivial** where physics says so (broad for scatterers, sharp for
   clean), matching real — not a delta.

## 5. Minimal prototype (what to build first)

Goal: test the single hypothesis **"an explicit carried state fixes coherence"** with the least
new machinery, reusing the current 512-bin tokenized tracker.

1. **Slice**: also store `true_x/y/z`; compute per-hit **tangent direction** (2 angles) from the
   true positions; store it as two extra per-hit features.
2. **Model**: add a `dir` token group (2 binned angles) with its own embedding + head; predict it
   at each step and feed it into the next step's embedding (like abspos). Position heads unchanged
   (binned residuals) but now the hidden state carries an explicit, supervised direction.
3. **Train/generate**: identical loop; generation now also samples the direction and feeds it back.
4. **Measure** vs the 512-bin baseline: `half-persist` (target → 0.70), `z_r_resid` spike, the
   coherence features, **and the event gate / per-track AUC** (does it move the aggregate, unlike
   helix-z?).

If half-persist climbs and the aggregate improves → the state hypothesis holds → add the **END
token** (count/existence) and move toward Option C. If it doesn't → the aggregate really is
saturated and we bank the 512-bin tracker.

**Parallel, independent minimal win:** the **END token** can be prototyped on its own (predict a
stop token instead of using truth `n_hits`) — it directly attacks the frontier and doesn't need the
state. Cheapest path to "runs from Pythia input."

## 6. Risks
- **AR state drift** at generation (compounding errors in the fed-back direction) — the classic
  autoregressive failure; needs monitoring, maybe scheduled sampling.
- **Phase space** far harder than the paper's — expect worse than "comparable to full sim."
- **Derived (not true) state** — the tangent is a finite-difference proxy for momentum; noisy for
  short/kinked tracks and secondaries.
- Bigger surface area than any single fix so far — this is a rebuild, justified only because the
  current tracker is genuinely characterized as saturated.

## 7. Open questions before building
- Is `measured − true` smearing large enough to matter vs the model's excess scatter? (quick check)
- Do we want the state to be **physical** (tangent, interpretable, driftable) or a **free latent**
  (flexible, harder to train)?
- Multi-species + secondaries: does one state-model cover them, or do kinks (nuclear/conversions)
  break the smooth-state assumption? (relevant given our cascade work)
