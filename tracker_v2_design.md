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

### Option A — decompose `measured = true trajectory + smearing`
Model the **true** trajectory (smooth, coherent) and add **per-hit smearing** (`measured − true`,
a calibrated marginal per detector type) at generation. Pros: cleanly splits the physical noise
from the model's; smoother target. Cons: predicting the true trajectory autoregressively *still*
injects per-hit scatter unless combined with a state — so A alone doesn't fix coherence. Best as a
**target/eval refinement layered under B**.

### Option B — state-carrying AR (the paper's idea, adapted to our data) — RECOMMENDED
Carry an explicit **trajectory-direction state** per hit. Since we lack per-hit momentum, derive it
from `true_x/y/z`: `tangentᵢ = normalize(true_posᵢ − true_posᵢ₋₁)` (2 angles) — a per-hit,
supervisable, physical proxy for "momentum direction after the hit." Then:
- predict the direction state per hit (tokenized), **feed it back**;
- predict the position **conditioned on the current state** (small offset from where the state
  points), so all hits share one evolving trajectory → coherence + persistence by construction;
- add an **END token** → learn stopping → drop truth-`n_hits`.

### Option C — full paper-style rebuild
Flat token stream, `surface_id` geometry, local coords, end token, state features, sliding-window
attention. Biggest change; defer until B validates the hypothesis.

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
