# Tracker v2 — implementation plan (concrete parametrization)

See [tracker_v2_design.md](tracker_v2_design.md) for the why. This is the exact build.

## Parametrization decision (see also the fork note in the design doc)

- **Position stays per-layer-local** (fine resolution; the 512-bin win). This is NOT the thing
  meant to carry the trajectory — it fully encodes the position given the layer, that's all.
- **The substantive additions** are the explicit **direction state** (predicted, updated,
  supervised) and the **END token**. These are what we actually lack.
- **HARD CONSTRAINT** (design doc §4a): the position head is LEARNED, conditioned on the state.
  No analytic geometry. The `true_xyz` tangent is a supervision target for the state, not a
  generator.

## Sequence

Per particle:  `START(vertex, init-dir) → hit₁ → hit₂ → … → hit_N → END`

Each `hitᵢ` carries tokens (predicted as a group, teacher-forced like current):
- `layer` ∈ {0..47}          — which layer fires (learned occupancy)
- `r,φ,z resid` (512 bins)    — per-layer-standardized position (unchanged encoding)
- `time` (64 bins)           — unchanged
- `dirθ, dirφ` (256 bins)    — **NEW**: the momentum-direction STATE after the hit, the tangent
                                `normalize(true_posᵢ − true_posᵢ₋₁)` as (polar, azimuth). Predicted,
                                fed back into the next step, supervised to the true tangent.
- `stop` (binary head)        — **NEW**: 1 at the last hit → END. Kills the truth-`n_hits` cheat.

## Data / slice (`build_tracker_slice_v2.py`)

Built from the **source** arrow shards (they have `true_x/y/z`; the preprocessed stage2 dropped it).
Join `tracker_hits` ↔ `particles` by `(event_id, particle_id)`.
Per particle, sorted inner→outer by measured r:
- position residuals `(r,φ,z,time)` — same as v1 (`(phys − LAYER_MEANS)/LAYER_STDS`).
- **direction** `(θ,φ_dir)` per hit from the TRUE positions (clean: true r_mono = 1.0). First hit's
  direction = from vertex to hit₁.
- cont conditioning: the 7 shared features (unchanged) + perigee d0/z0 available if useful.
- store `n_hits` too (for eval + optional teacher signal), but generation will NOT use it.

## Model (`tracker_state_ar.py`, new — keep v1 intact)

- Reuse the causal transformer decoder + tokenized embeddings + abspos + vertex seed.
- Add `dir` embedding + head (2 angle token groups). The predicted direction is embedded and added
  to the next step's input (the explicit carried state), alongside abspos.
- Add a `stop` head (binary) at each position.
- Losses: CE(layer,r,φ,z,time) + CE(dirθ,dirφ) [state supervision] + BCE(stop).
- Generation: sample layer/pos/dir/stop each step; feed the sampled dir back; **stop when `stop`
  fires** (no truth `n_hits`). Position head conditioned on state — no analytic intersection.

## Verification gate (design doc §4a) — before trusting v2

1. **Analytic-helix baseline**: build a dumb helix generator; v2 must beat it on NON-ideal tracks
   (soft-p_T / scatter / secondary), not just tie on clean ones.
2. **Ablate the state** (zero the dir tokens): coherence (half-persist, z_r_resid spike) must
   degrade → proves the state does the work.
3. **Count/existence**: n_hits distribution from the END token vs real (now emergent, not truth).
4. Standard: event gate, per-track AUC, coherence spikes, hit marginals — vs the 512-bin baseline.

## Build order

1. [ ] `build_tracker_slice_v2.py` (source join + direction) → slice on scratch.  ← START HERE
2. [ ] `tracker_state_ar.py` + wrapper (dir state + stop head).
3. [ ] `train_tracker_v2.py` (dir + stop losses).
4. [ ] eval: extend diagnostics to v2 (dir-aware generate, stop-based length); analytic-helix
       baseline; state ablation.
5. [ ] train (SLURM) → verification gate → compare to 512-bin.

## Risks (carried from design doc)

AR state drift at generation; harder phase space than the paper; derived (not true-momentum) state;
big surface area. Mitigations: state ablation + analytic baseline as guardrails; scheduled sampling
if drift shows.

## RESULT — 30k pion eval (first signal)

- **Direction supervision HELPS coherence:** half-persist v1 0.58 -> v2 0.65 (ablated), near real
  0.68. The auxiliary direction task improved the learned representation.
- **State FEEDBACK backfires (exposure bias / drift):** with-state 0.567 < ablated 0.647 — feeding
  the model's OWN noisy predicted direction back compounds error. Fix options: (a) auxiliary-only
  (drop feedback — already beats v1), (b) scheduled sampling (train on own predictions).
- **Stop head roughly works but length not matched** (gen too long: r_mean 390 vs 308) — confounds
  z_r_resid (252 vs 152) and per-track AUC (0.85 vs v1 0.80). Needs length/stop tuning.
- Verdict: IDEA validated (explicit direction improves coherence); FEEDBACK mechanism needs a fix;
  stop head needs tuning. Caveats: 30k (undertrained), pion-only.
