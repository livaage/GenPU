# STATUS — GenPU tracker

Rolling state. Append-only detail lives in `experiment-memory/`. See also `TRACKER_GATE_FINDINGS.md`.

_Last updated: 2026-08-12_

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

## Calo (was undocumented until 2026-08-12; see [calo audit](experiment-memory/2026-08-12-calo-audit-and-metrics.md))
- **CaloClouds-lite flow model** (`src/genpu/flow/calo_flow.py`): global mixture + point flow-matching
  (50-step ODE) + energy mixture. Conditions on the 7 CONT_FEATURES.
- **Photon** (`photon_qtd_v1`): held-out gate **0.557** (≈in-sample 0.546), energy linearity/resolution
  good, lateral `shower_width` the weak spot (W/σ 0.44), **14.5 µs/particle**. Working + revalidated.
- **Pion** (`pion_qtd_v2`, retrained current-code): held-out gate **0.809**; width GOOD (core: 0.32),
  **energy weak** (broad resolution + huge-cell tail outliers → the fix target). Speed 27.8 µs.
- Photon & pion fail OPPOSITELY (photon=width, pion=energy). Unified fixes: core on photon + bound the
  energy-head tail.

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
- **Count head is wrong for pions**: multispecies count head undercounts (gen median 1 vs real 10) →
  honest count-driven pion gate meaningless until a per-species/pion count head exists. Orthogonal to tracker.
- **Generalize v3**: all-species surface slice + per-species count head → honest full-event gate vs the 0.77.
- **Stale checkpoints**: old `pion_v1`/`pion_v2_long` are 128-bin/pre-abspos, unloadable by current code.

## Ruled out
- v1 drift levers (all failed to beat 0.77; see TRACKER_GATE_FINDINGS.md): lower sampling
  temperature; count-routed single-hit placement; vertex anchoring (2 retrains); input jitter;
  momentum-z input feature (moved drift only ~10%).
- **v3 vertex/seed lever** — teacher-force test showed the v3 seed is already correct on multi-hit
  tracks; the "too-inner seed" was a population artifact. See
  [2026-08-10](experiment-memory/2026-08-10-v3-exposure-bias-scheduled-sampling.md).
