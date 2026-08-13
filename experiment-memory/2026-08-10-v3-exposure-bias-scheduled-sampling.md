# 2026-08-10 — v3 drift is exposure bias → scheduled sampling

- **Commit / branch:** flow-response (working tree)
- **Runs:** teacher-force test job 12204371; SS fine-tune job 12204514 (init from `surface_pion_v3/checkpoint_060000`, out `surface_pion_v3_ss`).

## Hypothesis
After the v3 surface-local win (matched pion gate 0.9996→0.80, see
[2026-08-06](2026-08-06-surface-local-tracker-v3.md)), the per-step plot showed v3 still drifts
outward — but now in the discrete MODULE-SELECTION sequence, not a continuous coordinate. Predicted
this is **exposure bias** (compounding of own errors on off-distribution history), made *directional*
by the strictly inner→outer training order (an outward overshoot can't be corrected inward because
inward transitions never appear in training). If so: teacher-forced generation should NOT drift,
free-running should — and scheduled sampling (train on own history) is the direct fix.

## Change
1. Added `TrackerModuleARModel.teacher_forced_modules` + `scripts/tracker_teacherforce_test.py`:
   per-step mean module-r for real vs teacher-forced (true history) vs free-running (own history).
2. Implemented **2-pass scheduled sampling** in `TrackerModuleARModel.loss(ss_prob=...)`: pass 1
   (no grad) samples the model's own next-hit predictions from true history; a fraction `ss_prob`
   of input positions are replaced by them; pass 2 predicts the TRUE targets from that partly
   self-generated history. Refactored forward into `_decode`/`_heads` with a `target_layer` arg so
   the surface mask uses the true target layer even when the input is self-sampled. Trainer flags
   `--ss_prob/--ss_warmup/--ss_ramp/--init_from`.

## Result
**Teacher-force test (decisive):** teacher-forced ≡ real to <1mm at ALL 16 steps (step 8: 395.9 vs
395.6; step 13: 658.6 vs 658.1); free-running sits above with the gap growing monotonically with
step (0 at seed → ~40–60mm by step 13–15). → **Exposure bias confirmed.** The conditional is
perfectly learned; drift is pure own-history compounding. Also: the seed is fine on the multi-hit
population (step 0 all ~85mm) — the earlier "seed too inner (110 vs 275)" was a population artifact
(all-tracks incl. endcap-first), so **the vertex is NOT the lever**.

**Scheduled-sampling fine-tune (job 12204514, 25k steps from 60k TF, ss→0.4):** matched pion
truth-count gate **0.80 → 0.63**. Every spatial feature now |Δ|/σ ≤ 0.06 (r_mean 304 vs 303,
layer_mean 18.5 vs 18.6, z_std 1181 vs 1188) — tracker marginals statistically indistinguishable
from real. Free-running per-step drift curve now hugs real (`plots/tracker/v3_ss_drift.png`); only a
small residual at steps 13–15 (longest tracks). val losses unchanged (SS didn't hurt clean-history
accuracy). Residual 0.63 is mostly `hits_per_pion` (0.18) = the count head, not the tracker.
Progression: v1 0.9996 → v3-TF 0.80 → v3-SS 0.63.

## Verdict
Diagnostic KEPT: drift = exposure bias, not mislearned conditional, not seed. This redirects the fix
from vertex-anchoring (dropped) to either scheduled sampling (cheaper, in-model, testing now) or a
helix anchor (heavier, physics-guaranteed, reco bonus — fallback if SS underperforms).

## Count-head fix (normalization bug, not a modeling gap)
The "count head undercounts pions (median 1 vs 10)" was a NORMALIZATION MISMATCH, not a training
problem: the gate standardized `cont` with the *pion tracker-slice* stats while the count head was
trained with the *count-slice (all-species)* stats (vr mean 71 vs 403, logpt −1.1 vs −5.1). With the
correct stats the existing count head reproduces pions perfectly (median 10, mean 10.06 = real 9.97;
`plots/tracker/pion_count_dist.png`). Fix: made `CountHead` self-normalizing (`cont_mean`/`cont_std`
buffers + `sample_raw`), baked count-slice norm into `count_head_d0_selfnorm.pt`, gate passes RAW
cont. Prevents recurrence on any slice.

**HONEST (count-driven, fixed count) pion gate — the meaningful number:** v3-SS **0.596**, v3-TF 0.79.
Every feature |Δ|/σ ≤ 0.12 (most ≤ 0.04); largest residual is hits_per_pion 0.12. Honest ≈ truth-count
(0.596 vs 0.63) now that the count matches. This is directly comparable *in kind* to the multispecies
0.77 deliverable — and below it, though still pion-only.

## Next
- **Generalize to all species**: build all-species surface slice, retrain v3+SS → the honest
  full-event gate directly comparable to the 0.77 multispecies deliverable.
- Optional refinements: helix anchor for the residual drift tail (steps 13–15); bigger surf head.
