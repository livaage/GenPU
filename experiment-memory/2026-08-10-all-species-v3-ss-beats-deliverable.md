# 2026-08-10 — All-species v3+SS honest gate 0.61 (beats v1 0.77)

- **Branch:** flow-response
- **Runs:** slice 12207311, train 12207346 (`surface_ms_v3_ss/checkpoint_050000`), gate 12207347.

## Hypothesis
The pion v3+SS result (honest gate 0.596) should generalize: the surface-local representation +
scheduled sampling is species-agnostic (module + local coords; no helix needed), so an all-charged-
species v3+SS should beat the documented v1 multispecies deliverable (honest gate 0.77).

## Change
- Built `surface_multispecies.npz` matching the 0.77 population exactly: pdg classes
  [0,1,3,4,7,8,11,12] = e± π± p/p̄ μ±, shards 0-2 (6.87M particles, 34M hits, 0 unseen modules).
- Trained v3 from scratch, 50k steps, single TF→SS schedule (teacher-forced to 18k, ramp ss→0.4 by
  30k). Reused the same 18,824-module geometry and self-normalizing count head.

## Result
**Honest full-event gate (all charged species, count-driven): AUC 0.6105** vs v1 deliverable 0.77.
Count matches (gen median 2 mean 4.88 = real 4.79). Per-feature |Δ|/σ all ≤ 0.19 (largest: layer_mean
0.19, r_mean 0.19, frac_inner 0.14). Residual now leans slightly INNER (r_mean 356 vs 368) — SS mildly
over-corrected, opposite of v1's outward drift. 0.61 vs 0.77 = 0.16 gap, well beyond ~0.05 gate noise.

Progression (honest gates): v1 layer-residual 0.77 → v3-ms surface+SS **0.61**.

## Verdict
**KEPT — milestone.** The surface-local + scheduled-sampling tracker beats the v1 deliverable on a
like-for-like honest gate (same species/shards/features/classifier). Caveats: (1) cleanest confirmation
is running the v1 multispecies checkpoint through THIS gate to verify it reproduces ~0.77 (protocol is
identical — same event_features + MLP — but worth confirming); (2) val_surf still 2.62 at 50k (vs pion
1.65) → not fully converged, 0.61 may improve with more steps.

## Next
- Verify v1-ms reproduces ~0.77 through the current gate (rigor).
- Regenerate all-species drift/compare plots.
- Optional: more steps (surf head not converged); helix anchor for residual; reco-level eval.
