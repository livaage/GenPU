# 2026-08-06 — Surface-local tracker (v3)

- **Commit / branch:** fc60e2e (working tree) / flow-response
- **Run / ckpt:** `checkpoints/tracker/surface_pion_v3/checkpoint_060000.pt`; gates jobs 12074333 (honest), 12077190 (truth-count); baseline `pion_baseline_current/checkpoint_060000` (job 12074345).

## Hypothesis
v1's tracker gate plateaued at 0.77 because of a systematic autoregressive **outward drift**: it
predicts `layer_class(48) + per-layer-standardized (r,φ,z) residual`, and the radial residual spans
±60 mm within a layer — an unbounded continuous coordinate that compounds outward step-by-step. The
reference paper (arXiv:2512.24254) never uses residuals: it predicts an **absolute local coordinate
on a discretely-chosen module** + carries momentum, so global position lives in the discrete token
and can't drift. Pre-build tests confirmed ColliderML `surface_id` gives 18,824 modules with within-
module spread ≤~50 mm and `H(next|current)`=5.13 bits (learnable). Expected: a module-token + local-
coordinate representation collapses the drift.

## Change
- Built `module_geometry.npz` (18,824-module vocab, full detector coverage, per-module local frames)
  via `scripts/build_module_geometry.py`; `surface_pion.npz` slice (1.62M pions, shards 0-2) via
  `scripts/build_tracker_surface_slice.py`. Loader `src/genpu/module_geometry.py`.
- v3 model `src/genpu/models/tracker_module_ar.py`: **hierarchical module head** — `layer_head(48)`
  + layer-conditioned **masked** `surface_head(3360)` (better-conditioned than a flat 18,824-way
  softmax, reuses the proven layer head, ~5.5× fewer output params) — + per-module-standardized local
  Cartesian `x/y/z`(512-bin)+`time`(64-bin) heads. Decoder/conditioning mirror v1. Wrapper
  `tracker_module_model.py`, trainer `train_tracker_module.py`, gate adapter
  `tracker_honest_gate_module.py` (maps module→layer, x,y→r to feed the SAME gate).
- Trained 60k steps, A100 (job 12072376). Losses: layer 3.67→0.35, surf 5.6→1.7, cont 5.24→3.2.
- Baseline: retrained v1-pion with CURRENT code (512-bin+abspos), plain, `pion.npz`, 60k steps — the
  old `pion_v1`/`pion_v2_long` checkpoints are stale (128-bin, pre-abspos) and unloadable.

## Result
Matched TRUTH-count gate (same pion slice, current code, 60k steps, both plain, no vertex; removes
the count-head confound):

| feature | real | v1 gen (|Δ|/σ) | v3 gen (|Δ|/σ) |
|---|---|---|---|
| **AUC** | 0.5 | **0.9996** | **0.80** |
| layer_mean | 18.6 | 29.3 (1.84) | 19.2 (0.23) |
| r_mean | 303 | 569 (1.86) | 325 (0.36) |
| frac_inner | 0.39 | 0.07 (1.83) | 0.39 (0.06) |
| z_std | 1188 | 1607 (1.45) | 1199 (0.06) |
| r_std | 267 | 284 (0.44) | 286 (0.39) |

The surface representation **eliminates the gross outward misplacement** — v1 puts pion hits ~2× too
far out; v3 sits on the real distribution.

## Verdict
**KEPT.** Decisive on the matched comparison (0.9996→0.80). The drift-driven features are fixed.
Caveats: (1) 0.80 is not below the best documented multispecies v1 0.77 — apples-to-oranges (pion-only
truth-count, plain v3, noisy surf head). (2) A **residual mild outward bias** remains (r_mean 325 vs
303, layer_mean 19.2 vs 18.6, |Δ|/σ~0.3) — the 0.80 keys on this; likely the module-SELECTION
sequence drifting slightly outward (discrete-anchor version of the old drift). (3) Separate open bug:
the multispecies count head undercounts pions (gen median 1 vs real 10), so the HONEST (count-driven)
pion gate is meaningless until a per-species count head exists — orthogonal to the tracker.

## Next
- Attack the residual module-selection drift: vertex-seed the module sequence; more steps / larger
  surf head.
- Per-species (or pion) count head so the honest count-driven gate is meaningful.
- All-species surface slice → honest full-event gate vs the 0.77 multispecies deliverable.
