# 2026-08-14 — `helix_at_z` endcap sweep-sign bug; corrected Phase 0b; Phase 1 anchor built

- **Commit / branch**: `9e0bbc2` / `flow-response` (working tree, not committed)
- **Jobs**: 12374054 (pion slice), 12374055 (e± slice + Phase 0b re-run), 12374056 (train+metrics,
  `--dependency=afterok` on both)

## Hypothesis

Phase 1 of [calo_tracker_coupling_plan.md](../calo_tracker_coupling_plan.md): store the shower core
as a **residual from the truth-helix extrapolation to the calo front face** instead of as an
absolute offset from the particle direction, so the GlobalHead mixture predicts a small local
residual rather than a 1.5 m magnetic-bending displacement. Truth conditioning only, so no exposure
bias. Phase 0b had gated this GO at 4.6x (pion) / 11.2x (e±) φ tightening.

## Change

**1. Found and fixed a sign bug in `src/genpu/helix.py:helix_at_z`.** Its sweep direction was
`w = -sign*sign(q)`, the opposite of `helix_at_r` and `layer_references` (both `sign*sign(q)`).
Validated against **80k real endcap tracker hits** (shard 0, charged tracks with ≥3 hits, outermost
hit with |z| > 1200 mm), predicting φ and r at the hit's own z:

| sweep sign | median \|Δφ\| | p90 \|Δφ\| | median \|Δr\| |
|---|---|---|---|
| `-sign*sign(q)` (old) | 0.633 rad | 2.85 rad | 32.7 mm |
| `sign*sign(q)` (fixed) | **0.0159 rad** | 0.460 rad | **8.9 mm** |

`helix_at_z` is used **only** by the calo anchor path (`grep` confirms), so no tracker result is
affected — but **the endcaps carry 83% of deposited calo energy**, so every Phase 0b number in
STATUS.md was measured through the wrong sign.

**2. Third anchor branch — the turning point.** The honest validity of a face crossing is far lower
than the old code reported (its endcap fallback silently accepted arc-length-clipped curlers, e.g.
anchoring a 0.2 GeV track at η+2.5). Only **37% of pion / 28% of e±** showers come from a particle
that actually reaches the calo face; the rest are soft secondaries (median pT 0.27 GeV, born at
vr ≈ 420 mm) that curl back first, with their calo energy booked to the parent. Those now anchor at
the helix **turning point** (outermost radius the circle reaches, |c| + R) rather than at the
particle direction. Measured on the invalid population, shard 0:

| species | σ(core_φ) particle → turning pt | σ(core_η) |
|---|---|---|
| pion | 1.493 → 0.617 (**2.42x**) | 2.028 → 0.445 (4.6x) |
| proton | 1.470 → 0.609 (2.41x) | 1.916 → 0.512 |
| e± | 0.263 → 0.132 (1.99x) | 0.295 → 0.070 |

**3. Phase 1 plumbing** (shared trunk — Phase 0c's split regressed the photon and stays a diagnostic
instrument, not the shipping config):
- New `src/genpu/calo_geom.py` — the single implementation of the front-face geometry and the
  anchor. The slice builder, `calo_metrics.py` and the Phase 0b probe all call it, so the training
  frame and the generation frame cannot drift apart. Returns a branch `mode`
  (0 barrel / 1 endcap / 2 turning point / 3 none) so every run can attribute its result.
- `build_calo_slice.py --core_anchor helix` — cell deltas are taken from the anchor, so `glob` dims
  2,3 hold the residual and the point cloud is unchanged. Stores `anchor` + `anchor_mode`.
- `calo_flow.py` — `core_anchored` buffer; `sample_showers(core_anchor=...)` adds the anchor back
  and **raises** if an anchored checkpoint is sampled without one (silently omitting it would put
  pion showers ~1.5 rad off in φ while still producing plausible marginals).
- `calo_metrics.py` recomputes the anchor from truth with the same code (φ and vx,vy are not in the
  `cont` contract, which is why the anchor is computed outside the model and handed in).
- `calo_core_diag.py` adds the anchor back on both the real and generated side, so the per-bin
  spread ratio stays directly comparable with the pre-anchor 0.58x / 0.94x.

## Result

Corrected Phase 0b, pion, 50k showers (`calo_helix_core_probe.py`): median |core| 0.846 → **0.068**,
φ tightening **16.46x** (was 4.6x), η 5.65x. Gain rises with pT bin from 11.4x to 22.1x.

Full pion slice smoke build (40k showers, shard 0), all branches combined:

| | particle frame | anchored | gain |
|---|---|---|---|
| median \|core\| | 1.639 | **0.436** | |
| σ(core_φ) | 1.282 | **0.353** | 3.63x |
| σ(core_η) | 1.502 | **0.271** | 5.54x |
| — barrel branch (10.5%) | 1.112 | 0.061 | **18.4x** |
| — endcap branch (25.4%) | 0.918 | 0.063 | **14.5x** |
| — turning point (64.1%) | 1.568 | 0.607 | 2.58x |

Round-trip check against an identically-built un-anchored slice: `residual + anchor` reproduces the
old core to 1e-7 for 96.6% of showers. The 3.4% that differ are clouds straddling ±π relative to
the particle direction, where the old `mean(wrap(cell − particle))` core is genuinely ill-defined —
a **latent bug the anchor also fixes** (point-cloud `d_phi` std 0.306 → 0.273).

## Verdict

**Kept.** The frame change is measured, not argued, and the sign fix quadrupled the expected gain.
Training result pending (job 12374056).

## Next

- Read the mechanism check first, not the gate: per-(charge × pT)-bin core spread ratio, target
  ~1.0 from 0.58x (e±) / 0.94x (pion). The gate moved on `pion_ctx` while the mechanism did not.
- The turning-point branch is a *proxy* for secondaries whose energy is booked to a parent that
  never reached the calo. That is exactly the population Phase 4 (track-conditioned calo) should
  own — the generated track knows where the particle actually stopped. Worth revisiting there.
- STATUS.md's Phase 0 table is superseded; all four species are being re-probed in job 12374055.
