# 2026-08-10 — Helix anchor pre-build (validated) + v4 design

- **Branch:** flow-response. Geometry: `src/genpu/helix.py`; pre-build `scripts/helix_prebuild.py`.

## Why
v3-ms (gate 0.61) marginals are indistinguishable but `track_feats` shows the per-track HELIX
COHERENCE spikes (z_r_resid, phi_r_resid, dr_std at ~0) are smeared — structural to per-hit-
independent sampling (the momentum is never a fixed global latent; it's re-inferred each step). Fix:
predict positions as helix + small deviation, so coherence is by construction. Momentum is NOT
recomputed — compute the helix ONCE from initial (pT,eta,phi,q,vertex,B) and only evaluate it
(avoids the paper's predict-and-feed-back drift).

## Pre-build test (real pion tracks) — VALIDATED
- Transverse **circle-fit** residual (per-track fit): median **0.45 mm** (p90 18, p99 44) → tracks
  are clean circles.
- **Implied B = 3.07 T, IQR/median = 0.09** → initial pT predicts curvature via ONE global B;
  **energy loss does NOT break the fixed-momentum assumption**. B calibrated for free.
- Circle residual grows 0.4 mm (inner) → 4 mm (r≈900) — small energy-loss term.
- z-vs-r line residual large (median 31, p90 571) BUT slope=sinh(eta) exact → **use ARC-LENGTH
  parametrization, not z-vs-r** (v1's use_helix approximation fails for curly low-pT tracks).
- **Fixed-momentum helix** (circle from initial momentum, not fitted): transverse residual **2.4 mm
  median** (p90 63), z **8.5 mm median** — with calibrated **sign=-1**. This is the bounded
  'deviation' scale the model will predict.

## v4 design (hard residual anchor; user-selected)
Simpler than v3 — the helix subsumes the surface head. `layer_head(48)` → evaluate fixed helix at
that layer's ref (r_L barrel via `helix_at_r`, z_L endcap via `helix_at_z`) → deviation heads predict
(dx,dy,dz) ~few mm → position = helix + deviation. Coherence by construction; module (if needed) =
nearest surface to the output. Decoder/conditioning/count/stop/scheduled-sampling carry over.

## Build + the crossing-selection fix
v4 built: `models/tracker_helix_ar.py` (layer + dev heads, no surface head, 1.36M params),
`tracker_helix_model.py`, `train_tracker_helix.py`, `build_tracker_helix_slice.py` (particle-frame:
rotate by -phi0, store [layer,devx,devy,devz,time] + rotated vertex). Works in the phi-invariant
frame so r,z,layer (the gate features) are unchanged.

**Blocker + fix (crossing selection):** first attempt evaluated the helix at each layer's RADIUS r_L
— but a circle crosses r_L at TWO points and no deterministic rule disambiguated them (forward-sweep
~53%, forward-momentum ~82%), giving along-track dev median 92mm. FIX (per user's prompt "you compute
it once per track"): don't evaluate per-layer-radius — MARCH arc length s along the one fixed
trajectory and take the FIRST time it reaches each layer (`helix.layer_references` now grid-marches s;
barrel = first rising r crossing, endcap = exact since z monotonic). Along-track dev median 92mm ->
**2.2mm**. Residual 18.5% tail = low-pT (0.26) central (|eta|0.43) OUTER hits (step~11) — the
energy-loss limit where the fixed helix diverges; clipped at the bin edge.

## RESULT (v4 trained, job 12257170; gate/coherence 12257171/12258357)
v4-pion (helix_pion_v4/checkpoint_050000, 50k, TF->SS). val_dev 2.13mm (model predicts deviation
tightly).
- **COHERENCE FIXED (the goal):** `plots/tracker/v4_eval/track_feats.png` — gen `phi_r_resid` and
  `z_r_resid` now have SHARP SPIKES AT 0 matching real (v4 gen phi_r_resid med 0.030 vs real 0.019;
  v3 gen was a broad bump to 1-2, ~10x wider). The helix anchor reproduces per-track coherence that
  v3's per-hit-independent sampling structurally couldn't. **Concept validated.**
- **MARGINAL gate REGRESSED:** honest pion gate **0.78** vs v3-SS 0.596, from the 18.5% clipped tail
  (low-pT curly outer hits) inflating `r_std` (|Δ|/σ 0.29). Trade-off: v3-SS wins marginals, v4 wins
  track realism.

## Verdict
Helix anchor KEPT as validated concept. Marginal regression is NOT fundamental — it's the 18.5% tail.
Path to BOTH: hybrid reference (layer-mean fallback where the arc-length march is unreliable for
curly tracks) or nonlinear/wider deviation bins for that subpopulation. Then reco-level eval to decide
if coherence is worth pursuing over v3-ms (0.61, beats 0.77).
