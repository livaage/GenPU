# 2026-08-24 — calo 3D structure mapped: continuous depth is the right representation, and depth explains ~half of intrinsic width

- **Commit / branch**: `42b2947` / `flow-response`
- **Job**: 12881108 (`calo_3d_structure.py`, 250 events)
- Prerequisite for adding depth back, per
  [the 2D finding](2026-08-24-calo-is-2D-contrary-to-plan.md). Measurement only.

## A/C — layer structure: MIXED, so use a CONTINUOUS coordinate

Depth measured from the front face: barrel {10,13} = r - 1259.2, endcap {9,11,12,14} = |z| - 3212.5.

| det | type | uniq depth | med spacing | depth p50 | E frac |
|---|---|---|---|---|---|
| 9 | endcap | **48** | 5.05 mm | 70.7 | 0.270 |
| 11 | endcap | **48** | 5.05 mm | 70.7 | 0.284 |
| 12 | endcap | **36** | 51.0 mm | 1047 | 0.133 |
| 14 | endcap | **36** | 51.0 mm | 1098 | 0.146 |
| 10 | barrel | **2,526** | 0.08 mm | 39.4 | 0.159 |
| 13 | barrel | 291 | 3.6 mm | 502.9 | 0.008 |

Endcaps are cleanly segmented (48 EM layers, 36 hadronic). The barrel is effectively CONTINUOUS —
2,526 distinct depths at 0.08 mm, consistent with flat plates tiling a cylinder so cell-centre
radius varies across each plate.

**=> a CONTINUOUS depth coordinate.** It contains the discrete endcap layers as a subset, handles
the barrel natively, and is the cheap model change: `PointFlow` CFM `pts[:, :2]` -> `pts[:, :3]`,
no new categorical head.

## B — longitudinal profiles (never computable before this)

| species | depth mean (mm) | spread |
|---|---|---|
| gamma | **113.0** | 25.2 |
| e- / e+ | 301.4 / 300.8 | 41.8 |
| p | 334.6 | 48.3 |
| pi- / pi+ | **483.0 / 470.0** | 92.0 / 86.9 |

Textbook: EM shallow, hadronic deep, 4x separation gamma vs pion. This is the plan's named
acceptance metric (`pileup_generator_plan.md:356`) and the model has been blind to all of it.

## D — depth correlates with INTRINSIC width at 0.40-0.54

| species | corr(depth_mean, width) | **corr(depth_std, width)** |
|---|---|---|
| e- | 0.204 | **0.539** |
| e+ | 0.201 | **0.537** |
| p | 0.257 | **0.518** |
| gamma | 0.340 | **0.502** |
| pi- | 0.203 | 0.399 |
| pi+ | 0.196 | 0.406 |

Above the 0.3 threshold on every species, and an UNDERESTIMATE: showers are grouped by direct
depositing particle (re-attribution is not built yet), so each covers less depth than the real
shower it belongs to.

**PRECISION — this does NOT contradict "the core is 83-93% of width variance."** The width here is
RMS about the shower's OWN centroid, i.e. the INTRINSIC cloud size — not the metric's width about
the particle, which is core-dominated. The claim is narrower and sharper: the intrinsic size, which
that decomposition leaves as the residual 6-17%, is ~half driven by depth.

**Reinterprets a logged failure.** `--width_norm` normalised the intrinsic cloud size and moved
nothing ([2026-08-13](2026-08-13-calo-width-in-global-FAILED.md)). Intrinsic size varies
substantially WITH DEPTH, so normalising it inside a 2D model was flattening the projection of an
unmodelled variable. That does not make width_norm correct — it explains why it could not work.

## Verdict

Adding depth is not merely "more complete". It supplies a dimension that drives ~25-29% of intrinsic
width variance (r^2) and cleanly separates species longitudinally. Representation decision: one
extra CONTINUOUS coordinate.

## Next

- Sidecar the depth coordinate onto the existing `calo_hits_flat` (do NOT re-preprocess — the
  existing data is incomplete, not wrong; same reasoning as the particle_ids sidecar).
- Carry it into `build_calo_slice.py` `points_flat` (P,3) -> (P,4), bundled with re-attribution.
- `PointFlow` to 3D positions.
- Re-measure D after re-attribution, where the correlation should be stronger.
