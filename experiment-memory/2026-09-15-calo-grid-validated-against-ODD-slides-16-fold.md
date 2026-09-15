# External validation against ODD slides: pitch CONFIRMED exactly, and the symmetry is 16-fold, not 32

**Date** 2026-09-15 · **Commit** `32d0cab` · **Branch** `flow-response`
**Login node, real data, no model, no job** (shard 0, 1,200-8,000 events).
Ground truth supplied by the user from Open Data Detector slides — the first EXTERNAL check any of
this derivation has had. Supersedes the "32 sectors" of
[v3](2026-09-15-calo-grid-v3-ecal-closed-hcal-sparsity-confound.md) and its 8-fold speculation.

## The slides

> ECAL: 48 sampling layers with 1.9 mm W, 0.5 mm Si, and readout (PCB, glue, air, ...), with
> **5.1 mm square cells**.
> HCAL: **Hexadecagon in cross section.** 30 sampling layers with 30 mm Fe, 3 mm Sci, and readout,
> with **30 mm square cells**.

## Scorecard

| slides | measured here | verdict |
|---|---|---|
| ECAL 48 sampling layers | **48** planes, all populated (137k-493k cells each) | exact |
| ECAL 1.9 W + 0.5 Si + readout | longitudinal pitch **5.050 mm** (2.4 mm active + ~2.65 readout) | consistent |
| **ECAL 5.1 mm square cells** | **5.09998 mm**, IQR 0.00000 over 24 wedges | **exact** |
| **HCAL hexadecagon (16 faces)** | **16 faces at offset 11.25 deg** — see below | **confirmed** |
| HCAL 30 mm Fe + 3 mm Sci + readout | longitudinal pitch **51.000 mm** (33 active + ~18 readout) | consistent |
| **HCAL 30 mm square cells** | ~29.99-30.03; the 16-face fit uses **30.0** | consistent |
| HCAL **30** sampling layers | **36** planes at exactly 51.0 mm, each holding 36k-120k cells | **MISMATCH** |

**The 5.1 mm confirmation is the one that matters most.** It independently validates
`refine_pitch` and retires the 5.0900 histogram mode for good: the refinement recovered a round
design constant to 5 decimal places from hit positions alone, with zero inter-wedge spread.

## THE SYMMETRY IS 16-FOLD — "32 sectors" was an alias, in BOTH detectors

"Hexadecagon" prompted the test the sector scan structurally could not do: our binning is
`floor((phi + pi) / (2*pi/N))`, so sector boundaries are FIXED at `-pi + k*(2*pi/N)`. If the real
face boundaries are offset, N=16 straddles every one of them — which is exactly why every earlier
scan rejected 16 (post-fit R ~0.03) and settled on 32. **32 was never the structure; it was the
coarsest grid whose boundaries happened to contain the true ones.**

Scanning the sector PHASE OFFSET at the correct pitch:

| offset (deg) | 0.00 | 2.81 | 5.63 | 8.44 | **11.25** | 14.06 | 16.88 | 19.69 |
|---|---|---|---|---|---|---|---|---|
| ECAL N=16 on-grid | 0.0375 | 0.0342 | 0.0370 | 0.0588 | **0.9999** | 0.0625 | 0.0364 | 0.0382 |
| ECAL N=32 on-grid | 1.0000 | 0.5416 | 0.5409 | 0.5359 | **1.0000** | 0.5416 | 0.5409 | 0.5359 |
| HCAL N=16 on-grid | 0.0008 | 0.0003 | 0.0009 | 0.0024 | **0.9344** | 0.0027 | 0.0007 | 0.0004 |

A single sharp spike at **11.25 deg = half of 22.5 deg**, i.e. face boundaries at
`11.25 + k*22.5 deg`. So:

- **ECAL endcap is ALSO a hexadecagon** — the slides state it only for HCAL, but ECAL gives 0.9999
  at N=16, statistically identical to N=32's 1.0000 with half the parameters.
- **HCAL improves on the alias**: 0.9344 at N=16 vs 0.8768 at N=32, because each face is fitted as
  one unit instead of being split in two.
- The angle clusters finally make sense: 16 faces at 22.5 deg spacing, taken mod 90 deg (a square
  lattice's axes are indistinguishable), give exactly 4 orientations — 0, +-22.5, 45 — which is the
  `-45x4, -22.5x8, 0x8, +22.5x8, +45x4` pattern seen at every layer of both ECAL endcaps.
- **The 2026-09-15 "8-fold with alternating orientations" speculation is WRONG** and is retired.

**Method lesson worth keeping**: the scan swept N over a decade and never swept the phase. A
symmetry search that fixes the origin can only ever find multiples of the truth, and it reports
those with total confidence — post-fit R was 1.0000 at N=32. Sweep the offset with the order.

## HCAL is NOT one global grid

Checked first, since "hexadecagon in cross section" could have meant a hexagonal OUTLINE over a
single global square grid: scanning pitch 29.90-30.10 with no rotation and no sectors leaves
Rx, Ry ~ 0.24 at every value including exactly 30.0, and on-grid < 0.001. So the faces really do
carry individually rotated lattices.

## The one real mismatch: HCAL layer count

The data is unambiguous — det 12 and det 14 each have **36** distinct |z| values at exactly 51.0 mm
spacing, indices 0..35 all present, 35,985 to 120,369 cells per plane. That is not a sparse-tail
artifact. Candidates: the slides describe a different ODD revision; or they describe the BARREL.
**The barrel cannot settle it from this data** — PIPELINE gap #2 already established that barrel
layers overlap in radius because staves tile a cylinder, so counting radial shells there is invalid
by construction (a crude attempt gave ~7 usable shells over 1647-3495 mm at ~50.8 mm spacing, which
is a detection threshold artifact, not a layer count). Left OPEN.

## Verdict

**ECAL endcap: CLOSED and now externally validated.** 16 faces with boundaries at
`11.25 + k*22.5 deg`, per-face angle and origin, pitch **5.1 mm exactly**, 48 depth planes at
5.050 mm. on-grid 0.9999.

**HCAL endcap: 16 faces at 30 mm confirmed, on-grid 0.9344** — a real improvement on 0.8797, and the
remaining ~6.6% is now the only open piece. Sparsity at depth (v3) still applies to layers past 0.

## Next

- Rebuild `snap_to_cell` on the **16-face** model, not 32 — half the parameters and it is the real
  geometry.
- Fix the sector scan to sweep phase alongside order, then re-run: every "32" in v2/v3 is an alias
  and the stored `calo_cell_grid_v*.json` are wrong in that field.
- HCAL's residual 6.6%: now worth asking whether it is the face BOUNDARIES (cells straddling two
  faces) rather than the lattice, which the 16-face model makes directly testable.
- The ask to ColliderML is now narrower and better motivated: with 5.1 mm and the hexadecagon
  independently confirmed, the open questions are the HCAL layer count (36 vs 30) and the residual
  6.6%.
