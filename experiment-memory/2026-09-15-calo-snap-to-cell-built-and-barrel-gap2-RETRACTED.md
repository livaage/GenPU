# `snap_to_cell` built for the WHOLE calorimeter — and PIPELINE gap #2 (barrel unreachable) is RETRACTED

**Date** 2026-09-15 · **Commit** `32d0cab` (+ new `src/genpu/calo_cells.py`) · **Branch** `flow-response`
Login node, real data, no model, no job. Closes the derivation line that began with the cell-grid
work; constants from the ODD XML (see
[XML confirmation](2026-09-15-calo-geometry-CONFIRMED-from-ODD-xml.md)).

## Result — the acceptance test

Snapping REAL cells must move them ~0, since real cells already sit on the grid. Shard 0, 800 events,
4,290,156 cell-hits:

| det | kind | cells | median displacement | frac within 10 um |
|---|---|---|---|---|
| 9 | ECAL endcap | 1,478,827 | 0.00005 mm | **0.9999** |
| 11 | ECAL endcap | 1,523,215 | 0.00005 mm | **0.9999** |
| 10 | **ECAL barrel** | 831,483 | 0.00005 mm | **0.9982** |
| 13 | **HCAL barrel** | 13,076 | 0.00003 mm | **0.9948** |
| 12 | HCAL endcap | 218,043 | 0.00001 mm | 0.9744 |
| 14 | HCAL endcap | 225,512 | 0.00001 mm | 0.9755 |

**Overall 4,277,213 / 4,290,156 = 0.99698 within 10 microns of a real cell centre.** ECAL endcap cell
IDs are bijective on real data (3,104,127 distinct positions -> 3,104,127 distinct ids, ratio
1.0000); ECAL barrel 0.9999.

## PIPELINE gap #2 IS WRONG AND SHOULD BE RETRACTED

Gap #2 says exact cell identity is "unreachable for barrel EM" because barrel cells from different
layers overlap in RADIUS (max gap 0.985 mm over a 106 mm span). The overlap is real. The conclusion
is not: **radius is simply the wrong coordinate.** A barrel stave is a FLAT plate, so depth is the
PERPENDICULAR distance to the stave plane, and radius blends depth with the across-stave position —
which is exactly why it looks continuous.

In the stave frame all three coordinates are exactly discrete (circular concentration R; 1.0 = every
value on one grid):

| det | R(z) | R(along) | R(perp) | layers recovered |
|---|---|---|---|---|
| 10 ECAL barrel | **1.0000** @5.1 | 0.9979 @5.1 | **0.9998** @5.050 | 48 |
| 13 HCAL barrel | **1.0000** @30 | 0.9985 @30 | **1.0000** @51.00 | 36 |

Three DIFFERENT pitches — cell size on the two in-plane axes, layer pitch on depth. The barrel is
~18% of ECAL cells and 17.3% of ECAL energy, and it was written off.

## Nothing here is a free parameter

The stave/face phase is the XML's own `90 - 180/16 = 78.75 deg`, which modulo the 22.5 deg face
spacing is **11.25 deg**. A first attempt found 11.953 deg by maximising depth discreteness — a blunt
objective with a wrong origin. Maximising `R(along)` instead, a far sharper one, lands on **11.250
deg exactly**.

The sensitive-slice inset is also derivable: the first cell centre sits at half the first sensitive
slice, i.e. the XML stack up to and including half the sensitive layer.

| | derived inset | barrel apothem + inset | measured | endcap zmin + inset | measured |
|---|---|---|---|---|---|
| ECAL | 1.90 + 0.15 + 0.10 + 0.50/2 = **2.40** | 1250 + 2.4 = 1252.4 | 1252.4 | 3200 + 2.4 = 3202.4 | 3202.4 |
| HCAL | 30 + 16 + 3/2 = **47.50** | 1600 + 47.5 = 1647.5 | 1647.5 | 3600 + 47.5 = 3647.5 | 3647.5 |

All four agree exactly. The `z0` constants that had been carried as MEASURED since job 12884120 are
now derived.

## Two bugs found by the acceptance test, both worth recording

1. **Half-integer vs integer cell centres.** The first version put centres at `(i + 0.5) * pitch` on
   both in-face axes. Real `v0` is **0** — centres are at INTEGER multiples along the face normal.
   Symptom: a constant 3.2169 mm displacement with **median == p95 exactly**, and 100% of cells
   outside their own cell. A rigid offset, not a rounding error, and the equality of median and p95
   is what says so.
2. **The circular mean is a biased origin estimator when concentration < 1.** It gave HCAL endcap
   u0 = 2.5419, leaving a rigid **-0.05279 mm** on every cell with **p5 = p95 = -0.0528**, i.e. zero
   spread. This was first misread as HCAL's "structural residual"; it was an origin error. Corrected
   to 2.48911 the irreducible spread is 1e-5 mm and HCAL endcap goes 0.0000 -> **0.974**.

## What is left

- **HCAL endcap ~2.5%** and **barrel ~0.2-0.5%** tails, with p99 around 11.7 mm (ECAL barrel 1.55 mm,
  HCAL barrel 4.50 mm). These are FACE/STAVE-BOUNDARY effects — cells near a boundary assigned to the
  neighbouring face — not lattice errors, since the in-face residual is now ~1e-5 mm. Fixable by
  testing both adjacent faces and keeping the better fit.
- The earlier "HCAL has no single pitch / the lattice model is wrong" reading is now **largely
  retracted**: with the origin fixed, HCAL endcap sits on its lattice to 1e-5 mm. What remains is
  boundary assignment, which is a different and much smaller problem.

## WIRED IN (same day)

`etaphidepth_to_xyz`, `cell_ids` and `snap_and_merge` added; `calo_metrics.py --snap_cells` applies
them. Projection is kept as a SEPARATE deterministic step after the continuous sample rather than
folded into `sample_showers`, which is what `pileup_generator_plan.md:257` specifies.

END-TO-END on 2,102,552 real cells, using only what generation has (eta, phi, depth):

| stage | result |
|---|---|
| round trip (eta,phi,depth) -> xyz | position error **0.00000 mm** |
| detector recovered from the eta cut | **0.9950** (matches calo_geom's own 0.9951) |
| snapped position vs the ORIGINAL cell | median 0.00005 mm, **0.9920** within 10 um |
| **same cell id recovered** | **0.9925** |
| merge within one event | collision **0.0004** -- real cells are already distinct, so this is the id map's own error floor |
| energy conservation | **1.000000** |

Merging is not optional: a cell is one readout channel and two generated points landing in it must
ADD (PIPELINE gap 3b). A first version keyed the merge on `cid ^ (src << 3)`, which is NOT injective
-- distinct (cell, shower) pairs can collide and silently merge channels that were never shared.
Replaced with an exact pair key. The first merge test also looked alarming (14% merged, "3x the real
3.5%") purely because it merged across 400 EVENTS; within an event it is 0.0004.

## Next

- Run `jobs/calo_snap_gate.sh`: two arms one flag apart, continuous vs projected, on
  `multispecies_v2_s0`. Pre-registered expectation: merging removes cells, so `n_cells` and
  `cells_per_src` FALL and per-cell energies RISE; the gate improves if the generator's co-occupancy
  is realistic and worsens if it piles too many points into one cell, with `multi_contrib_frac`
  against the real ~3.5% saying which.
- Fix boundary assignment (test adjacent faces) -- the residual 2.5% HCAL endcap / 0.2-0.5% barrel.
- A real CELL-LEVEL metric, now that gap #2 is open: occupancy, cells-per-shower and the
  energy-per-cell spectrum against real cells rather than against a continuous cloud.
