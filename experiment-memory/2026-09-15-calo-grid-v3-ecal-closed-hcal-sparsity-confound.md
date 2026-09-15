# Grid v3: ECAL closed on every layer and both endcaps; HCAL's failure is NOT simply "no single pitch"

**Date** 2026-09-15 · **Commit** `32d0cab` (+ untracked `scripts/calo_cell_grid_derive.py`, `jobs/calo_grid_v3.sh`)
**Branch** `flow-response` · **Job** 13926563 (1m45s) · 5 shards x 7,000 events = 190,471,166 cell-hits
`--r_band 1e9` (radial banding OFF, deliberately) · out `calo_geom/calo_cell_grid_v3.json`

Follows [the pitch correction](2026-09-15-calo-ecal-pitch-is-5.1-exactly-supersedes.md).

## Change

`pitch_from_nn`'s mode is now only a SEED; `refine_pitch` maximises circular concentration and
reports the INTER-WEDGE IQR of the estimate. Two wrong turns on the way, both now in the code:

- **Radially-subdivided patches broke it.** Pitch precision scales with the LEVER ARM, not the patch
  count; a 13x3 grid on HCAL returned 30.131 mm and on-grid 0.127, worse than not refining. Patches
  now span the full radius (`n_r = 1`).
- **A hardcoded 64x16 grid silently did nothing on HCAL** — ~9 cells per patch, under the floor, so
  `n_patch` was 0 and the log read like a refinement that agreed with the mode. Grid is adaptive and
  a skipped refinement now prints a WARNING.

## Result — ECAL endcap CLOSED

All 12 layer-fits (dets 9, 11 x layers 0/4/12/20/30/40), with radial banding OFF:

- **pitch 5.09998 mm, IQR 0.00000 mm at EVERY layer** (one layer 0.00001). Every wedge independently
  recovers the same constant to 5 dp. The true value is 5.1 mm exactly.
- **on-grid 0.9993 - 1.0000**, median residual 0.047 - 0.124 mm, p95 0.14 - 0.19 mm.
- **32 sectors everywhere**, and the angle pattern is IDENTICAL at every layer of both endcaps:
  `-45x4, -22.5x8, 0x8, +22.5x8, +45x4` — a period-8 palindrome in sector index (-45 and +45 are the
  same orientation mod 90). So ECAL's module orientations are structured, not uniform, and the model
  absorbs it completely.

`--r_band` is confirmed irrelevant: 1.0000 survives with banding fully off, which is what the pitch
correction predicted.

**UNEXPLAINED, and left open honestly**: median residual is 0.05-0.12 mm here against **0.0022 mm**
on the login-node check of the same detector and layer at 4,000 events. p95 is tight (0.15-0.19 mm),
so it is a shift of the whole distribution, not a tail. Leading hypothesis — 35,000 events exposes
many more RARELY-HIT cells (207,213 distinct vs 40,831), plausibly at module edges and transition
regions, where 4,000 events only ever shows the well-populated centres. NOT verified. It does not
affect snapping (on-grid >= 0.9993 either way) but it should not be quoted as 0.002.

## Result — HCAL endcap: the "no single pitch" framing was TOO STRONG

| det 12 / 14 | layer 0 | 4 | 12 | 20 | 30 |
|---|---|---|---|---|---|
| distinct cells | 13,463 | 12,516 | 10,327 | 7,376 | 5,176 |
| pitch IQR (mm) | **0.0056 / 0.0121** | 0.256 / 0.250 | 0.210 / 0.204 | 0.285 / 0.182 | 0.174 / 0.106 |
| verdict | **STABLE** | UNSTABLE | UNSTABLE | UNSTABLE | UNSTABLE |
| on-grid | 0.8797 / 0.8582 | 0.6247 / 0.6156 | 0.8237 / 0.8266 | **0.3660** / 0.4479 | 0.6070 / 0.6071 |

**The instability is CONFOUNDED WITH SPARSITY** — distinct-cell count falls monotonically with depth
and the IQR rises with it. So "a pitch that depends on the partition means there is no single pitch"
cannot be asserted for the deep layers: not-enough-data and no-single-pitch produce the same
signature, and this run cannot separate them. That was overclaimed in the job script's own comment.

**But layer 0 is the case that matters, and it is decisive**: pitch IS stable there (IQR 0.0056 mm,
29.9894 mm) on the largest sample HCAL has, and on-grid is still only **0.88**. Adequate statistics,
a well-determined lattice constant, and 12% of cells still off-grid. **So the rotated-square-lattice
model is genuinely incomplete for HCAL, independently of sparsity.** Earlier per-sector work found
this is not uniform — most sectors reach ~0.95 while specific ones (16, 26) sit at 0.67-0.86, i.e.
most modules fit and some do not.

Also note det 14 layer 30 produced 14 angle clusters including -35, -12.5, +30, +32.5 — a fit
collapsing on 5,231 cells, and a useful marker of what failure looks like.

Projective towers remain FALSIFIED (pitch identical across 6 radial bands, r 362-3047 mm).

## Verdict

**ECAL endcap: CLOSED.** 32 sectors, per-sector angle and origin, pitch 5.1 mm, depth on the
48-plane ladder, on-grid >= 0.9993 at every layer of both endcaps with no radial index.
79.6% of cells / 76.0% of energy (e±). Ready to build a `snap_to_cell` against.

**HCAL endcap: OPEN, and the question is now sharp** — not "what is the pitch" (29.989, stable at
layer 0) and not projective towers (falsified), but why ~12% of cells in specific sectors sit off a
lattice whose constant is well determined.

## Next

- Build `snap_to_cell` for the ECAL endcap and wire it into `sample_showers`. That is gap #1's first
  deliverable and it no longer has an unknown in it.
- HCAL: work at LAYER 0 ONLY, where statistics are adequate, and ask which sectors fail and whether
  their cells form a coherent sub-lattice of their own. Deeper layers cannot settle anything until
  the event count rises — worth checking how many shards would be needed before spending them.
- Still worth asking ColliderML for `cellID` or the calorimeter XML: it would close HCAL immediately
  and validate the ECAL derivation, and it is one uint64 they already have upstream.
