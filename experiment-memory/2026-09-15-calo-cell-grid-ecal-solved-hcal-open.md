# Calo cell grid: ECAL endcap is an analytic 5.09 mm / 32-sector lattice; HCAL resolves but does not close

**Date** 2026-09-15 · **Commit** `32d0cab` (+ untracked `scripts/calo_cell_grid_derive.py`) · **Branch** `flow-response`
**Jobs** 13920087 (first pass, estimator bug), 13921623 (corrected, 11m52s) · 5 shards x 7,000 events
= 190,471,166 cell-hits · out `calo_geom/calo_cell_grid_v2.json` · real data, no model

## Hypothesis

PIPELINE.md gap #1: the calo model emits a continuous point cloud and nothing snaps it to a cell.
Gap 3e settled the LONGITUDINAL axis (48/36 exact planes, 100% on-grid). The TRANSVERSE axis had
never been characterised — gap 2a knew only that ~2e7 distinct positions exist and that 5 mm
quantisation merges 2.4% of them. Gap 2a's ~2e7 vocabulary made a per-module LOOKUP TABLE look
unavoidable.

## First, the prerequisite that had to be checked: are cell ids really absent?

**Verified at four levels — they are, and we did not drop them.** (1) our arrow shard: 9 columns;
(2) the dataset README's documented `calo_hits` schema: the same 9, with `x,y,z` defined as "Cell
center position (mm)"; (3) **the source parquet as served by HF datasets-server**, bypassing our
conversion: identical; (4) `ttbar_pu200_calo_hits` and `ggf_pu0_calo_hits`: identical, so it is not
a `pileup_only_pu0` reduction. The HF repo is 47,295 parquet files plus README and .gitattributes —
no geometry file at all.

**The smoking gun**: EDM4HEP's `SimCalorimeterHit` has exactly four members — `cellID`, `energy`,
`position`, `contributions`. ColliderML published three. **`cellID` is the single member dropped in
the EDM4HEP->Parquet conversion**, and that converter is not public. The public ODD repo
(`acts-project/OpenDataDetector`, single `main` branch, 6 XML files) ships the TRACKER only — no
calorimeter. For contrast `tracker_hits` DOES ship `volume_id/layer_id/surface_id`; the calo ships
none, so this is an upstream design choice.

Useful by-product: `colliderml/physics/detector_enums.py` confirms our mapping exactly —
ecal_neg_endcap 9, ecal_barrel 10, ecal_pos_endcap 11, hcal_neg_endcap 12, hcal_barrel 13,
hcal_pos_endcap 14 — independently validating the `BARREL={10,13}` / `ECAL_DETS=(9,10,11)`
constants that had been inferred from geometry.

**So the lattice must be FITTED.** Enumerating is not an option at any affordable event count: a
single ECAL layer holds ~2.6e5 cells and occupancy never saturates (43% of each event's cells still
unseen at event 4,000). The first draft of the script grouped cells by connected components of the
"neighbours at ~pitch" graph and duly fragmented — 3,558 components, largest 11 cells — which is
what enumeration looks like when it fails.

## The estimator bug, and why it is recorded

Job 13920087 LOST both ECAL layers 0 and 12: `pitch_from_nn` returned **7.21 mm = 5.09*sqrt2**, the
lattice DIAGONAL, and every sector fit built on that constant failed. Cause was `--max_pts` RANDOM
subsampling. Deleting lattice points at random deletes each cell's true nearest neighbours, so the
NN mode climbs the harmonics. Measured on det 9 layer 0 (12,485 distinct, true pitch 5.09):

| retained | 1.00 | 0.50 | 0.25 | 0.17 | 0.10 |
|---|---|---|---|---|---|
| random (old) | 5.09 | **7.21** | **11.41** | 11.41 | 11.41 |
| annuli (new) | 5.09 | 5.09 | 5.09 | 5.09 | **5.09** |

The job kept 17% at layer 0. **Layer 30 survived at 21% by luck of density, not by design** — so
13920087's ECAL numbers were right but not safely obtained, and the numbers below supersede them.

Fix: `annulus_subsample` keeps whole RADIAL annuli, so retained points keep their neighbours except
at two edges. Annuli and NOT phi wedges deliberately — wedge boundaries would impose an angular
period, and the angular period is exactly what the sector scan measures, so wedges would have made
the 32-sector result partly circular. Effect is stark: NN mass at the true pitch **0.234 -> 0.98**.

## Result

### ECAL endcap (det 9, 11) — SOLVED, and constant with depth

| layer | 0 | 4 | 12 | 20 | 30 | 40 |
|---|---|---|---|---|---|---|
| pitch (mm) | 5.0900 | 5.0900 | 5.0900 | 5.0900 | 5.0900 | 5.0900 |
| sectors | 32 | 32 | 32 | 32 | 32 | 32 |
| on-grid (<0.5 mm) | 0.9994 | 0.9993 | 0.9989 | 0.9992 | 0.9993 | **0.9996** |
| median residual (mm) | 0.086 | 0.085 | 0.088 | 0.083 | 0.083 | 0.080 |

Both endcaps agree to 4 dp. **Pitch and sector count do NOT vary with depth** — that was the open
question after 13920087, which had only ever fitted one ECAL layer cleanly.

Evidence the 32 is real and not a fitting artifact: global-grid R is 0.044 (a single axis-aligned
lattice would be 1.0); the scan is flat and low through 24 sectors and JUMPS at 32; and in the
earlier pass 48 sectors DEGRADED on-grid to 0.72 while 64 and 96 stayed above 0.98 — module
boundaries align only at multiples of 32.

### HCAL endcap (det 12, 14) — resolves at 32 sectors, but does NOT close

| layer | 0 | 4 | 12 | 20 | 30 |
|---|---|---|---|---|---|
| pitch (mm) | 29.99 | 29.99 | 29.99 | 29.99 | 29.99 |
| sectors | 32 | 32 | 32 | 32 | FAILED |
| on-grid | 0.9393 / 0.9367 | 0.9443 / 0.9499 | 0.9188 / 0.9279 | 0.9535 / **0.8939** | sparse |
| median residual (mm) | 0.118 / 0.117 | 0.104 / 0.108 | 0.112 / 0.099 | 0.086 / 0.108 | — |
| p95 residual (mm) | **0.79 / 0.78** | 0.63 / 0.50 | 0.72 / 0.66 | 0.43 / 0.70 | — |

32-fold symmetry is SHARED with ECAL at a 5.9x coarser pitch. But on-grid is 0.89-0.95 against
ECAL's 0.999, and the p95 residual tail runs to 0.79 mm against ECAL's 0.19 mm.

**That gap is not the subsampling artifact**: HCAL was never subsampled (13,463 distinct < the
30,000 cap), so `annulus_subsample` cannot explain it. Note also HCAL's global R is 0.23 vs ECAL's
0.044 — HCAL is already partly aligned in the global frame, which a uniform rotated-square model
does not predict.

`det 12/14 layer 30` fails on genuine sparsity (5,176 / 5,231 distinct). `layer 40` correctly
skipped — HCAL has 36 layers.

## Verdict — KEPT for ECAL, OPEN for HCAL

**The ECAL endcap projection is ANALYTIC, not a table**: sector = `floor(phi/(2*pi/32))` -> rotate
-> round to 5.09 mm -> snap depth to the 48-plane ladder. This is the result that matters, because
gap 2a's ~2e7 vocabulary had made a per-module LUT look forced. It is not forced; the table is a
formula. ECAL endcap is 79.6% of cells / 76.0% of energy for e±.

HCAL is 28.5% of calo energy and a median residual of 0.10 mm with a 0.79 mm tail is not good enough
to snap on.

## Next

- **HCAL follow-up, hypotheses in order**: (a) PROJECTIVE towers — cell size grows with radius, which
  would produce exactly this signature and is testable by fitting pitch per radial annulus;
  (b) a RECTANGULAR cell (separate pitch on the two lattice axes), testable by fitting u and v
  independently; (c) two pitches in different radial regions. The non-zero global R (0.23) should be
  explained by whichever wins.
- The BARREL remains untouched and deliberately so — staves tile a cylinder, gap #2 showed layers
  overlap in radius, gap 3e found only 4.6% of barrel cells on a plane. It needs its own treatment
  and conflating it with the endcap is how the depth axis became ambiguous in the first place.
- **Worth doing in parallel and cheap**: ask the ColliderML authors to restore `cellID` to the calo
  tables (one uint64 they already have upstream) or to publish the calorimeter XML. That would turn
  this whole derivation into a one-line validation, and would settle HCAL immediately.
- Residual is 0.08 mm where the longitudinal derivation cleared exactly 0.0000. Probably the pitch
  constant (a histogram mode on a 0.02 mm bin); a least-squares refinement of the pitch should
  shrink it. NOT chased — 0.08 mm against a 5.09 mm pitch does not affect cell assignment.
