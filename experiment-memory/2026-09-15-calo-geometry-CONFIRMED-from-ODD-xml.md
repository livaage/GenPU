# The ODD calorimeter XML exists and every derived number is CONFIRMED — including the one the slides got wrong

**Date** 2026-09-15 · **Commit** `32d0cab` · **Branch** `flow-response` · **No job** (network fetch + arithmetic)
Source: `github.com/OpenDataDetector/OpenDataDetector`, files `xml/detectors/CalorimeterECal.xml`,
`xml/detectors/CalorimeterHCal.xml`, `xml/OpenDataDetectorEnvelopes.xml`. Supplied by the user.

## Correction to a claim I made earlier today

[The grid entry](2026-09-15-calo-cell-grid-ecal-solved-hcal-open.md) states that the calorimeter
geometry "is not retrievable from where the data came from" and that "the public ODD repo ships the
TRACKER only". **That was wrong.** I searched `acts-project/OpenDataDetector` (tracker only, 6 XMLs)
and `OpenDataDetector/ColliderML` (the data pipeline) and concluded the geometry did not exist
publicly. I never searched **`OpenDataDetector/OpenDataDetector`**, which is the actual geometry repo
and contains full calorimeter XML. The correct conclusion from that search was "I have not found it",
not "it is not published".

The derivation was still worth doing — it is now validated rather than assumed — but it was not
forced, and several hours of fitting could have been a `curl`.

## Every number confirmed

| quantity | ODD XML | measured here | |
|---|---|---|---|
| ECAL cell size | `ECal_cell_size = 5.1*mm` | **5.09998 mm** (IQR 0.00000) | exact |
| ECAL layers | `<layer repeat="48">` | 48 | exact |
| ECAL layer pitch | slices 1.90+0.15+0.10+0.50+0.10+1.30+0.25+0.75 = **5.05 mm** | **5.050 mm** | exact |
| ECAL symmetry | `ecal_e_symmetry = 16` | **16 faces** | exact |
| ECAL endcap z | `ecal_e_min_z = 3.2 m` | 3202.4 mm | consistent |
| ECAL endcap r | 315 - 1500 mm | 316.8 - 1496.9 observed | consistent |
| HCAL cell size | `HCal_cell_size = 30*mm` | ~30.0 mm | exact |
| **HCAL layers** | **`<layer repeat="36">`** | **36** | **exact — see below** |
| HCAL layer pitch | slices 30 + 16 + 3 + 2 = **51 mm** | **51.000 mm** | exact |
| HCAL symmetry | `hcal_e_symmetry = 16` | **16 faces** | exact |
| HCAL endcap z | `hcal_e_min_z = 3.6 m` | 3647.5 mm | consistent |
| HCAL endcap r | 355 - 3436 mm | 362 - 3299 observed | consistent |
| endcap segmentation | `CartesianGridXZ` (LOCAL stave frame) | per-face rotated square lattice | matches |

### The phase offset is in the XML

Both endcap envelopes carry `<rotation z="90*deg - 180*deg/symmetry"/>`. With symmetry = 16 that is
**78.75 deg**, and modulo the 22.5 deg face spacing it is **11.25 deg** — exactly the offset the
phase scan found as a single sharp spike (ECAL on-grid 0.9999, HCAL 0.9344, against ~0.001-0.06 at
every other offset). The measurement and the source agree to the digit.

### THE SLIDES WERE WRONG ABOUT HCAL LAYERS, AND THE DATA WAS RIGHT

The slides said "30 sampling layers"; we measured 36 and flagged it as the one mismatch. The XML
says `<layer repeat="36">` for BOTH HCal barrel and endcap. **36 is correct.** The slides' material
list (30 mm Fe, 3 mm Sci, readout) is right and matches `Steel235 3.0cm + siPCBMix 1.6cm +
Polystyrene 0.3cm + Air 0.2cm = 51 mm`; only the layer count was off. Worth keeping as a reminder
that a summary slide is not a source, and that "the data disagrees with the documentation" was
resolvable by going to the primary source rather than by assuming our measurement was wrong.

### The dropped identifier, exactly

Both readouts declare
`<id>system:8,barrel:3,module:4,stave:1,layer:6,slice:5,x:32:-16,z:-16</id>`.
So the `cellID` that EDM4HEP carries and the ColliderML conversion dropped is this bit field —
note **`module:4`**, i.e. 4 bits = up to 16 modules, independently corroborating 16-fold symmetry.

## What this does NOT explain

**HCAL's residual ~6.6%** (on-grid 0.9344 at the 16-face model). Both endcaps declare the same
`gap="0.25*cm"` between staves and ECAL still reaches 0.9999, so the inter-stave gap is not the
cause. The XML gives no ECAL/HCAL asymmetry that obviously accounts for it. Still open.

Note also `ecal_e_outer_symmetry` / `hcal_e_outer_symmetry` are referenced by the envelope shapes but
defined in neither `OpenDataDetectorEnvelopes.xml` nor `OpenDataDetectorDefs.xml` — unresolved, and
possibly relevant to the boundary behaviour.

## Verdict

**The geometry is now KNOWN, not inferred.** `snap_to_cell` should be built from these constants:
16 faces, envelope rotation `90 - 180/16` deg, cell 5.1 mm (ECAL) / 30 mm (HCAL), 48 / 36 layers at
5.050 / 51.000 mm from zmin 3.2 m / 3.6 m.

The fitted values remain useful as a VALIDATION of the snap once built — an independent check that
the implementation reproduces real cell positions — which is the role the derivation should have had
from the start.

## Next

- Build `snap_to_cell` from the XML constants; use the fit as the acceptance test, not the source.
- Read `ODDPolyhedraEndcapCalorimeter` (the DD4hep plugin) to get each stave's exact placement, which
  would replace the per-face fitted origin with the true one and may close HCAL's 6.6%.
- Re-check whether the barrel is now tractable from the XML: `ecal_b_rmin/rmax` and the same 48-layer
  stack are given, and PIPELINE gap #2 declared the barrel unrecoverable from DATA alone — that
  judgement was made without the XML and should be revisited.
