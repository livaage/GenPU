# 2026-08-24 — calo layer structure: endcaps are EXACTLY layered (83% of energy), barrel EM is not recoverable

- **Commit / branch**: `b3012c0` / `flow-response`
- **Jobs**: 12884069 (bad threshold, discarded), 12884120 (result)
- **Corrects** the blanket "use a continuous coordinate" conclusion in
  [the 3D structure entry](2026-08-24-calo-3d-structure-mapped.md), which was right for the barrel
  and wrong for the endcaps.

## Result

**Endcaps — perfectly discrete:**

| det | distinct depths | gap (p10 = p50 = p99) | fraction of cells on them |
|---|---|---|---|
| 9, 11 (EM) | **exactly 48** | **5.050 mm** | **1.000** |
| 12, 14 (hadronic) | **exactly 36** | **51.000 mm** | **1.000** |

Every gap identical to three decimals; `|z|` recovers the layer index EXACTLY. Endcaps carry ~83%
of deposited energy.

**Barrel EM (det 10) — not layered in r:** 2,776 distinct radii, **max gap 0.985 mm** over a 106 mm
span, no boundary anywhere; 360 depths at 0.1 mm hold only 61.9% of cells. Staves tile a cylinder so
a cell's radius depends on its position ALONG the stave, and cells from different layers overlap in
r. **`r` alone cannot identify the barrel layer, and `calo_hits` carries no layer id** — only
`detector`, `x`, `y`, `z`. Recovering it needs stave geometry that is not in the dataset.
(Det 13, barrel hadronic, does show structure — 39 gaps > 10 mm — but is 0.8% of energy.)

## Methodology note

The first probe (12884069) used `threshold = 20 x median_gap`, which gives 101 mm for endcaps whose
layer spacing IS 5.05 mm — merging all 48 layers into one "cluster" and reporting the opposite of
the truth. Replaced with a threshold-free read of the gap distribution. **A hand-picked cutoff
scaled off the very quantity being measured is self-defeating; print the distribution instead.**

## Representation decision

**Categorical layer index where it exists (endcaps, 83% of energy), continuous depth where it does
not (barrel).** Exact cell identity for the dominant fraction, graceful degradation elsewhere, and
it matches the tracker's proven `layer_class + residual` pattern where the analogy actually holds.

Rejected: continuous everywhere (a continuous flow fits 48 spikes badly); categorical everywhere
(barrel bins would be arbitrary, with no boundary to align to).

## KNOWN LIMITATION

**A fully exact cell-level metric is NOT reachable for barrel EM with this dataset.** Data
limitation, not a design choice. Endcap cell identity is exact.

## Next

- Keep the raw `(r, z)` sidecar as the source of truth; derive layer index at SLICE-BUILD time so
  the representation can change without a data pass (the tracker does exactly this with
  LAYER_MEANS/LAYER_STDS).
- Add to PIPELINE.md §7.
