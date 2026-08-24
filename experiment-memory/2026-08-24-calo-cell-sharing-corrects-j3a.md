# 2026-08-24 — calo cells have 1.21 contributors on average; J3a's merged cell count was inflated 1.154x

- **Commit / branch**: `b3632cb` / `flow-response`
- **Job**: 12879425 (`calo_cell_sharing.py`, 400 events, 2,102,552 cells)
- Corrects a number in [calo over-splitting](2026-08-24-calo-shower-oversplitting.md) (entries are
  append-only, so the correction lives here).

## Why

`calo_hits.contrib_particle_ids` is a LIST per cell and preprocessing books the cell to EVERY
contributor. J3a measured the effect of re-attribution by SUMMING the cell counts of merged
fragments, which double-counts any cell an ancestor and its descendants both touch.

## Result

| | fraction of cells |
|---|---|
| 1 contributor | **0.8933** |
| >1, SAME calo-incident ancestor (within-shower) | **0.0718** |
| >1, DIFFERENT ancestors (true superposition) | **0.0349** |

Mean contributors per cell 1.206, median 1, p90 2, **max 134**.

**Re-attributed shower cell count: naive SUM 28.82 vs DEDUPLICATED 24.97 = 1.154x inflation.**

## Corrections and consequences

1. **J3a's "e± cells/shower 10.49 -> 20.72" should read ~17.9** (pooled dedup factor applied). The
   over-splitting conclusion is unchanged in direction and magnitude — still ~1.7x with the photon
   control flat at 4.39 — but the quoted figure was 15% optimistic. Any re-attribution
   implementation must DEDUPLICATE cells, not sum them.
2. **Within-shower sharing (7.2%) is 2x the cross-shower kind (3.5%)**, which is the ordering that
   justifies re-attribution deduplicating.
3. **The GlobalHead's `n_cells` target is currently inflated by shared cells.** Modest (89.3% of
   cells are single-contributor) but real, and `n_cells` is the variable `frac_near_floor` depends
   on — the metric at the centre of the whole energy-head thread.
4. **The 3.5% superposition is a PU0 FLOOR, not the operating point.** This is ONE min-bias
   interaction. M3 targets mu = 30/60/140/200, where independent showers overlap far more often, so
   M3's `scatter_add` superposition matters considerably more than 3.5% implies. Worth re-measuring
   at realistic mu before M3 is designed.

## Next

- Apply the deduplicating merge in any re-attribution implementation (do not sum `n_calo_hits`).
- Re-measure superposition at realistic mu when M3 is scoped.
