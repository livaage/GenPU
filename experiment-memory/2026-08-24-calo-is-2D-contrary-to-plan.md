# 2026-08-24 — the calo pipeline is 2D; the plan specified 3D, and the acceptance metrics were never computable

- **Commit / branch**: `89d7711` / `flow-response`
- **Job**: 12880353 (`calo_cell_grid.py`) plus direct inspection of the slice and model.
- Found while trying to build the partition-invariant cell-level metric that
  [the gate-features entry](2026-08-24-gate-features-are-partition-dependent.md) called for.

## Finding

**A cell-level metric cannot be built with the current model, because the model does not generate
depth.**

1. **(eta, phi) is not a grid.** Detector 9: 85,764 distinct eta values across 318,092 cells, median
   spacing 1e-5; fill fraction ~0.000 for every detector. The calorimeter is 3D, and projecting
   cells at different radii/layers into (eta, phi) smears them into a near-continuum. There is
   nothing to snap generated points to.
2. **The model is 2D.** `points_flat` is **(N, 3)** = `(d_eta, d_phi, logE)`; `sample_showers`
   returns `pos (P,2)` + `logE`. `PointFlow` runs CFM on `pts[:, :2]`. A generated shower carries
   nothing that could identify a cell.
3. **The data HAS 3D** — `calo_hits` carries `x, y, z` per cell.
4. **preprocessing.py discards it**: it converts (x,y,z) -> (eta, phi) and keeps only
   `[eta, phi, log_contrib_E, frac, detector]`. Depth is lost at the FIRST step and everything
   downstream inherits it. `detector` (6 values) is the only surviving depth proxy, and the model
   does not generate that either.
5. **The plan specified 3D.** `pileup_generator_plan.md:257` — "Continuous (x, y, z, E) output;
   separate deterministic projection onto cells." Line 356 names the acceptance metrics as
   "layer-wise energy fractions, shower width/depth profiles, cell energy spectrum".

**So neither the projection-to-cells step nor the depth-dependent acceptance metrics were ever
built, and the metrics the plan names as acceptance criteria have never been computable.**

## Consequence

Every calo result on record — the width gate, `frac_near_floor`, the core anchor, the whole August
sequence — was measured on a 2D projection of a 3D object. Not wrong, but not what the plan set out
to evaluate, and not comparable to the CaloClouds line it is benchmarked against (CaloChallenge
datasets are voxelized in layer x radial x angular bins; CaloClouds generates 3D point clouds and
projects to cells; both report longitudinal profiles).

## Fix chain (well-defined, and the data supports it)

1. `preprocessing.py` — keep the depth coordinate (r/z, or x,y,z) in the calo hit record.
2. Re-run preprocessing (validated pattern from job 12873375).
3. `build_calo_slice.py` — carry depth into `points_flat`.
4. `calo_flow.py` — point positions 2D -> 3D (`PointFlow` CFM on `pts[:, :3]`).
5. Metric — real cell projection, layer-wise energy fractions, longitudinal depth profile.

## Sequencing recommendation

**Bundle with re-attribution.** That also requires a preprocessing + slice change, so doing them
separately means two re-preprocessing passes and two retrains. One data change carrying BOTH depth
and re-attribution, then the cell-level metric, then one retrain and gate.

This makes the tower-level (eta, phi)-binned metric unnecessary — it would be superseded within
days. Cost: nothing calo-side is evaluated until the chain completes.

## Note

This does NOT invalidate the tracker work, which is measurement-level and 3D throughout.
