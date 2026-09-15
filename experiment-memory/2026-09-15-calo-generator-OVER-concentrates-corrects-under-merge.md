# CORRECTION: the generator OVER-concentrates, it does not under-merge — and the reference was the wrong number

**Date** 2026-09-15 · **Commit** `f9310b6` · **Branch** `flow-response`
Login node, real data, no model, no job. **Corrects
[the projection entry](2026-09-15-calo-cell-projection-first-gate-and-a-preexisting-wrap-bug.md)**,
which is left in place per the append-only rule. Where they disagree, this one is right.

## What the projection entry claimed, and why it was wrong

It concluded: *"The generator UNDER-merges: co-occupancy 3.09% against the 7.2% within-shower
sharing ... its points are too spread out to share a channel."*

**The 7.2% is the wrong reference.** That figure (2026-08-24 cell-sharing entry) counts raw
CONTRIBUTIONS sharing a cell — deposits from different mid-shower fragments of the same incident
particle. The v2 slice's re-attribution **already merges those**, so the slice stores one row per
DISTINCT CELL and the model's `log_n` is trained to emit a count of distinct cells. The correct
reference is therefore ~0, not 7.2%.

Measured directly. Real slice, 200,000 showers / 4,986,861 rows, pushed through the same
`etaphidepth_to_xyz -> snap_cells -> cell_ids` path the generated arm uses:

| | collision rate |
|---|---|
| **real slice cells** | **0.0003** |
| **generated (job 13929818)** | **0.0361** |

**120x higher.** Real rows are all distinct in the slice's own coordinates (duplicate rate exactly
0.0000 on `(d_eta, d_phi)` and on `(d_eta, d_phi, depth)`), so the 0.0003 is our cell map's own error
floor, not physics. Per detector the real floor is 0.0002-0.0003 for ECAL and 0.0025-0.0027 for the
HCAL endcaps, consistent with the known boundary-assignment residual there.

**So the generator OVER-concentrates**: it places 3.6% of its points close enough together that the
detector would read them as one channel. The direction is the opposite of what was logged.

Everything else in the projection entry survives and is now better explained:
- `cells_per_shower` W 0.0104 -> 0.0202 — merging removes cells the model intended to emit, pushing
  the count below both the model's own target and the real distribution.
- energy marginals worsening (`logE_p90` +0.106, `logE_mean` +0.078) — a merged cell carries the
  sum, so per-cell energy rises above real.
- The verdict stands: the projection EXPOSES a defect the continuous representation hid. Only the
  mechanism was stated backwards.

## Three of my own errors produced this, all worth recording

1. **Wrong reference number.** 7.2% was within-shower CONTRIBUTION sharing, already merged away by
   re-attribution. Picking a number from STATUS without checking what population it counted.
2. **Double-standardising `cont`.** The probe computed `p_eta = cont[:,1]*cont_std[1] + cont_mean[1]`,
   but the slice stores `cont` RAW — `calo_metrics.py:172` correctly uses `cont[:,1]` and the model
   standardises internally. Applying the transform to raw values inflated |eta| to a median of 4.5
   and put 83% of cells inside the beam pipe (r median 71 mm against an ECAL endcap inner radius of
   315 mm), which read as a 28% "collision rate".
   **`calo_metrics` does NOT have this bug, so no job result is affected** — only this probe was.
3. **Blaming the composition before the inputs.** With `p_eta` wrong, all four candidate compositions
   of (particle, anchor, core residual, cell offset) looked broken, and I began doubting the
   composition. With raw `p_eta` the ORIGINAL composition — `p + anchor + core_residual + d` — is
   right, and the test that proves it is physical: r p1 = 319.4 mm against the XML's
   `ecal_e_inner_radius = 315 mm`, with **0.0000** of cells below it.

The general lesson: a geometry probe has a free physical check available — does the reconstructed
position lie inside the detector. Running that first would have caught the inflated eta immediately,
instead of after two wrong hypotheses.

## Verdict

The generator places too many deposits in one readout channel. This is newly measurable — it was
invisible while cells were a continuous cloud — and it is a cell-scale statement of the same thing
the coherence probe says at shower scale: generated showers do not distribute energy the way real
ones do. Whether the two are the same defect is still NOT verified.

## Next

- **Where do the collisions sit?** The obvious candidate is the shower core. Needs generated cells,
  so it needs a GPU job: bin collisions by distance from the core and compare with the real
  nearest-neighbour cell spacing within a shower, which can be measured from the slice for free.
- If collisions concentrate in the core, the defect is core density, which the PointCFM sampler
  controls directly, and it is testable without retraining by resampling with a widened core.
- Do NOT "fix" the merge. Merging is the physically correct response to two deposits in one channel;
  the error is upstream, in placing them there.
