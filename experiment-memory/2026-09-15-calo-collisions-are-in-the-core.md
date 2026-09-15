# Cell collisions are a CORE effect — 1000x gradient from core to fringe — but the i.i.d. sampler is an unexcluded cause

**Date** 2026-09-15 · **Commit** `d71f661` · **Branch** `flow-response`
**Job** 13931553 (`jobs/calo_cooccupancy.sh`, 36s) · ckpt `multispecies_v2_s0/checkpoint_060000`
200,000 showers · `--max_cells 4096` · out `calo_geom/calo_cooccupancy.json`

## Hypothesis (pre-registered in the job script)

Generated cell-collision rate is 0.0361 against a real floor of 0.0003 (120x), so the generator
places deposits too close together. Recorded before the run:
- rising toward `r/width = 0` -> the CORE is too dense, testable by resampling, no retrain;
- flat -> a global density error (would be surprising, `shower_width` W/sigma is only 0.0505);
- rising at large `r/width` -> not anticipated.

## Result — CONFIRMED, and sharper than expected

Sanity checks first: real collision 0.0003, endcap cells inside r = 315 mm **0.0000**, real
within-shower NN spacing median 0.00314 (matching the 0.00318 measured independently on the login
node). Generated collision 0.0348 on 5,029,696 points.

| r / width | points | collision rate |
|---|---|---|
| 0.00 - 0.25 | 954,078 | **0.1168** |
| 0.25 - 0.50 | 1,114,729 | 0.0837 |
| 0.50 - 0.75 | 878,451 | 0.0596 |
| 0.75 - 1.00 | 1,014,445 | 0.0478 |
| 1.00 - 1.50 | 735,916 | 0.0250 |
| 1.50 - 2.00 | 143,106 | 0.0018 |
| 2.00 - 3.00 | 112,531 | 0.0005 |
| 3.00 - 5.00 | 59,111 | 0.0001 |
| 5.00 - 100 | 17,329 | 0.0001 |

**Monotone over three orders of magnitude**: 11.7% in the core, 0.01% in the fringe. No flat
component, and nothing at large radius. The "global density error" and "fringe" branches are both
excluded.

## THE CONFOUND — do not read this as "the core density is wrong" yet

`PointCFM.sample` draws each cell from its own `torch.randn` and integrates a velocity field that
sees only that cell (`calo_flow.py:114`). The cells of a shower are therefore **i.i.d. given the
conditioning** -- and i.i.d. draws collide EVEN AT EXACTLY THE RIGHT DENSITY, by the birthday
argument. Real cells cannot: a real shower is a SET of distinct channels, i.e. a draw without
replacement, and its collision rate is 0 by construction.

So the profile is equally consistent with:
  (a) the core density being too high, and
  (b) the core density being exactly right, with every collision an artifact of sampling n points
      i.i.d. instead of choosing n distinct cells.

Both predict a monotone rise toward the core, because the core is where density is highest and the
birthday effect strongest. **This experiment cannot separate them**, and the job script's
pre-registered reading ("rising toward r=0 -> the core is too dense") was written without noticing
that (b) exists. That reading is too strong.

If (b) dominates, this is the SAME architectural fact as the missing two-point coherence -- cells
that do not see each other -- showing up at cell scale, and no amount of density tuning fixes it.
That link was previously speculative; under (b) it would be mechanical.

## The measurement that separates them

Compare the within-shower nearest-neighbour SPACING, real vs generated, at matched cell count.
Real is already measured (median 0.00314, p10 0.00128, p90 0.03707). If generated spacing matches
real, the density is right and the collisions are the i.i.d. artifact -> (b). If generated spacing
is systematically smaller, the cloud really is too tight -> (a).

`scripts/calo_cell_cooccupancy.py` computes this statistic for the REAL side only -- an omission,
since it is exactly the discriminator and the generated side was already sampled in this job. One
line to add, and the job is 36 seconds.

## Verdict

The collisions are a core phenomenon; that much is settled and the two alternative branches are
excluded. WHY the core collides is not settled, and the leading candidate is now the sampler rather
than the density.

## Next

- Add generated NN spacing to the script and re-run (36s) -- the discriminator above.
- Do NOT tune core density until that runs; under (b) it would be fitting a symptom.
- If (b): this is a second, independent argument for the set/attention head, and a stronger one
  than the coherence result because it is mechanical rather than statistical.
