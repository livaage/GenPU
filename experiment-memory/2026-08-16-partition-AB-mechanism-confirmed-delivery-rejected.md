# 2026-08-16 — partition A/B: the n=1 mechanism is CONFIRMED, the delivery mechanism is REJECTED

- **Commit / branch**: `99879a3` / `flow-response`
- **Job**: 12471293 (`jobs/calo_partition_ab.sh`, COMPLETED 00:23:36) — no training, one flag on the
  existing Phase 1 checkpoints
- **Checkpoints**: `pion_anchor/checkpoint_040000`, `electron_anchor/checkpoint_060000`
- **Artifacts**: `plots/calo/metrics/{floor_dispersion,metrics}_part_{pion,electron,positron}.json`
- Follows [the floor i.i.d. falsification](2026-08-16-floor-iid-test-FALSIFIED-and-n-dependence.md)

## Hypothesis

That entry localised the e± `frac_near_floor` defect entirely to **n = 1**: real p(floor) = 0.011
(e−) / 0.009 (e+) against a generated 0.061 / 0.062, a 6x excess over ~8% of showers each. Proposed
physical reading: a one-cell shower's single cell carries the **entire** shower energy, so it cannot
be a faint fringe cell, and the energy head has no way to know that.

`partition` should fix exactly that by construction. In `sample_showers`, an n=1 all-floor shower has
`sum_rest = 0`, so `ok` is FALSE and the **degenerate** path fires: `s_all = total / e_floor` is
applied to the floor cell itself, lifting it to carry the whole total. (The docstring's "at-floor
cells are pinned so the floor pile survives" holds only in the `ok` branch.) Prediction: n=1 goes
0.061 → ~0, n >= 2 mixed showers unchanged.

## Result — the prediction is confirmed at n=1 and wrong everywhere else

p(floor | n), real vs generated, partition OFF → ON:

| | n=1 real | OFF | **ON** | n=2 real | OFF | **ON** | n=4 real | OFF | **ON** |
|---|---|---|---|---|---|---|---|---|---|
| e− | 0.011 | 0.061 | **0.011** | 0.052 | 0.064 | 0.075 | 0.061 | 0.060 | 0.085 |
| e+ | 0.009 | 0.062 | **0.006** | 0.052 | 0.064 | 0.073 | 0.063 | 0.064 | 0.084 |
| pion | 0.018 | 0.016 | 0.007 | 0.018 | 0.015 | 0.040 | 0.022 | 0.016 | 0.053 |

**e− n=1 lands exactly on truth (0.011 vs 0.011)**; e+ slightly undershoots. The 6x excess is gone by
precisely the predicted mechanism. The pion, which had no n=1 problem (0.016 vs 0.018), is pushed too
low — as expected, since the fix is indiscriminate.

**But n >= 2 gets substantially WORSE on every species** (pion n=4: 0.016 → 0.053 vs real 0.022). The
reason is the same code path: in the non-degenerate `ok` branch the above-floor cells absorb
`total − sum_floor`, so whenever the sampled total is below the sum of drawn cells every cell is
scaled DOWN and some cross into the near-floor band. Partition manufactures fringe cells.

It also **inverts the dispersion**: generated `D_floor` 1.031 → **3.805** (pion), 1.024 → 2.09 (e−),
against real 1.199 / 1.035. One shared multiplicative scale correlates all of a shower's cells at
once, so we go from slightly under-dispersed to badly over-dispersed.

Pooled cost, worse than the 2026-08-13 rejection:

| | gate8 OFF → ON | gateW | `frac_near_floor` | `cell_logE` |
|---|---|---|---|---|
| pion | 0.8055 → **0.9977** | 0.9833 → 0.9988 | 0.6176 → 0.9722 | 0.0223 → **0.309** (14x) |
| e− | 0.7456 → **0.9882** | 0.9639 → 0.9937 | 0.6008 → 0.9585 | 0.0153 → 0.1194 (8x) |
| e+ | 0.7488 → **0.9903** | 0.9578 → 0.9941 | 0.5924 → 0.9591 | 0.0170 → 0.1212 |

## Verdict — mechanism KEPT, delivery ABANDONED

This was a **mechanism test, not a candidate fix**, and it did its job: it confirms the physical
reading of the n=1 defect at the one bin that mattered, and shows the fix cannot be a
generation-time global rescale that touches every cell's continuous energy. `partition` stays off,
now for a second independent reason.

## Next — the training-time version, through the floor Bernoulli only

Give the energy head a **non-linear view of n**, feeding only the floor logit: an embedding over
cell-count buckets rather than the `log_n` scalar (`--energy_glob_idx 1`, worth only 0.618 → 0.574
because a single scalar into an MLP dominated by large-n showers under-fits the small-n corner).

Why this cannot repeat the failure above: it changes a **discrete membership decision**, never a
cell's energy, so it cannot produce the `cell_logE` blowup or the shared-scale correlation. This is
also the distinction already argued in `EnergyHead`'s own comment — the ruled-out 2026-08-13 failure
was dim 0, `total_logE`, making continuous cell energy a sharp function of a noisy energy SCALE;
a count is a different quantity and it is the floor Bernoulli that needs it.

Target curve is unusually well specified — hit real p(floor | n) per species:
e± ~0.010 at n=1, ~0.052 at n=2, rising to ~0.062 by n=3-4 then flat ~0.038 by n>=9;
pion a monotone climb 0.018 → 0.032 across n=1..12 (its defect is a MISSING SLOPE, not a corner).
