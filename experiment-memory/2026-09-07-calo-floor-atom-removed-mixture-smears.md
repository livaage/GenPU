# Dropping the floor atom: the atom is NOT load-bearing, but the mixture smears the edge

**Date** 2026-09-07 · **Commit** `32d0cab` (+ uncommitted `--no_floor_atom`) · **Branch** `flow-response`
**Job** 13557402 (`jobs/calo_noatom_smoke.sh`, 14m06s) · ckpt `ele_noatom_s0` · **1 seed**
Control: `electron_v2_s0/s1` (job 12930682), same slice, same recipe, atom ON.

## Hypothesis

The at-floor Bernoulli pins cells to exactly `log_floor`. Measured this session on 13,054,567
held-out e± cells (`electron_v2_h5.npz`): **0.000000** sit there (exact, `|logE-LF| < 1e-6`),
**0.387%** sit BELOW, and the threshold is a **x19 density step** (4,336 -> 82,294 per 0.1 bin).
The mixture beside the atom is trained with a hole cut out (`cont` mask) that the atom fills back
in — circular, and the likeliest reason ten experiments since 2026-08-13 could not move
`frac_near_floor`.

Recorded before the run: `at_floor_gen` -> 0 is mechanical. The real unknown is `sub_floor_gen`
against real 0.00387 — near 0.004 means a Gaussian mixture can hold the edge; 0.01-0.03 means it
smears, and the fix is to TRUNCATE at the threshold, not to restore the atom.

## Change

`--no_floor_atom` (`calo_flow.py`): no Bernoulli, no BCE term, mixture trained on ALL cells, no
point-mass override at sampling. **Output width deliberately unchanged** (`1 + 3*n_mix`) so the A/B
is architecture-matched and every existing checkpoint still loads — and loads with the atom ON,
since `floor_atom_on` is absent from them (`from_checkpoint` defaults it to 1.0, the only flag
there that defaults on). `floor_n_buckets > 0` with the atom off now raises.
New `floor_edge` block in `calo_metrics.py`: at / sub / near-floor fractions, real vs gen — the
gate's `frac_near_floor` band is 0.5 wide and cannot see where inside it the cells sit.

## Result

| | real | control (atom ON) | **no atom** |
|---|---|---|---|
| `at_floor_gen` | 0.000000 | ~0.0056 *(recorded 2026-08-27, not re-measured here)* | **0.000000** |
| `sub_floor_gen` | 0.00387 | not measured | **0.01032** (**x2.7 too many**) |
| `near_floor_gen` | 0.03793 | not measured | **0.03934** (x1.04) |
| `cell_logE` W/sigma | — | 0.0280 / 0.0262 | **0.0244** (best on record) |
| gate8 | — | 0.8105 / 0.8073 | **0.8169** |
| `frac_near_floor` AUC | — | 0.5045 / 0.5461 | **0.5394** |
| full / marg / copula | — | 0.8253 / 0.6007 / 0.7004 | **0.8418 / 0.6360 / 0.7362** |

**Two findings, opposite directions:**

1. **The atom is not load-bearing.** Removing it left the gate essentially where it was (+0.006 on
   gate8, ~2x the control's own 0.003 seed spread, on ONE seed) and **improved the per-cell energy
   fit** — `cell_logE` 0.0280/0.0262 -> **0.0244**, the best recorded. `frac_near_floor` AUC 0.5394
   sits INSIDE the control's own seed range (0.5045-0.5461). Ten experiments' worth of machinery
   can go and nothing of value is lost.
2. **The predicted smearing is real.** `sub_floor_gen` **0.01032** vs real 0.00387 — the mixture
   puts **2.7x too many** cells below a threshold that in the data is a hard cut. It is at the low
   end of the pre-registered 0.01-0.03 "smearing" band, not the 0.004 "holds the edge" outcome.
   `near_floor` is nearly exact (1.04x), so the error is concentrated exactly at the edge.

## Verdict

**KEEP the removal; the atom is deleted on evidence, not assumption. The replacement is TRUNCATION,
which the pre-registered criterion now selects.** A sum of Gaussians cannot represent a
discontinuity; normalising the mixture on `[log_floor, inf)` makes the edge exact by construction
and costs no new component — the head still ends up SIMPLER than it is today.

The 0.387% genuinely below the threshold is real physics (a shared cell whose ATTRIBUTED share
falls under, while the cell total does not) and wants one small ordinary mixture component, not a
special mechanism.

**Caveats, stated plainly**: one seed, so the +0.006 gate and the marginal/copula rises
(+0.035/+0.036) are not separable from noise at the control's spread — this run was designed to
answer the smearing question, which needs no seeds (a fraction over 13M cells), not the gate
question, which needs two. And the control's `sub_floor`/`near_floor` are UNMEASURED: the
`floor_edge` diagnostic is new, so the x2.7 is against REAL, not against the atom-on model. Re-run
`calo_metrics.py` on `electron_v2_s0` (~7 min) to close that.

## Next

1. **Truncate the mixture at `log(5e-5)`** — the indicated fix, now with a measurement behind it.
2. Re-run the control eval for its `floor_edge` numbers, so the comparison is model-vs-model.
3. Only then the 2x2 against `--energy_pos`: today's other result showed the position fix got the
   floor GRADIENT right (x1.00 -> x2.3-3.5 vs real x3.5) and, on the dedicated arm, the LEVEL wrong
   (core 0.032 vs real 0.022). Both live in this branch. Measuring that gate through a known-smeared
   edge would waste the run.
