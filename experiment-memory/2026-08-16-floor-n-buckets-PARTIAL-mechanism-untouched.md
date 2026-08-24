# 2026-08-16 — floor Bernoulli on a cell-count embedding: mechanism UNTOUCHED, e± gate improved

- **Commit / branch**: `99879a3` / `flow-response` (working tree)
- **Job**: 12473556 (`jobs/calo_floor_nbuckets.sh`, COMPLETED 00:33:19)
- **Checkpoints**: `pion_floorn/checkpoint_040000`, `electron_floorn/checkpoint_060000`
- **Change**: `--floor_n_buckets 32` — the at-floor logit routes through its own net which also sees
  an `nn.Embedding` of the shower's cell count (n clamped to [1,32]); the Gaussian mixture is
  untouched. +9,473 params (240,287 → 249,760). Otherwise byte-identical to the Phase 1 recipe.
- Follows [the floor i.i.d. entry](2026-08-16-floor-iid-test-FALSIFIED-and-n-dependence.md) and
  [the partition A/B](2026-08-16-partition-AB-mechanism-confirmed-delivery-rejected.md)

## Hypothesis

Job 12470841 measured p(floor | n) and found the e± error is ENTIRELY at n=1 (real 0.011 / 0.009 vs
generated 0.061 / 0.062, a 6x excess) with the pion missing a monotone slope (real 0.018 → 0.032,
generated flat). Job 12471293 confirmed the physical reading by forcing it with `partition` (e− n=1
→ 0.011, exactly real) but rejected that delivery, since rescaling every cell's energy detonated the
marginal. So: give the **Bernoulli** a non-linear view of n — a discrete membership decision only,
which cannot blow up `cell_logE` or introduce a shared multiplicative scale.

## Result

**(1) The mechanism did not move.**

| | n=1 real | arm A | **arm B** | n=12 real | arm A | **arm B** |
|---|---|---|---|---|---|---|
| e− | 0.011 | 0.061 | **0.058** | 0.037 | 0.039 | 0.038 |
| e+ | 0.009 | 0.062 | **0.061** | 0.039 | 0.038 | 0.038 |
| pion | 0.018 | 0.016 | 0.015 | 0.032 | 0.019 | **0.019** |

Per-shower floor fraction unchanged to 3 decimals (e− gen 0.0471 both arms); `val_ehl` moved ~0.004,
i.e. noise; `D_floor` gen 1.026 vs 1.031. The head has the capacity, is fed the right variable, and
changed essentially nothing.

**(2) The e± 8-feature gate improved, with no collateral.**

| | gate8 A → B | gateW | `cell_logE` | `frac_near_floor` |
|---|---|---|---|---|
| pion | 0.8055 → 0.8086 | 0.9833 → 0.9832 | 0.0223 → **0.0212** | 0.6176 → 0.6141 |
| e− | 0.7456 → **0.7204** | 0.9639 → 0.9641 | 0.0153 → **0.0145** | 0.6008 → 0.6048 |
| e+ | 0.7488 → **0.7015** | 0.9578 → 0.9589 | 0.0170 → **0.0137** | 0.5924 → 0.5940 |

First change since Phase 1 itself to move the e± gate favourably with `cell_logE` improving on all
three species — and it did so **without** moving `frac_near_floor`, so the gain is not coming from
the thing the head was built for. `logE_p90` drifted slightly worse on e− (0.548 → 0.5625).

## Reading — the attribution was WRONG, and the diagnostic conflated two components

`EnergyHead`'s Bernoulli models a **narrow** pile, `|logE − log_floor| < floor_eps` (0.05). But
`frac_near_floor` — and the probe built to measure it — count a **wide** band,
`logE < log_floor + 0.5`, which is produced by TWO different components: the Bernoulli's exact pile,
and any **mixture** draw that lands low, including the entire physical sub-floor tail that
`EnergyHead.loss` deliberately leaves to the mixture ("the physical SUB-floor tail … is modelled by
the mixture, so we don't clamp it away").

So if the e± n=1 excess is **mixture-tail** rather than pile, no Bernoulli change can ever fix it —
which is exactly what this run shows. It also explains why `partition` could: rescaling cell energies
moves mixture draws out of the band, which a membership flag cannot do.

## Verdict — PARTIAL, and not yet promoted

The change is a plausible small win for an **unclear reason**, and the defect it was built for is
untouched. Single run per arm, no replicate, and single-run gate deltas have misled this project
repeatedly (see the whole 2026-08-14 sequence), so a −0.047 on e+ is not enough to move
"Current best". Flag stays opt-in, default 0 = off (every pre-existing checkpoint rebuilds with it
off; verified `from_checkpoint` infers the bucket count and adds no missing/unexpected keys).

## Next — job 12526752 answers both, and they are independent

1. **Component decomposition** (`calo_floor_dispersion.py` now splits the band into sub-floor /
   pile / just-above, per n bin, real vs gen) on BOTH arms. Settles whether the remaining fix is a
   Bernoulli problem at all. If it is mixture-tail, the target moves to the mixture's lower tail at
   small n — and note the mixture already has an "extra component for the sub-floor tail" (n_mix=4),
   so the question becomes whether it is mis-fit or mis-conditioned.
2. **Seed replicate** (`--seed 1`, new flag, default 0 so nothing else shifts) of arm B on e±.
   Decides whether the gate gain is real.
