# 2026-08-24 — floor band decomposition: the excess is MIXTURE TAIL, and the floor_n_buckets gate gain was seed noise

- **Commit / branch**: `99879a3` / `flow-response` (working tree)
- **Job**: 12526752 (`jobs/calo_floor_components.sh`, COMPLETED 2026-08-17 09:00) — results sat
  unlogged for a week; logged here on 2026-08-24.
- **Probe**: `scripts/calo_floor_dispersion.py`, extended to split the near-floor band into three
  components per n bin (real vs gen), plus a `--seed 1` replicate of arm B.
- Closes the two follow-ups left open by
  [the floor_n_buckets entry](2026-08-16-floor-n-buckets-PARTIAL-mechanism-untouched.md).

## Hypothesis

Two independent questions, both raised by the Aug-16 run:

1. `frac_near_floor` scores a WIDE band (`logE < log_floor + 0.5`), while `EnergyHead`'s Bernoulli
   models only a NARROW pile (`|logE - log_floor| < 0.05`). If the e± n=1 excess is fed by low
   **mixture** draws rather than the pile, no Bernoulli change can ever reach it — which would
   explain why `--floor_n_buckets 32` moved p(band | n=1) by 0.003, and why `partition` (which
   rescales cell energies) could.
2. The same run moved e− gate8 0.7456 → 0.7204 and e+ 0.7488 → 0.7015 *without* moving
   `frac_near_floor`, i.e. a favourable move for an unknown reason, one run per arm. Replicate.

## Result

**(1) It is the mixture tail. Confirmed, and cleanly localised.** Band decomposition at n = 1,
electron (real / gen):

| component | real | gen | ratio |
|---|---|---|---|
| **sub-floor (mixture)** | 0.0019 | **0.0368** | **19x** |
| pile (the Bernoulli) | 0.0007 | 0.0054 | 7x, but 7% of the excess in absolute terms |
| just-above (mixture) | 0.0086 | 0.0184 | 2x |

The n=1 error is ~90% owned by the **mixture's sub-floor draws**, not the Bernoulli. By n ≥ 5 the
sub-floor component agrees to ~0.001 (real 0.0272 / gen 0.0253) and stays matched out to n = 24.
Pion pooled: sub 0.0155/0.0187, pile 0.0027/0.0052, above 0.0227/0.0237 — no n = 1 corner, matching
the Aug-16 reading that the pion's defect is a missing slope, not a small-n spike.
`D_floor` real 1.035 vs gen 1.024 (e±) / 1.199 vs 1.031 (pion): **"NOT structural on the floor
count — mean/conditioning problem"**, consistent with the ruled-out i.i.d. test.

**(2) The gate gain was seed noise.** Arm B, e±, seed 1 vs seed 0:

| | Phase 1 baseline | arm B seed 0 | **arm B seed 1** |
|---|---|---|---|
| e− gate8 | 0.7456 | 0.7204 | **0.8272** |
| e+ gate8 | 0.7488 | 0.7015 | **0.8133** |
| e− gateW | 0.9639 | 0.9641 | 0.9693 |
| e+ gateW | 0.9578 | 0.9589 | 0.9637 |

Seed 1 is worse than the Phase 1 baseline on both species, and the seed-to-seed spread (0.107 on
e−, 0.112 on e+) is an order of magnitude larger than every "win" this project has promoted on a
single run. `frac_near_floor` 0.592 (e−) / 0.587 (e+), i.e. unchanged, as on seed 0.

## Verdict — ABANDONED (`--floor_n_buckets`), and a methodology correction

`--floor_n_buckets` is dead: it does not move its mechanism target (0.003), and its apparent gate
gain does not survive a replicate. Flag stays opt-in at default 0.

The larger result is the **seed spread itself: ~0.11 on e± gate8**. Every single-run gate delta on
record smaller than that is uninterpretable, which retroactively covers most of the 2026-08-14
sequence. Two-seed minimum from here on any gate claim.

## Next

- The energy-head target is now exact: **the Gaussian mixture's lower tail at n = 1 for e±**. The
  mixture already carries a dedicated sub-floor component (`n_mix=4`), so the question is whether it
  is mis-fit or mis-conditioned — and it is conditioned on nothing that distinguishes n = 1, since
  `--no_energy_glob` is the default. A one-cell shower's cell carries the ENTIRE shower energy, so
  the correct constraint is a bound, not a bucket: at n = 1, `cell_logE` must equal `total_logE`.
  That is what `partition` enforced globally and why it worked at n = 1 and detonated everywhere
  else. **An n = 1-only constraint is the untried version.**
- Any retest needs >= 2 seeds per arm.
