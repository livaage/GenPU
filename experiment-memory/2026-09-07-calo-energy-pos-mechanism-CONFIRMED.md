# `--energy_pos`: the mechanism WORKED — position reaches the energy distribution on every species

**Date** 2026-09-07 · **Commit** `32d0cab` (+ uncommitted `--energy_pos` head) · **Branch** `flow-response`
**Job** 13556379 (`jobs/calo_epos_coupling_ab.sh`, 14m23s) · script `scripts/calo_energy_position_coupling.py`
Controls: job 13032162, identical script and arms, the pre-`--energy_pos` checkpoints.

## Hypothesis

The `--energy_pos` experiment (train 13038484, eval 13045670) put each cell's own
(depth-within-section, transverse radius) + a 4-way ECAL/HCAL x barrel/endcap token into
`EnergyHead`, whose every input had previously been a per-SHOWER quantity gathered by `[src]` —
so cell energy was independent of cell position BY CONSTRUCTION.

The GATE was measured and moved, but **the mechanism target was never re-measured**, leaving two
incompatible readings alive: "the head learned the coupling and we paid for it elsewhere" vs "the
head never learned it at all". Recorded prediction before this run: **gen rho becomes non-zero and
same-signed as real on every species.**

## Change

None. Re-ran the existing coupling probe on the four `--energy_pos` checkpoints. Both seeds on the
two e± arms (that is where the gate arms disagreed most, 0.941 vs 0.876); pion and photon on seed 0,
matching the control. `null` = real cells with each shower's shape paired to another shower's energy
multiset — i.e. the architecture's own conditional independence, and what the controls sat on.

## Result — CONFIRMED on every arm

`rho(logE, depth)` within shower, and the `p(floor)` core->fringe gradient:

| arm | rho(logE,r) real / **gen** | rho(logE,depth) real / **gen** | p(floor) core->fringe real / **gen** |
|---|---|---|---|
| e± dedicated s0 | -0.0977 / **-0.1000** | -0.0110 / **-0.0166** | x3.55 / **x2.30** |
| e± dedicated s1 | -0.0977 / **-0.0873** | -0.0110 / **-0.0242** | x3.55 / **x2.97** |
| e± pooled s0 | -0.0977 / **-0.0930** | -0.0110 / **-0.0135** | x3.55 / **x2.98** |
| e± pooled s1 | -0.0977 / **-0.1040** | -0.0110 / **-0.0209** | x3.55 / **x3.48** |
| pi± pooled s0 | — | +0.2490 / **+0.1549** | x3.83 / **x2.65** |
| gamma pooled s0 | — | -0.1000 / **-0.0469** | x3.90 / **x3.04** |

**Every control number in the `gen` columns was +0.000 and x1.00.** The `null` side reproduces
+0.000 / x1.00 exactly in this run too, so the estimator is unchanged and the movement is real.

- **e± is essentially exact on the lateral gradient** (-0.093..-0.104 vs real -0.098) and now
  slightly OVERSHOOTS on depth (-0.014..-0.024 vs real -0.011).
- **Partial recovery where the real coupling is large**: pi± recovers 62% of +0.249, gamma 47% of
  -0.100. Signs are right everywhere, including gamma's negative against pi±'s positive — the
  species interaction the design argued for (`cond_embed` carries PDG, the MLP mixes them) is
  happening.
- **The floor gradient is no longer flat**: x1.00 -> x2.3-3.5 against a real x3.5-3.9.

## Verdict

**KEPT as a mechanism result. The gate verdict is a SEPARATE and still-open question.**

This disambiguates the 13045670 gate result: the dedicated-e± regression (gate8 0.811/0.807 ->
0.941/0.876) is NOT a failure to learn the coupling. The head learned it.

**Candidate reconciliation, from the numbers in this run**: real p(floor) runs 0.022 -> 0.078
core->fringe. Dedicated-epos gen runs **0.032** -> 0.073 — the gradient is there but the CORE LEVEL
is ~45% too high. Pooled-epos runs 0.025 -> 0.075, much closer. That matches the gate exactly:
`frac_near_floor` marginal AUC 0.5045 (control) -> **0.6967** (dedicated epos) while the pooled arm
improved. So the head bought the gradient and paid with the level. **Not verified** — no run has
tested it.

## Next

1. The reconciliation above is a hypothesis with an obvious test: compare generated vs real
   OVERALL floor rate, not just the ratio, per arm. Cheap, no retrain.
2. It lands on the **floor redesign already ranked item 0** (2026-08-27): the Bernoulli pins cells
   to a `log_floor` the data never contains, and the gate's band holds 12x more cells than the
   head's. A level error in exactly that branch is what this result predicts. Doing the redesign
   first may make the dedicated-arm regression evaporate without any further position work.
3. `--energy_pos` should NOT be reverted on the strength of the gate alone. It is the only change
   on record that moved the joint mechanism, and the copula half of the gate is the bigger half.
