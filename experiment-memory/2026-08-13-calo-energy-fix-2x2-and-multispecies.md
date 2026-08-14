# 2026-08-13 — Calo: energy-fix 2x2 (ABANDONED), physical bounds (KEPT), gate attribution, first multi-species head

- **Date:** 2026-08-13 · **commit:** 9e0bbc2 (working tree; not committed) · **branch:** flow-response
- Companion entry: [species census + energy audit](2026-08-13-calo-species-census-and-energy-audit.md)
  (the measurements that motivated this). Jobs: 12324309/12324310/12324560 (2x2),
  12324681 (clamp), 12324949 (bounds), 12324950/12324951 (multispecies), 12324973 (attribution).

## Hypothesis
From the audit: a shower's energy scale is not a function of the particle (per-shower mean cell
log-E R² = 0.05 from the 7 CONT_FEATURES, 0.85 given (total_logE, log_n)), and the sampled
`GlobalHead` total is a 4x better total estimator than the sum of independently drawn cells
(W/σ 0.017 vs 0.073). So conditioning `EnergyHead` on the global and renormalising cells to the
sampled total should fix the pion's broad energy resolution (1.3-1.8 vs real 0.9-1.2) and its
0.809 gate. Prediction recorded in the companion entry: "D (glob conditioning, no rescale) is the
variant expected to work."

## Change
- `EnergyHead(glob_dim=2)`: optional conditioning on the standardised (total_logE, log_n).
- `CaloFlow.sample_showers()`: ONE generation path for metrics/eval/full-event, with optional
  `partition` (renormalise cells to the sampled total, floor cells pinned, `scale_saturated`
  reported), a per-cell `logE_max` clamp, and an `e_true` energy-conservation bound.
- `--qt_total` (bounded quantile inverse on total_logE), `--no_energy_glob`.
- `calo_metrics.py`: median/IQR response + `respmax` (mean response is tail-dominated),
  per-feature gate AUC, `--logE_max` / `--logE_max_slice` / `--no_econs` / `--no_partition`.

## Result — the fix FAILED, on both species (held-out shard 5)
2x2 of {baseline ckpt, glob-conditioned ckpt} x {sum-of-cells, partition}:

| variant | photon gate | pion gate | pion cell_logE W/σ |
|---|---|---|---|
| A baseline, sum-of-cells | **0.557** | **0.809** | **0.020** |
| D glob-conditioned only | 0.820 | 0.935 | 0.073 |
| B partition only | 0.944 | 0.998 | 0.301 |
| C both | 0.808 | 0.977 | 0.130 |

Every variant degrades the cell-log-E marginal and the gate tracks it monotonically. Causes:
conditioning cells on a SAMPLED global makes them a sharp function of a noisy input (exactly what
the original `EnergyHead` docstring warned about — it had already been discovered once); and a hard
rescale of n i.i.d. cells injects a common log-shift of scatter ~σ/√n_above_floor, worst where n is
small (photon ~4 cells/shower → cell_logE 0.016 → 0.146). The unconditional head wins BECAUSE it is
unconditional: it fits the marginal the gate is made of almost exactly.

**Physical bounds — kept, free:** per-cell clamp at the training slice's hardest cell plus
`E_reco <= E_true` (truth conditioning, so a hard bound at no cost):

| run | gate | worst resp_max | showers touched |
|---|---|---|---|
| photon baseline | 0.557 | 240x E_true | — |
| photon + bounds | 0.558 | **1.00** | 6.3e-05 |
| pion baseline | 0.809 | 657701x E_true | — |
| pion + bounds | 0.809 | **1.00** | 1.3e-05 |

Real showers never exceed E_true (measured `over_etrue_frac_real` = 0.0), so the bound only ever
touches pathology. Note the photon's 240x outlier was NOT a giant cell — the per-cell clamp alone
left it unchanged; it was a very-low-E_true particle handed an ordinary shower, which only the
energy-conservation bound catches.

**Gate attribution (the important result).** Fixing the energy tail by five orders of magnitude
moved the pion gate by 0.000, so the 0.809 is not the energy tail. Single-feature AUCs:

| feature | photon | pion |
|---|---|---|
| frac_near_floor | 0.510 | **0.666** |
| logE_max | 0.506 | **0.614** |
| logE_mean / logE_std / logE_p90 | ≤0.517 | 0.528-0.538 |
| n_cells / log_totE / cells_per_src | ≤0.513 | ≤0.515 |

The pion's gate is carried by the **fraction of cells at the zero-suppression floor** and the
**hardest cell per event** — the DISCRETE floor structure, not the continuous energy scale. The
floor Bernoulli is an independent per-cell draw, so generated per-event floor fractions are too
tightly concentrated while the pooled marginal stays right (cell_logE W/σ 0.020). Also note the
gate's 8 features are all energy/multiplicity — no lateral-shape variable — so the pion's genuinely
bad `shower_width` (W/σ 0.32 vs photon 0.11) is INVISIBLE to it and needs its own metric.

**First multi-species head** (`multispecies_v1`, one PDG-conditioned CaloFlow, all 17 classes,
2.65M showers / 34.2M points, 60k steps, ~11 min on one A100). Per-species held-out gates:

| species | gate | cell_logE | width | µs/particle | vs single-species |
|---|---|---|---|---|---|
| gamma | 0.681 | 0.033 | 0.137 | 6.8 | 0.557 |
| pi | 0.870 | 0.024 | 0.321 | 21.3 | 0.809 |
| electron | 0.888 | 0.022 | 0.177 | 15.1 | — (new) |
| positron | 0.887 | 0.018 | 0.177 | 14.6 | — (new) |
| proton | 0.749 | 0.043 | 0.332 | 9.0 | — (new) |
| rest (11 classes) | 0.626 | 0.011 | 0.244 | 19.9 | — (new) |

It covers 100% of depositing species end-to-end and nothing broke, but shared capacity costs
per-species fidelity (gamma 0.557 → 0.681, pion 0.809 → 0.870) at 240k params / 60k steps.
Caveat found after the fact: this eval clamped EVERY species at the hadronic cell max (-1.13),
4x too loose for EM (photon slice max -2.55) — re-eval with per-class bounds is job 12325597
(`logE_max_pdg` buffer + `--logE_max_slice`).

## Addendum (same session, job 12325597) — per-species bounds + cross-species attribution
Re-evaluated `multispecies_v1` with per-class cell clamps (`logE_max_pdg` fit from the slice:
γ -2.99, e- -3.16, π -2.02/-1.89, p -2.02, µ -3.8, π0 -4.68, vs the single shared -1.13 used before).
**No effect: ms-gamma 0.6806 → 0.6808.** So the multi-species per-species degradation is capacity /
shared representation, not a mis-set bound. Cross-species per-feature attribution:

| run | gate | top feature | 2nd |
|---|---|---|---|
| photon (single) | 0.558 | logE_p90 0.517 (nothing wrong) | cells_per_src 0.513 |
| ms-gamma | 0.681 | **frac_near_floor 0.622** | logE_mean 0.570 |
| ms-proton | 0.749 | **logE_p90 0.570** | logE_mean 0.552 (frac_near_floor 0.528) |
| ms-pion | 0.870 | **frac_near_floor 0.650** | logE_max 0.569 |
| pion (single) | 0.809 | **frac_near_floor 0.666** | logE_max 0.614 |
| ms-electron | 0.888 | **frac_near_floor 0.702** | logE_std 0.585 |

`frac_near_floor` leads 4 of 5 failing cases. **The proton is the exception** — it deposits a median
3 cells (vs 9 for e±/π±), too few for the floor fraction to dominate a per-event statistic, so its
defect is the upper energy percentiles instead. Consequence for planning: the floor fix below should
move e±/π±/γ and should NOT be expected to move the proton.

## Verdict
- **ABANDONED:** glob-conditioned energy head and partition-to-sampled-total. Both degrade the
  cell-log-E marginal more than they buy in totals. Code kept behind `--no_energy_glob` (default
  keeps conditioning OFF for new runs is NOT automatic — pass it) and `--no_partition`.
- **KEPT:** per-cell clamp + `E_reco <= E_true`, and the per-species `logE_max_pdg` variant.
  Free, physical, gate-neutral.
- **KEPT:** the multi-species head as the direction (user decision 2026-08-13: one PDG-conditioned
  head, not per-species checkpoints), with capacity as the open question.

## Next
1. **The pion's real target is the floor structure**, not the energy scale: make the per-shower
   floor-cell fraction a modelled per-shower quantity (e.g. add `frac_floor` as a GlobalHead dim
   and drive the floor Bernoulli from the sampled value) so event-level floor fractions get the
   right shower-to-shower spread. This adds a shower-level latent WITHOUT making continuous energy
   a sharp function of a noisy input — the failure mode that killed variants B/C/D.
2. **Add a lateral-shape feature to the gate** (`shower_width` percentiles). The current gate is
   blind to width, and width is the pion/proton weak spot (0.32/0.33 vs photon 0.11).
3. **Multi-species capacity sweep** (wider heads / more steps / per-class output scaling) to close
   the 0.557 → 0.681 photon gap under the shared head.
4. **Incidence head is still missing** — every calo model/gate is conditioned on `n_cells >= 1`,
   but only 68.7% of particles deposit (species-dependent: e- 50%, e+ 95%, n 35%). Logistic probe
   on the standard contract already reaches AUC 0.897. Required for an honest full-event calo gate.
