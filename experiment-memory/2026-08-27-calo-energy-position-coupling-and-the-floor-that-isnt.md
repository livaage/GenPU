# 2026-08-27 — cell energy is blind to cell position, and the "floor pile" does not exist

- **Commit / branch**: `32d0cab` / `flow-response`
- **Job**: 13032162 (`jobs/calo_epos_coupling.sh`, COMPLETED 00:13:28, diagnostic only — no training)
- **New**: `scripts/calo_energy_position_coupling.py`
- **Ckpts scored**: `electron_v2_s0/checkpoint_060000`, `multispecies_v2_s0/checkpoint_060000`
- **Artifacts**: `plots/calo/metrics/epos_coupling_{ele_dedicated,ms_ele,ms_pion,ms_photon,survey}.{json,png}`

## Hypothesis

The 2026-08-25 decomposition put the v2 calo defect majority in the JOINT (copula 0.70-0.72 vs
marginals 0.60-0.62) and showed the copula half did NOT move when pooling moved the marginal half by
0.13. Nothing tried since 2026-08-13 could have moved it. Proposed mechanism: `EnergyHead._out`
takes only `(cond_embed, glob_std[:, idx], n_cells)` (`calo_flow.py:192`), and in
`sample_showers` all three arrive as `[src]`-gathered PER-SHOWER values — so every cell of a shower
receives a **bit-identical input vector**. Energy is therefore independent of position within a
shower by construction, and no amount of energy conditioning can change that: the head emits one
distribution per shower and draws n i.i.d. samples from it.

## Change

`scripts/calo_energy_position_coupling.py`, validated on synthetic arms with known answers before
being pointed at real data. Measures, per species, real vs generated:
- **A** p(floor) and `<logE>` vs the cell's within-shower RANK of r and of depth
- **B** within-shower Spearman rho(logE, r) and rho(logE, depth)
- **C** shower-level partial Spearman of a SHAPE against an ENERGY quantity, controlling (log_n, log_totE)

## Result — the gradient is universal, species-specific, and the model emits ZERO

| class | showers | rho(logE, **depth**) real | gen | rho(logE, r) real | p(floor) fringe/core |
|---|---|---|---|---|---|
| p | 66,604 | **+0.542** | — | +0.068 | x4.51 |
| mu+ / mu- | 24,843 | **+0.457 / +0.440** | — | +0.202 / +0.184 | x0.89 / x2.55 |
| K+ / K- | 29,283 | +0.290 / +0.185 | — | +0.126 / +0.062 | x3.45 / x3.66 |
| pi+ / pi- | 373,656 | +0.271 / +0.226 | **+0.000** | +0.107 / +0.077 | x3.87 / x3.81 |
| n / nbar | 57,395 | +0.129 / -0.007 | — | -0.025 / -0.017 | x1.72 / x2.67 |
| e- / e+ | 400,000 | -0.013 / -0.010 | **+0.000** | -0.097 / -0.099 | x3.49 / x3.56 |
| gamma | 200,000 | -0.101 | **+0.000** | -0.130 | x3.92 |
| **ALL pooled** | 200,000 | +0.067 | — | -0.042 | x3.63 |

Four species were generated directly; all returned flat to ±0.001, confirming the architectural
reading. The e± profile in full (p(floor) by within-shower r-rank, core -> fringe):

```
real 0.022 0.018 0.019 0.021 0.025 0.031 0.040 0.052 0.065 0.078
gen  0.037 0.038 0.038 0.038 0.039 0.038 0.038 0.038 0.038 0.037
```

`<logE>` runs -7.86 -> -8.25 across those bins in real and **-8.00 flat to ±0.01** in generated,
while the overall floor rate matches to four decimals (0.0379 vs 0.0378). The floor RATE is right and
its PLACEMENT is absent — not approximate, absent.

**The sign flips by species and the effects CANCEL when pooled** (-0.042 / +0.067, smaller than any
single class). A shared head that learned "the average gradient" would be wrong for everyone; the
position input must interact with PDG, which `cond_embed` already carries.

This statistic is multiplicity-independent, so unlike the per-species gate numbers it IS comparable
across species — which answers the caveat raised in the 2026-08-25 multispecies entry about p and n
looking artificially good at 7.5-9.9 showers/event.

## The bigger finding — the zero-suppression PILE does not exist

`EnergyHead` models the low-energy region as an at-floor Bernoulli whose fired cells are pinned to
exactly `log_floor` (`calo_flow.py:229`), documented as "the exact zero-suppression pile
(single-contributor cells at 5e-5)". Measured from the raw source (`calo_hits.total_energy`, shard 0,
1.53M cells):

```
min>0 = 5.0001e-05 GeV      frac below 5e-5 = 0.00000
distinct 1,493,002 / 1,528,778       most frequent value x17
density across log(5e-5) per 0.1 bin:  0.0000  0.0000 | 0.4810  0.5239
```

The cutoff constant is right — **5e-5 GeV = 50 keV, applied to the CELL TOTAL** — but it is a **hard
truncation, not a pile**. Zero suppression discards sub-threshold cells; it does not stack them on
the threshold. And the premise fails in the slices too, in **both** versions:

| | v1 `electron_anchor` | v2 `electron_v2_h5` |
|---|---|---|
| cells exactly at log(5e-5) | **0** | **0** |
| distinct logE values | 3,782,896 / 8.5M | 4,635,428 / 13.1M |
| sub-floor tail (E < 5e-5) | 1.06% | 0.39% |
| head band \|logE-LF\|<0.05 | 0.299% | 0.302% |
| gate band < LF+0.5 | 4.10% | 3.79% |

The sub-floor tail is per-shower ATTRIBUTED energy being a *share* of the cell total, so a shared
cell's share falls below 5e-5 though its total cannot; v2's tail is smaller than v1's because
re-attribution merges contributions v1 split. What the slice shows at LF is a **density step** (0.033%
per 0.1 bin below, 0.583% above, ~17x) — the truncation edge of the unshared population.

**Consequence.** The model places ~0.56% of its cells at a single value where real data has zero
(gen head band 0.0056 vs real 0.0030), and the two bands are 12x apart in population: the gate's
`frac_near_floor` scores 3.79% of cells while the head's Bernoulli trains on 0.30%, so **92% of what
the gate measures is drawn by the MIXTURE, not the floor logit**. `--floor_n_buckets` routes only the
at-floor logit through its own net (`calo_flow.py:205`). Every floor-branch experiment since
2026-08-13 was therefore tuning the rate of an atom that should not exist, on a branch governing 0.3%
of cells, while the metric measured 3.8%.

## Also settled — the calorimeter is NOT phi-symmetric

The `cont` contract omits phi, vx, vy deliberately ("response is phi-invariant"), which is why the
helix anchor is computed outside the model. Measured on raw `calo_hits` (1200 events, 6.46M cells):

| | occupancy vs phi, rms/mean | min/max | strongest harmonic |
|---|---|---|---|
| endcap (9,11,12,14) | 0.153 | 0.62 / 1.69 | **n=80** (amp 0.143), then 160, 144, 112, 32 |
| barrel (10,13) | 0.084 | 0.79 / 1.31 | n=128, 2, 20 |

Barrel median r modulates 1288.1 -> 1310.1 mm (**22 mm**) with phi — the stave polygon, the same
geometry PIPELINE.md invokes for barrel EM layer irrecoverability. An 80-fold harmonic is not
physics; min-bias pileup is phi-uniform on average. phi-invariance is currently **harmless** (the
model emits continuous offsets and produces no cell-level geometry at all) but becomes **binding**
the moment the cell-grid projection of gap #1 exists.

## Verdict — KEPT as a method; two self-corrections recorded

The one-point measurement stands and is replicated across species and both charges. Two things in
the script were WRONG and are recorded so no one trusts them:

1. **Part C does not isolate cell-level coupling.** Controlling for (log_n, log_totE) leaves the
   shared dependence on the PARTICLE conditioning, which both the point flow and the energy head
   see. The model already reproduces most shape x energy partial correlations
   (`depth_mean x logE_mean` real -0.173 / gen -0.143) and on the floor pairs has **3x too much**
   (`depth_mean x frac_floor` real +0.023 / gen +0.068). The residual n-dependence is also nonlinear
   and a rank-linear control removes only the linear part. Part C measures shared conditioning.
2. **The cross-shower null was index-aligned** — cell k of the recipient took cell k of the donor —
   so it inherited the real gradient through stored cell order (null rho(logE, r) = -0.062 against a
   real -0.098, where a null must be 0). Parts A and B were affected; part C was not, since it
   depends only on the multiset. FIXED (block order now scrambled): null rho -> +0.0006, p(floor)
   ratio -> x1.00, and `width__frac_floor` unchanged at +0.1059, confirming C did not move.
   The valid null for A/B throughout was `gen`, which is the architecture's own zero.

## Next

- **The floor redesign outranks the position conditioning**: it is a falsified premise, not a missing
  refinement. Drop the point mass; use a mixture TRUNCATED at log(5e-5), and re-purpose the Bernoulli
  to the physically real binary (is this cell's attributed share sub-threshold, 0.39%).
- **`partition` breaks a truncated mixture**: it rescales non-floor cells by `s_rest` AFTER the draw,
  moving the edge to `log_floor + log(s_rest)` per shower. Preferred fix is to generate energies as
  FRACTIONS on the simplex (softmax/Dirichlet over the shower's cells), which makes the sum exact by
  construction, deletes `partition` and its `scale_bounds`/saturation branches entirely, and keeps the
  edge sharp — the head already sees `total_logE` so it can place the edge at 5e-5/total.
- Position conditioning goes into `self.net` (the mixture), NOT the floor branch, and must interact
  with PDG. Use ABSOLUTE standardised depth first: a within-shower NORMALISED depth couples the cells
  through the empirical moments, and 2026-08-16 measured D_floor,real = 1.035 for e±, so real showers
  have little shared fluctuation to spend.
- Open and unmeasured at the time of writing: the TWO-POINT function. Per-cell conditioning fixes the
  profile but leaves a salt-and-pepper realisation where real showers have a contiguous hot core.
  `scripts/calo_shower_coherence.py` + job 13033389 test exactly that.
