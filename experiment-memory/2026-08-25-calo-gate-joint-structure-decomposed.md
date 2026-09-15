# 2026-08-25 — the v2 calo gate is majority JOINT structure, and `frac_near_floor`'s CORRELATIONS carry it

- **Commit / branch**: `32d0cab` / `flow-response`
- **Jobs**: 12930682 (gate diagnosis, both e± v2 seeds), 12930683 (all-species v2 census, shard 0)
- **Ckpts**: `electron_v2_s0/checkpoint_060000`, `electron_v2_s1/checkpoint_060000` (unchanged)
- **New**: `scripts/calo_gate_diagnose.py`; `calo_metrics.py --dump_features` + species filter
- **Artifacts**: `plots/calo/metrics/gate_diag_electron_v2_s{0,1}.json`,
  `calo_slice/multispecies_v2_census.npz`

## Hypothesis

The 2026-08-24 entry recorded "composite gate 0.81 while no single feature exceeds 0.58 — the
classifier is using multivariate structure, so per-feature AUCs alone will not say what to fix",
and stopped there. But that reading is not the only one. A composite far above every marginal can
come from either of two causes, and they call for opposite fixes:

- **AGGREGATION** — many independent small marginal offsets adding in quadrature. Nothing about the
  joint is wrong; there is simply no single feature to chase, and the fix is broad calibration.
- **DEPENDENCE** — the marginals are right and the JOINT is wrong. Every single-feature AUC sits at
  0.50 and the composite still climbs. The fix is whatever couples the features.

Prediction written before running: an equal-variance Gaussian model of pure aggregation, given the
12 measured per-feature AUCs, predicts a composite of **0.627**. Observed is 0.81-0.83. So
dependence should be present — but how much, and in what, was unmeasured.

## Change

`scripts/calo_gate_diagnose.py` decomposes the composite by destroying one cause at a time, using
the *same* classifier `calo_metrics.py` uses (50/50 split, 2x64 SiLU MLP, 800 Adam steps, held-out
rank AUC), so every number below is comparable to the ones in `metrics_*.json`:

- **marginals only** — permute each column WITHIN its own class: dependence destroyed, marginals exact.
- **copula only** — rank-transform each column WITHIN its own class: marginals made identical
  (uniform), dependence untouched.
- **copula, quadratic logistic** — a second-order discriminant in copula space, i.e. exactly the
  part of the dependence gap that lives in the CORRELATION MATRIX and nothing beyond it.
- **copula, linear logistic** — a control that MUST land at 0.5, since after rank-matching the
  class means are equal by construction.

Validated on two synthetic arms with known answers before it was pointed at real data: a pure
marginal shift gave copula 0.489 (chance), and a pure correlation change (rho 0.3 -> 0.6 at
identical marginals) gave marginals 0.487 (chance), copula 0.711, quadratic 0.713.

`calo_metrics.py --dump_features` writes the PAIRED per-event feature matrices — row k of real and
of generated is the same event, generated from the same truth particle list, with the same n_src.
The gate throws that pairing away; it is what makes the residual analysis below possible.

## Result — replicated on both seeds

| held-out AUC | seed 0 | seed 1 |
|---|---|---|
| **full (raw features)** | **0.8253 ± 0.0030** | **0.8329 ± 0.0018** |
| marginals only (joint destroyed) | 0.6007 | 0.6166 |
| **copula only (marginals matched)** | **0.7004** | **0.7242** |
| copula, quadratic logistic | 0.6620 | 0.6666 |
| copula, linear logistic [control] | 0.4845 | 0.4827 |
| best single feature | 0.5758 | 0.6262 |
| *aggregation prediction (independent)* | *0.6269* | *0.6269* |

Three things follow, and none of them were visible in the per-feature table:

1. **The marginal half is pure aggregation and nothing more.** Marginals-only lands at 0.60-0.62
   against a from-first-principles independent prediction of **0.627**. There is no hidden marginal
   defect; the 12 features are each slightly off and they add. Chasing any single marginal is
   chasing ~0.01.
2. **The joint half is the bigger half.** Dependence alone (0.70-0.72) beats marginals alone
   (0.60-0.62). This is the first quantitative statement that the calo model's *structure* — not
   its one-dimensional calibration — is the dominant defect.
3. **Most, but not all, of the dependence is second-order.** The correlation matrix alone buys
   0.662-0.667 of the 0.700-0.724; the remainder is higher-order. So a fix that repairs the
   correlations is the large majority of the available gain, and the Spearman table below names
   exactly which ones.

### Which correlations — `frac_near_floor` is the culprit, and the sign is systematic

Generated correlations are systematically **WEAKER** than real: mean Δρ over all 66 pairs
**−0.025 / −0.033**, negative in 39/66 and 41/66. The generator is producing showers whose
event-level quantities are too nearly independent of one another.

`frac_near_floor` appears in **7 of the top 12 |Δρ| entries on both seeds**, with a mean Δρ over
its 11 pairs of **−0.077 / −0.082** — roughly 3x the all-pair average:

| pair | real ρ | gen ρ | Δρ (s0) | Δρ (s1) |
|---|---|---|---|---|
| logE_mean × frac_near_floor | −0.112 | −0.351 | **−0.239** | **−0.260** |
| logE_p90 × frac_near_floor | +0.120 | −0.079 | **−0.199** | **−0.213** |
| frac_near_floor × cells_per_src | +0.218 | +0.079 | −0.140 | −0.127 |
| log_totE × frac_near_floor | +0.220 | +0.091 | −0.128 | −0.122 |
| n_cells × frac_near_floor | +0.218 | +0.106 | −0.112 | — |
| logE_std × cells_per_src | +0.428 | +0.301 | −0.128 | — |

Note the second row inverts the SIGN, not just the magnitude: in real events a harder 90th-percentile
cell goes WITH a higher floor fraction (+0.120); in generated events it goes against it (−0.079).

**This is the same feature the entire August energy-head thread was chasing, and it is not fixed —
only its marginal was.** Under v2 `frac_near_floor` collapsed to chance as a marginal (0.5045),
which the 2026-08-24 entry recorded as the defect being resolved. It was not: the floor fraction is
now marginally correct and *structurally* wrong. Its coupling to the rest of the shower's energy
structure is the single largest source of joint mismatch in the model.

### Paired per-event residuals (gen − real, in units of the REAL spread)

| feature | bias/σ (s0) | bias/σ (s1) | scatter/σ (s0) | r(Δ, log n_src) |
|---|---|---|---|---|
| logE_max | **+0.271** | **+0.312** | 0.884 | +0.157 |
| logE_std | +0.145 | **+0.321** | 0.942 | +0.042 |
| frac_near_floor | +0.101 | +0.220 | **1.632** | −0.073 |
| width_mean | +0.150 | +0.133 | 0.733 | −0.040 |
| cells_per_src | −0.101 | −0.064 | 0.478 | −0.023 |
| n_cells | −0.031 | −0.022 | **0.098** | **−0.318 / −0.256** |
| log_totE | −0.010 | +0.003 | **0.158** | +0.071 |

- `frac_near_floor`'s paired scatter is **1.63x the real event-to-event spread** — by far the
  largest. It is not merely decorrelated; per event it is noise.
- `n_cells` and `log_totE` are near-exact per event (scatter 0.098σ and 0.158σ), as they should be:
  both are pinned by the truth particle list and `E_true`.
- But `n_cells` carries a **conditional** bias: r(Δ, log n_src) = **−0.26 / −0.32**. Generated cell
  counts fall increasingly short in high-occupancy events, while the marginal AUC on `n_cells` is
  0.5052 — nothing. This is the textbook shape of a defect only a multivariate gate can see, and it
  matches the `n_cells × log_totE` pairwise excess of **+0.077** (both singles ~0.50).

### Where the classifier is most confident

The 355 events the gate is surest are generated, mean z-shift vs all generated events:
`logE_max +0.81`, `n_cells +0.68`, `frac_near_floor +0.63`, `logE_std +0.55`, `log_totE +0.42`.
All the same sign — the flagged events are jointly over-populated AND over-hard AND over-floored.
Consistent with the correlation reading: the generator is not reproducing the trade-off that keeps
real events on a narrower manifold.

## Verdict — KEPT as a method; the finding redirects the energy thread

The decomposition is now the standard follow-up to any calo gate number. The substantive finding is
that **`frac_near_floor` was never fixed, it was only made marginally invisible**, and that the
model's dominant defect is dependence, not calibration. Ten experiments since 2026-08-13 targeted
the floor's marginal; none of them could have moved its correlations, and the one hypothesis that
addressed coupling directly — "cells need to know about each other" — was falsified on a
*dispersion* argument (D_floor real 1.035) that says nothing about cross-feature correlation.

## Also measured / fixed here

- **All-species v2 census** (job 12930683, shard 0, no cap): 1,668,788 showers / 41.4M cells /
  24.83 cells/shower (median 11); depth p95 **1098 mm** vs 146 mm for e± alone. Per-class showers:
  γ **410,578** · e− 338,662 · e+ 321,059 · π− 186,480 · π+ 183,381 · p 66,021 · n 48,539 ·
  π0 22,019 · K0L 16,068 · K+ 14,950 · K− 14,080 · μ+ 12,479 · μ− 11,920 · p̄ 9,038 · n̄ 8,982 ·
  other 4,071 · K0S 461.
  **Re-attribution makes γ the LARGEST class**, where the v1 depositor census had it at 2.3% of
  energy — the e± fragments that used to be their own "showers" now book to their photon ancestor.
  No v1 per-species number is comparable to a v2 one.
- **`n_src` in the v2 slice counted deduped CELLS, not depositors.** The predicate
  `row_of[o][newcell] >= 0` is true for every row, so every build reported "merged sources/shower"
  exactly equal to cells/shower (19.82 and 19.82 on the e± production slice; 24.83 and 24.83 on the
  census). Nothing reads the field, so no result is affected, but the number was being used to
  describe how much merging re-attribution does. Now counts unique (ancestor, depositor) pairs.
- **`calo_metrics.py --logE_max_slice` read `points_flat[:, 2]`**, which under v2 is DEPTH. Same
  class of bug as the one caught in `train_calo_flow` on 2026-08-24, in the sibling script; the
  "energy is the LAST column" sweep missed it. Unused by the v2 runs (no checkpoint needed the
  override), so again no result is affected.
- **`--pdg_class` was silently ignored when `--real_slice` was given** (`sel = np.arange(len(cont))`).
  Harmless while every v2 slice was single-species; it would have turned every per-species number on
  a multispecies slice into an all-species one. Now filters and compacts the CSR arrays.
- **The event-gate loop was O(n_showers × n_events) and O(n_cells × n_events)** — `np.where(ev_of ==
  ev)` and `np.isin(gen_src, pm)` per event, ~1e11 element ops on the e± slice. Replaced with stable
  argsort blocks; required for a multispecies slice to evaluate at all.

## Next

- The indicated lever is whatever sets the floor fraction's COUPLING to shower energy structure,
  not its marginal. `--floor_n_buckets` conditioned the floor Bernoulli on a cell-count embedding
  and left the mechanism untouched (0.003); the missing conditioning is on the shower's *energy*
  scale, which is what `logE_mean`/`logE_p90` measure. The EnergyHead already takes `glob_std`;
  whether the floor branch sees it is the thing to check first.
- `n_cells` shortfall vs occupancy (r = −0.26/−0.32) is a separate, smaller thread. It is a
  GlobalHead `log_n` question, not an energy one.
- Run the same decomposition on the multispecies model when it trains — the question of whether
  pooling changes the joint structure or only the marginals is exactly what this tool answers.
