# 2026-08-25 — multispecies v2: pooling costs +0.09 on e±, and the decomposition says it is ALL marginal

- **Commit / branch**: `32d0cab` / `flow-response`
- **Jobs**: 12953664 (build), 12953673 (train, 2 seeds x 60k), 12953682 (eval core), 12953717 (eval species)
- **Ckpts**: `multispecies_v2_s0/checkpoint_060000`, `multispecies_v2_s1/checkpoint_060000`
- **Slices**: `multispecies_v2.npz` (5,025,302 showers / 125.0M cells), `multispecies_v2_h5.npz`
- **Baseline compared against**: `electron_v2_s0/s1` (2026-08-24)

## Hypothesis

One PDG-conditioned head over all 17 classes was the decision of 2026-08-13, taken on v1 fragment
showers. Re-run on v2 re-attributed 3D showers, 60k steps and 2 seeds matched EXACTLY to the
`electron_v2` recipe, so evaluating the pooled model on classes 0,1 against the same held-out
population isolates the cost of pooling and nothing else. Expected: a modest cost on e±, bought
back by covering the other 92% of calorimeter energy.

## Result — the A/B is clean and the cost is much larger than expected

`ms_ele` and `electron_v2` score **660,330 showers over 7,186 events** — identical populations, so
the difference is the model and only the model.

| held-out gate | dedicated e± | **pooled (ms_ele)** | Δ |
|---|---|---|---|
| gate8 seed 0 | 0.8105 | **0.9079** | +0.097 |
| gate8 seed 1 | 0.8073 | **0.8852** | +0.078 |
| +width seed 0 / 1 | 0.8111 / 0.8018 | 0.9034 / 0.8805 | |
| +depth seed 0 / 1 | 0.8218 / 0.8371 | 0.9122 / 0.8904 | |

**The whole regression is in the MARGINALS. The joint defect is untouched.** Running the
2026-08-25 decomposition on both:

| | dedicated s0 / s1 | pooled s0 / s1 |
|---|---|---|
| full | 0.8253 / 0.8329 | 0.9107 / 0.8918 |
| **marginals only** | **0.6007 / 0.6166** | **0.7266 / 0.7232** |
| **copula only** | **0.7004 / 0.7242** | **0.7053 / 0.6959** |
| copula quadratic | 0.6620 / 0.6666 | 0.6474 / 0.6446 |
| copula linear [control] | 0.4845 / 0.4827 | 0.4831 / 0.4843 |
| aggregation prediction | 0.6269 / 0.6675 | 0.7693 / 0.7651 |

The copula column is flat (0.700 → 0.705, 0.724 → 0.696). The marginal column moves 0.601 → 0.727,
and the pure-aggregation prediction moves with it (0.627 → 0.769). So pooling degraded per-feature
calibration and left the dependence structure exactly where it was — the two defects are separable
and they respond to different things.

### The responsible feature is `frac_near_floor`, and pooling put it back WORSE than v1

| `frac_near_floor` single-feature AUC | s0 | s1 |
|---|---|---|
| v1 e± (`electron_anchor`) | 0.702 | — |
| **v2 dedicated e±** | **0.5045** | **0.5461** |
| **v2 pooled, e± population** | **0.7335** | **0.7448** |

In the pooled model it is the FIRST greedy pick and worth **0.758 on its own**; in the dedicated
model greedy starts with `logE_mean` and `frac_near_floor` never enters the top of the list. The
shared head does not merely fail to fix the floor on e± — it re-breaks it past its v1 level.

This is consistent with the coupling finding logged the same day. The floor mechanism depends on the
shower's energy scale, and the classes now sharing that head have wildly different ones: mean depth
alone runs γ 113 mm → e± 301 → p 335 → π± 470-483, and the v2 all-species depth p95 is 1098 mm
against 146 mm for e± alone. A single head asked to serve soft floor-dominated EM showers and deep
hard hadronic ones mis-serves the EM end.

### The 2026-08-24 prediction is CONFIRMED for pion and photon

Recorded prediction: "pion v2 should fall from 0.666 toward chance; photon, never broken (0.510),
should barely move."

| `frac_near_floor` | v1 | v2 pooled s0 / s1 |
|---|---|---|
| π± | 0.666 | **0.5257 / 0.5567** |
| γ | 0.510 | **0.5117 / 0.5742** |

Both hold. The fragment-fraction explanation for the floor artifact is correct for π and γ; e± is
the species where it does not survive pooling.

### Full-calorimeter number, measurable for the first time

All 17 classes pooled, 1,676,985 showers / 7,384 events: **gate8 0.9354 / 0.8891**, with depth
**0.9673 / 0.9539**. Near-separable. Its decomposition is marginal-dominated (marg 0.845 / 0.790,
copula 0.852 / 0.788, aggregation prediction 0.843 / 0.810) — broad per-feature miscalibration
across a heterogeneous mixture, with `logE_std` → `depth_mean` → `frac_near_floor` → `depth_std` as
the greedy order. `depth_mean` entering second is new: it is species-dependent, so a shared model
that gets the depth-vs-species relation wrong is exposed here and nowhere else.

### Per-species (pooled model, seed 0 / seed 1)

| | showers | showers/event | gate8 | +depth |
|---|---|---|---|---|
| e± | 660,330 | 91.9 | 0.9079 / 0.8852 | 0.9122 / 0.8904 |
| γ | 412,802 | 57.3 | 0.8842 / 0.8392 | 0.8962 / 0.8491 |
| π± | 373,656 | 51.2 | 0.9123 / 0.8664 | 0.9555 / 0.9435 |
| p | 66,604 | 9.9 | 0.7476 / 0.7009 | 0.8167 / 0.7876 |
| n | 48,590 | 7.5 | 0.7489 / 0.7062 | 0.7803 / 0.7298 |

**METHODOLOGICAL CAVEAT — these are not comparable across species.** The gate features are pooled
over an event's showers OF THE SELECTED CLASS, so a species with 7.5 showers per event yields far
noisier features than one with 91.9, and a weaker gate for that reason alone. p and n look best here
and are almost certainly not. Any cross-species claim needs multiplicity-matched events or a
per-shower metric; the within-species seed-to-seed and model-to-model comparisons above are sound.

### Seed spread blew up ~10x

| | dedicated e± | pooled e± | pooled all |
|---|---|---|---|
| gate8 spread | **0.003** | 0.023 | **0.046** |

The 2026-08-24 note that "v2 comparisons need far fewer seeds to be interpretable" holds for a
single-species model only. On the mixture, 2 seeds is the minimum, and a 0.02 effect is not readable
at all.

## Verdict — the slice and the tooling are KEPT; the single shared head is NOT settled

The multispecies v2 slice and the per-species eval path are correct and stay. The 2026-08-13
decision "one PDG-conditioned head, not per-species checkpoints" was taken on v1 fragment showers
and without a way to see which half of a gate was moving; it now has a measured cost of **+0.08-0.10
on e±**, localized to one feature, on identical populations. That is not by itself a reason to
abandon it — a full-event generator needs all 17 classes — but it does mean the shared head as
currently built is losing something a dedicated head had.

## Also fixed / measured

- `n_src` now reports what it claims: **2.67 distinct depositors merged per shower, 60.4% of showers
  unmerged** (max 558). Previous builds printed 24.87, which was the cell count. Re-attribution
  merges materially less than that number implied.
- Production slice: 5,025,302 showers / 124,999,386 cells / 24.87 per shower — within 1% of the
  estimate the census was used to make, and it trained in 15 min for 2 seeds.

## Next

- **The floor's energy conditioning is now the top lever, and it has a sharper target than
  yesterday**: whatever the dedicated e± head had that the shared one lost. Both have the same
  `EnergyHead`; what differs is that the shared trunk's gradients now come mostly from hadrons
  (γ+e±+π± are 75% of showers but the deep classes set the energy scale). `--separate_trunks`
  was tested on v1 and made things worse, but it has never been tested where the mixture is this
  heterogeneous — that is a different experiment from the one that failed.
- A **per-species logE floor/scale in the energy head** is the narrower version: `logE_max_pdg`
  already exists as a per-class buffer, so the machinery for per-class energy bounds is present and
  only the floor branch is class-blind.
- Re-run the decomposition after any fix. Marginal and joint move independently — this run is the
  proof — so a fix that improves one can silently be paid for out of the other.
