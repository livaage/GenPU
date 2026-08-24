# 2026-08-16 — floor fraction: the i.i.d. hypothesis is FALSIFIED; the defect is p(floor | n) at small n

- **Commit / branch**: `99879a3` / `flow-response` (probe untracked at run time)
- **Job**: 12470841 (`jobs/calo_floor_dispersion.sh`, diagnostic only, no training)
- **Script**: `scripts/calo_floor_dispersion.py`
- **Checkpoints scored**: `pion_anchor/checkpoint_040000`, `electron_anchor/checkpoint_060000`
  (Phase 1 helix, `--no_partition`, matching how they were scored in `jobs/calo_anchor_train.sh`)
- **Artifacts**: `plots/calo/metrics/floor_dispersion_{pion,electron,positron}.{json,png}`

## Hypothesis

`frac_near_floor` has been the top or near-top gate discriminator in every run since 2026-08-13 and
no per-head patch has moved it. Proposed mechanism: both calo heads are conditionally **i.i.d.
across cells** given (particle embedding, sampled globals) — `PointCFM.sample` integrates each cell
independently, `EnergyHead.sample` flips one Bernoulli per cell — so the only channel for
cell-to-cell correlation is the shared 4-dim global draw. If real showers are correlated
(compact-and-bright vs diffuse-and-fringy **as a whole**), the at-floor count should be
over-dispersed relative to Binomial(n, p), and an i.i.d. sampler structurally cannot reproduce it.
That would predict `frac_near_floor` is unfixable by conditioning and needs a richer sampled latent
or a set model — and would explain why `--energy_glob_idx 1` moved it only 0.618 → 0.574.

## Change

No training. Measured the **dispersion ratio** `D = Var_obs(k | n) / (n·p·(1−p))` in bins of
cells-per-shower, real vs generated, where `k` is the at-floor cell count. The denominator is the
*exact* variance conditionally-independent cells can produce, so `D = 1` is a hard architectural
floor, not a fitted baseline. `shower_width` carried alongside as a **positive control** with its
own delta-method i.i.d. null (`Var(width|n) = var(q)/(4·n·mean(q))`, `q = dη²+dφ²`), since width is
already believed i.i.d.-limited (`calo_metrics.py:39-40`, `calo_flow.py:490-492`).

The test is conservative by construction: within a real shower `p` varies cell to cell (bright core
vs faint fringe), and a Poisson-binomial with heterogeneous `p` is *under*-dispersed relative to
Binomial(n, p̄) by Jensen, so heterogeneity pushes `D_real` **down**. An observed `D_real > 1` is a
lower bound.

## Result — hypothesis FALSIFIED, twice over

**(1) Real showers are barely over-dispersed at all.**

| species | D_floor real | D_floor gen |
|---|---|---|
| pion | 1.199 | 1.031 |
| e− | **1.035** | 1.024 |
| e+ | **1.036** | 1.025 |

For e± the real at-floor count is statistically indistinguishable from Binomial — an i.i.d. sampler
is **adequate** for that variance. Only the pion shows a gap, modest and rising cleanly with n
(1.00 at n=2 → 1.64 at n=24) against a flat generated ~1.03.

**(2) The positive control killed the framing outright.** Both real *and* generated widths are
over-dispersed **3–16×** above the i.i.d. null (pion 6.43 real / 6.79 gen; e− 3.20 / 3.90), so the
shared-global channel already produces abundant shower-level correlation. And generated width is
consistently **more** dispersed than real, growing with n — too broad at fixed n, the opposite
direction from "i.i.d. loses variation". That excess is consistent with the known e± core
over-dispersion from the brem-wrong anchor (physical frame 1.29 → 1.93 at high pT) propagating into
width = sqrt(core² + intrinsic²).

**(3) The real defect is the MEAN conditioned on n, and it is sharply localized.**

| | real ⟨floor frac⟩ | gen | |
|---|---|---|---|
| pion | 0.0239 | 0.0180 | 25% too **low** |
| e− | 0.0409 | 0.0471 | 15% too **high** |
| e+ | 0.0417 | 0.0475 | 14% too **high** |

- **e±: the entire error sits at n = 1.** Real `p` = 0.011 (e−) / 0.009 (e+), generated **0.061 /
  0.062** — a **6× excess** of at-floor cells in single-cell showers, over 25.7k / 23.3k showers
  (~8% of the sample each). By n ≥ 9 real and gen agree to ~0.001.
- **Pion: the model has essentially no n-dependence.** Real `p` climbs 0.018 → 0.032 with n;
  generated is flat at ~0.016–0.019 across the whole range.

Cells-per-shower itself is well modelled (pion 15.57 real / 15.72 gen; e− 10.50 / 10.52), so `log_n`
is not the problem — what the head does with it is.

## Verdict — ABANDON the set-model / cell-coupling direction; the target is now precise

Both structural options (richer sampled latent for *correlation*, actual set model) are ruled out as
the lever for `frac_near_floor`: there is almost no over-dispersion in real data for it to buy.

The Aug-14 multiplicity diagnosis was **right in mechanism but wrong in target**. It is not "tell the
energy head n" (that was `--energy_glob_idx 1`, worth 0.618 → 0.574); it is that **p(floor) must be
an explicit, strongly non-linear function of n**, and a single `log_n` scalar into an MLP dominated
by large-n showers under-fits the n ≤ 2 corner where the whole e± error lives. Physically obvious in
hindsight: a shower with exactly one cell has that cell carrying the **entire** shower energy, so it
cannot be a faint fringe cell — real `p(floor | n=1)` ≈ 0.01, and the model emits 0.06.

## Next

1. **Free first probe — `partition` A/B.** These runs used `--no_partition` to match the logged
   metrics. With `partition` on, cells are renormalised to the sampled shower total, which for n = 1
   *forces* the single cell to carry `total_logE` and would fix the e± n=1 defect by construction.
   Partition was ruled out on 2026-08-13 for degrading the pooled `cell_logE` marginal — but that was
   measured pooled, never at small n. One flag on an existing checkpoint.
2. **Make the floor Bernoulli see n explicitly and non-linearly** — e.g. a small-n embedding /
   one-hot for n ∈ {1, 2, 3, 4+} rather than a `log_n` scalar, or the plan's floor-fraction-as-a-
   GlobalHead-dim with the Bernoulli driven from the sampled value.
3. Re-read the pion separately: its defect is a *missing slope* in n, not a small-n corner, so it may
   need the same fix for a different reason.
