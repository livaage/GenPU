# 2026-08-24 — first 3D calo model on re-attributed showers: frac_near_floor collapses to chance, seed spread 10x tighter

- **Commit / branch**: `7bc489c` / `flow-response`
- **Jobs**: 12899065 (train, 2 seeds x 60k), 12900324 (eval)
- **Ckpts**: `electron_v2_s0/checkpoint_060000`, `electron_v2_s1/checkpoint_060000`
- **Slices**: `electron_v2.npz` (1,986,107 showers / 39.4M cells), held out `electron_v2_h5.npz`

## NOT COMPARABLE TO ANY v1 NUMBER

v1 gated 2D fragment-showers (10.5 cells) against a v1 reference; this gates 3D re-attributed
showers (19.8 cells) against a v2 reference. Different attribution, dimensionality, population and
target. Reading 0.7456 -> 0.81 as a regression would be exactly the error the metric change exists
to prevent.

## (1) Depth IS modelled — the plan's acceptance metric, computable for the first time

| depth (mm from front face) | real | gen | ratio |
|---|---|---|---|
| -20-0 | 0.0497 | 0.0550 | 1.107 |
| **0-25** | **0.3051** | **0.3051** | **1.000** |
| 25-50 | 0.2718 | 0.2444 | 0.899 |
| 50-100 | 0.2321 | 0.2385 | 1.028 |
| 100-200 | 0.0691 | 0.0797 | 1.153 |
| 200-400 | 0.0052 | 0.0093 | 1.788 |
| 400-800 | 0.0090 | 0.0096 | 1.067 |
| 800-inf | 0.0580 | 0.0585 | 1.009 |

Energy-weighted mean depth **real 131.0 mm vs gen 137.7 / 138.7 mm** (~5%). Bulk bins within 3%;
the worst ratio is in a bin holding 0.5% of energy. W/sigma: `cell_depth` 0.0244 / 0.0159,
`shower_depth` 0.0359 / 0.0515.

## (2) `frac_near_floor` COLLAPSED TO CHANCE — 0.5045

Per-feature gate AUC (seed 0): logE_max **0.5758**, logE_std 0.5606, width_mean 0.5568,
logE_p90 0.5461, cells_per_src 0.5281, logE_mean 0.5253, depth_mean 0.5149, depth_std 0.5072,
n_cells 0.5052, **frac_near_floor 0.5045**, width_std 0.5036, log_totE 0.5018.

**This is the defect the entire August energy-head thread was chasing.** Top or near-top
discriminator in every run since 2026-08-13 at 0.59-0.60, surviving ~10 experiments: floor Bernoulli
variants, `partition`, `--energy_glob_idx 1`, separate trunks, `--floor_n_buckets`, the i.i.d. test.
Under v2 it carries essentially no signal.

Candidate causes, NOT separated (the three changes landed together, deliberately):
- **dedup** — merged cells sum contributions, removing the ~20% near-floor inflation measured on
  2026-08-24 (0.0334 contributions vs 0.0278 cells);
- **re-attribution** — showers are real showers rather than fragments, so multiplicity and energy
  structure are the physical ones;
- **depth** — the model gained a dimension driving 0.40-0.54 of intrinsic width.

The leading single discriminator is now `logE_max` (0.5758), i.e. the upper energy tail.

## (3) Seed spread is ~10x TIGHTER than v1

| | seed 0 | seed 1 | spread |
|---|---|---|---|
| gate8 | 0.8105 | 0.8074 | **0.003** |
| gate + width | 0.8109 | 0.8019 | 0.009 |
| **gate + depth** | 0.8219 | 0.8374 | 0.016 |

Against the v1 e± gate8 spread of **0.11** (job 12526752). Plausibly because the v2 population is
homogeneous (real showers) where v1 mixed fragments with real showers. Practical consequence: v2
comparisons need far fewer seeds to be interpretable.

Composite gate 0.81 while no single feature exceeds 0.58 — the classifier is using multivariate
structure, so per-feature AUCs alone will not say what to fix.

## Bugs caught on the way (both would have been silent)

1. `train_calo_flow` computed per-class `logE_max` from `pts[:, 2]`, which under v2 is **DEPTH**:
   "per-class logE_max: 2220.00" is exactly the max depth in mm. The energy head would have been
   bounded by a millimetre. Energy is now read as the LAST column everywhere.
2. `calo_metrics` recomputed the anchor from **stage2 particle rows** — under v2 that is the
   DEPOSITOR, not the calo-incident ancestor the shower belongs to, silently reintroducing the
   error re-attribution exists to fix. It now reads the anchor the slice stores.

## Next

- The three changes are confounded by construction. `--no_reattribute` exists if attribution ever
  needs isolating.
- Depth is not yet used for **cell identity** — the gate remains contribution-level, and
  cross-shower superposition (3.5% at PU0, far more at M3's mu) still double-counts.
- Pion and photon v2 slices not built. Photon is the interesting control: 0.1% split, so
  re-attribution should barely move it.
