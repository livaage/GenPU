# The 128-cell generation cap was a real defect; fixing it moved BOTH gate halves and neither the composite

**Date** 2026-09-15 · **Commit** `32d0cab` (+ uncommitted `n_max_pdg` / `--max_cells`) · **Branch** `flow-response`
**Jobs** 13920087 (cap A/B + first grid pass, 9m02s), 13921623 (gate decomposition, part of 11m52s)
ckpt `multispecies_v2_s0/checkpoint_060000` · slice `multispecies_v2_h5.npz` · **1 seed, by design**

## Hypothesis

`CaloFlow.sample_showers` clamped the sampled cell count at a bare `max_cells=128` literal that no
measurement ever justified. `STATUS.md:256` even states it backwards — "showers cap at 128 cells so
n^2 is free" — written as a property of SHOWERS. It is a property of our sampler.

Measured first, on `multispecies_v2_h5` (1,676,985 showers / 41,653,183 cells): 128 truncates
**2.10%** of showers carrying **17.7%** of cells and destroys **6.87% of ALL cells**, and it is
hadron-selective:

| class | pbar | nbar | K0L | K± | pi± | p | gamma | e± | mu±, pi0 |
|---|---|---|---|---|---|---|---|---|---|
| % showers >128 | 25.3 | 19.1 | 7.8 | 9-11 | 3.8 | 1.2 | 1.1 | 1.0 | **0.00** |
| % cells lost | 21.3 | 19.4 | 17.2 | 14-17 | 8.3 | 5.2 | 4.9 | 2.9 | **0.00** |

Every prior conclusion about the cap came from the DEDICATED e± model, where the loss is 2.9% — the
one place it looks harmless. Antibaryons annihilate and make the biggest showers, so the cap lands
hardest exactly on the hadrons that carry 51.6% of cells and all of the coherence signal.

Because `partition=True` renormalises survivors to the SAMPLED total, truncation does not drop that
energy — it PACKS it into fewer cells. So the pre-registered prediction was that this contaminates
every per-cell energy feature, not just `n_cells`.

## Change

`n_max_pdg` buffer on `CaloFlow` (same contract as `logE_max_pdg`: the largest shower each class
actually produced), `sample_showers` bounds by `min(max_cells, n_max_pdg[pdg])` and returns
`n_trunc`, `train_calo_flow.py` computes it from the slice, `calo_metrics.py --max_cells`. The bound
stays FINITE on purpose: `g[:,1]` is a draw from an unbounded Gaussian mixture in log-n, so removing
the clamp lets an exponential tail ask for millions of cells. Backward compatible — older
checkpoints lack the buffer, it defaults to `inf`, and `max_cells` alone governs as before.

CPU pre-check (3000 showers, seed-matched, no retrain): cap 128 -> n_trunc 2.37%, mean n 24.54,
73,629 cells; cap 4096 -> 0.00%, mean n 26.66, 79,966 cells; REAL 2.43%, 26.09, 78,280. **The
`log_n` mixture's TAIL is sound** — the cap, not the model, was the reason mean cells/shower ran 6%
low.

## Result

Control (`--max_cells 128`) reproduces the logged seed-0 number: gate8 **0.9328** vs **0.9354**,
`n_trunc` 2.123% vs the real 2.10% >128 rate. No drift, so the treatment arm is readable.

| per-feature AUC | cap 128 | cap 4096 | delta |
|---|---|---|---|
| `logE_mean` | 0.6292 | **0.5238** | -0.105 |
| `logE_p90` | 0.7098 | **0.6098** | -0.100 |
| `cells_per_src` | 0.5695 | **0.5123** | -0.057 |
| `frac_near_floor` | 0.5605 | **0.5081** | -0.052 |
| `logE_std` | 0.7552 | 0.7074 | -0.048 |
| `n_cells` | 0.5107 | 0.5046 | -0.006 |

W/sigma: `cells_per_shower` **0.0489 -> 0.0104** (4.7x), `cell_logE` 0.0708 -> 0.0597,
`d_eta` 0.0424 -> 0.0318, `d_phi` 0.0506 -> 0.0343.

**Composite gate did not move**: gate8 0.9328 -> 0.9356, width 0.9296 -> 0.9276, depth 0.9650 ->
0.9586.

### The pre-registered decomposition test — FALSIFIED on the copula side

Recorded before running: *marginals_only drops sharply, copula_only stays FLAT.*

| | cap 128 | cap 4096 | delta |
|---|---|---|---|
| full (raw) | 0.9675 +- 0.0026 | 0.9557 +- 0.0016 | -0.012 |
| marginals only | 0.8421 | **0.7432** | **-0.099** (predicted) |
| **copula only** | 0.8453 | **0.7760** | **-0.069** (predicted FLAT) |
| copula, quadratic | 0.7500 | 0.7251 | -0.025 |
| linear control | 0.4839 | 0.4814 | ok, ~0.5 by construction |
| aggregation prediction | 0.8440 | 0.7491 | -0.095 |

**The cap WAS entangled with the joint.** Obvious in hindsight and worth stating so it is not
re-derived: truncation acts SELECTIVELY ON LARGE SHOWERS, so it injects correlation between
`n_cells` and every energy feature. A defect that acts on a subpopulation cannot be purely marginal.

Two things survive:
- **The marginal half is still pure aggregation** — the independent prediction tracks
  `marginals_only` in BOTH arms (0.8440 vs 0.8421; 0.7491 vs 0.7432). The 2026-08-25 finding holds.
- **The fix restored the expected ordering.** At cap 128 marginals (0.842) ~= copula (0.845); at
  4096 copula (0.776) > marginals (0.743), i.e. the "joint is the bigger half" structure the pooled
  model had lost.

## Verdict — KEPT, and the composite-gate flatness is the lesson

Generation is now physically correct (no 128-cell spike, no energy repacked into truncated showers),
`cells_per_shower` is 4.7x better, both decomposition halves improved, and it costs ~7% more cells.

**The composite gate is flat because at 0.96 it has redundant paths**: signal can be removed from
both halves and the classifier still separates. Anyone reading only `event_gate_auc` would conclude
this change did nothing. It is the clearest case on record that the composite gate is a poor
instrument near saturation, and the decomposition is the one that reports honestly.

**CORRECTION to an in-session claim**: the cap was said to be "plausibly contaminating the pooled
multispecies numbers". It contaminated the MARGINALS and the copula; it did not move the composite,
so the pooled conclusions as recorded stand.

## Next

- **One seed is correct here and should not be "fixed" later**: this is a PAIRED A/B on the SAME
  checkpoint with one sampling flag changed, so the 0.046 pooled training-seed spread cancels
  identically. Two seeds would be needed only to compare different checkpoints.
- **Open discrepancy, not resolved**: `frac_near_floor` reads 0.5605 here on the pooled model where
  STATUS records **0.7335/0.7448** for pooled v2. The composite control matched, so this is not
  drift — most likely a different estimator (greedy-selection vs single-feature AUC), but that was
  NOT verified and should not be assumed.
- Nothing is retrained yet, so no checkpoint carries `n_max_pdg`; every existing ckpt still runs
  with the `max_cells` argument alone.
