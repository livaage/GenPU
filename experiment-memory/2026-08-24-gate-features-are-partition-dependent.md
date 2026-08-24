# 2026-08-24 — 9 of the calo gate's 10 features are PARTITION-dependent; the floor bias is real but only 20%

- **Commit / branch**: `723380b` / `flow-response`
- **Job**: 12879998 (`calo_floor_contrib_vs_cell.py`, 600 events, 3.85M contributions / 3.20M cells)

## The structural point (no measurement needed)

`preprocessing.py` books, per (cell, particle) pair, that particle's **contribution**, not the
cell's total. `calo_metrics.py:33` then computes every energy feature over those contributions. So
the stored objects are contribution PAIRS, not cells. Checking each gate feature for invariance to
how particles are partitioned:

| feature | partition-invariant? | why |
|---|---|---|
| `log_totE` | **yes** | sum of contributions = true event energy |
| `n_cells` | no | counts contribution pairs, not distinct cells |
| `logE_mean/std/max/p90` | no | statistics over contributions |
| `frac_near_floor` | no | fraction of contributions near floor |
| `cells_per_src` | no | explicitly divides by the source count |
| `width_mean/std` | no | "shower" IS the partition unit |

**One of ten.** Re-attribution halves the source count and merges contributions, so it moves nine
features by construction, independently of model quality. **That is why there is no valid comparison
to the 0.7456 baseline across a slice-definition change**, and why the fix is a cell-level metric
(scatter_add contributions into physical cells, then compute features on cell energies) — observable,
partition-invariant, and the same superposition step M3 needs.

## The measurement — and a claim of mine that did NOT hold up

I predicted that if contributions and cells differed materially, much of the energy-head thread
since 2026-08-13 had been chasing a partitioning artifact. Measured:

| | value |
|---|---|
| `frac_near_floor` over CONTRIBUTIONS | 0.0334 |
| `frac_near_floor` over CELLS | 0.0278 |
| ratio | **1.203x** |
| absolute difference | **+0.0056** |

log-energy: contributions mean -8.036 / std 0.893, cells mean -7.923 / std 0.927.

**The bias is real and exactly matches the 1.206 mean contributors per cell — but it is 20%
relative, 0.56 percentage points absolute.** That is NOT enough to explain the energy-head defect;
the real-vs-generated discrepancy is the larger effect. **My stronger framing was wrong and should
not be carried forward.**

## Verdict

Move to a cell-level metric for **correctness and comparability**, not as a fix for the energy head.
The reason to do it is that nine of ten features are otherwise meaningless across the re-attribution
change — not that it hides a big physics error.

## Next

- Build the cell-level event metric (scatter_add into cells) as the gate for anything post-re-attribution.
- It also forces a mismatch the current gate avoids: generated showers are continuous (eta, phi)
  points while real ones sit at cell centres. Binning both to the cell grid is required, and the
  present energy features dodge this only by never using position.
