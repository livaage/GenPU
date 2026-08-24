# 2026-08-24 — Q3 on the CLEAN curler population: the falsification was wrong, the conclusion survives on better evidence

- **Commit / branch**: `99879a3` / `flow-response` (working tree)
- **Job**: 12878141 (`--born_outside_only`, shard 5). Follows
  [the contamination finding](2026-08-24-turning-branch-contamination-phase4-REOPENED.md) and the
  r-ordering fix re-run (job 12871830).
- First measurement of Q3 on the population that actually has tracks. 2026-08-16 and the corrected
  re-run both pooled calo-internal shower fragments into the curler branch.

## Result

**(1) COVERAGE — the Phase 4 premise is inverted, and for e± it is BACKWARDS.**

| branch | pion no-hits | pion med n_hits | e± no-hits | e± med n_hits |
|---|---|---|---|---|
| barrel | 0.185 | 13 | **0.957** | 0 |
| endcap | 0.237 | 9 | **0.854** | 0 |
| **turning (TRUE curlers)** | **0.044** | **11** | **0.231** | **5** |

Against the logged "81% (pion) / 93% (e±) of curlers have zero tracker hits". STATUS's reading —
"tracks exist exactly where the helix already works and are absent where it is weak,
anti-complementary" — is false: for e± the tracker is EMPTY on the face branches (86-96%) and
POPULATED on the curler branch (77% tracked). That is complementary, the opposite of the claim.

**(2) Q2 — the pi/2 phi result is REAL, now confirmed on clean data.** |d_phi| median 1.5754 (pion)
/ 1.5750 (e±), i.e. the vacuum turning point's phi is uncorrelated with the real outermost hit's.
Radial agreement is good though: delta r median -11.5 mm (pion) / -12.2 mm (e±).

**(3) Q3 — the endpoint's gain is SMALL.**

| | sig_phi helix -> track | gain | sig_eta helix -> track | gain |
|---|---|---|---|---|
| pion turning | 1.0560 -> 1.1089 | 0.95 | 0.9946 -> 0.8582 | **1.16** |
| e± turning | 0.6875 -> 0.6868 | 1.00 | 0.7164 -> 0.5992 | **1.20** |

~1.16-1.20x tighter eta, nothing in phi. Compare the helix's own 15-24x on the face branches.

## Verdict — Phase 4 stays UNBUILT, on evidence that now means what it says

The reason it was dismissed (no tracker coverage) was an artifact of pooling. The decision survives
anyway, for a different reason: the ceiling is ~1.2x on eta only, over 18.7% of the pion turning
branch and ~8% of e±.

**And that 1.2x is an UPPER BOUND** — it uses the REAL tracker endpoint from truth hits. A generated
endpoint carries the tracker head's own error, and that head gates at 0.61 with documented
free-running drift, so the realized gain is strictly less than 1.2x.

## Next / carry-forward

- **Do not build Phase 4 now.** Revisit only if the tracker head improves materially, since the
  e± complementarity (tracks present exactly where the helix fails) is a real structural fact and
  would become useful if the endpoint were accurate.
- **The e± face-branch coverage collapse is itself a finding**: e± reaching the calo face leave
  almost no tracker hits (86-96% empty). Consistent with the logged brem picture — the deposit comes
  from photons radiated early that travel straight, so there is no charged track to see.
- **Phase 0b's per-branch table needs re-reading** — its "turning pt" column (pion 2.50x, proton
  2.45x, e± 1.98x) pools the same two populations, so the "steady ~2.5x on every charged species"
  may be two different effects averaged.
