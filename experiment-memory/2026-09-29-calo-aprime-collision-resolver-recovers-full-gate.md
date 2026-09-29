# Collision resolver: cross's merged gate 0.936 -> 0.787, identical to unmerged. The ceiling is reachable

**Date** 2026-09-29 · **Commit** `2604c3f` + uncommitted `calo_cells.resolve_collisions`,
`calo_metrics.py --resolve_collisions`, `jobs/calo_aprime_resolve.sh` · **Branch** `flow-response`
**Job** 14675671 (3-task array, gpu-test) · ckpts at 60k, seed 0 · all 17 classes · `--max_cells 4096 --snap_cells`
Follows [merge-explains-cross entry](2026-09-29-calo-aprime-merge-explains-cross-gate-CONFIRMED.md).

## Hypothesis (pre-registered in the job header)
If moving colliding cells to free neighbours recovers the unmerged gate, the 0.787 ceiling is reachable and
exclusion is worth building properly. Reachable: cross's `logE_p90` / `logE_mean` / `cells_per_src` <= ~0.60,
gate within ~0.05 of its 10-feature unmerged number and below self-only. Not reachable: cross stays >= ~0.90.
Watch: widening (`width_mean`, `width_std`).

## Change
`resolve_collisions` (`src/genpu/calo_cells.py`): in each (cell, shower) the highest-energy point keeps the
cell; each other point takes the nearest FREE in-plane neighbour (same layer, <= 2 pitches, ordered by
distance from its own continuous position); none free -> merged as before. Energy conserved exactly.
Synthetic check before GPU time: all collisions resolved, layer kept 100%, median move 1 pitch, unique
(cell, shower). Re-snap consistency 0.9956 vs 0.9964 for the plain snap (pre-existing face-edge effect).

## Result

| | baseline | selfonly | **cross** |
|---|---|---|---|
| collided points | 3.61% | 2.65% | **13.05%** |
| moved / fell back to merge | 1.44M / 80.7k | 1.10M / 243 | 5.43M / 1.8k |
| mean move (pitches, centre to centre) | 1.06 | 1.03 | 1.04 |
| merged after resolve | 0.19% | 0.00% | 0.00% |
| `event_gate_auc`, merged, **plain** (09-28) | 0.9451 | 0.9045 | 0.9361 |
| `event_gate_auc`, merged, **resolved** | 0.9346 | 0.8942 | **0.7867** |
| `event_gate_auc`, unmerged (12 feat, 09-29) | 0.9354 | 0.8942 | 0.7866 |
| decomposition full, resolved (10 feat) | 0.939 | 0.911 | **0.821** |
| decomposition full, unmerged, depth dropped (10 feat) | 0.939 | 0.912 | **0.821** |

Cross resolved per-feature: `logE_p90` **0.525**, `logE_mean` 0.576, `cells_per_src` 0.507, `width_mean`
0.509, `width_std` 0.666 -- identical to unmerged to the third decimal. No widening on any arm.

## Pre-registration scorecard
**"Reachable" -- met on every criterion.** Features <= 0.60 (all three); gate within 0.0001 of unmerged
(allowed 0.05); below self-only (0.787 vs 0.894). Widening: none.

## Reading -- and what the gate CANNOT see
The whole merged-gate penalty, on every arm, is the count and energy-per-cell effect of merging. That also
retroactively explains the 2026-09-15 projection cost on baseline (0.9357 -> 0.9458): collisions, all of it.

**Caveat:** resolved and unmerged agree to 4 decimals because the gate's 10 features are per-event energy
and width aggregates, and a 1-pitch move changes none of them. So this shows that ANY exclusion which keeps
cell count and per-cell energy recovers the gate. It does NOT show the resolver's placements are realistic:
it pushes 13% of cross's cells one pitch outward, which the gate is blind to and which could damage exactly
what attention bought (NN spacing, coherence). Not measured yet.

Baseline is the one arm where exclusion is not free: 80.7k points (5.3% of its collisions) found no free
neighbour within 2 pitches -- its cores are genuinely over-full, not just jittered.

## Verdict
**The 0.787 is reachable.** Cross + exclusion is the best calo generator on record (merged, deliverable-
style; previous 0.889 on a different feature set). Resolver KEPT as an opt-in diagnostic flag.
Whether the resolver itself is acceptable as the generation-time cell projection is a DECISION, not settled
here: it is deterministic post-processing like the snap, but it edits placement, which the 2026-09-15 rule
("do not fix the merge") was written to prevent.

## Next
1. Check the resolver does not undo the attention gains: co-occupancy NN spacing and coherence on RESOLVED
   cells (cross). If they hold, the resolver is a defensible projection step; if not, exclusion must be learned.
2. Learned exclusion (cell tokens without replacement) stays the principled route; HCAL-endcap grid still gates it.
3. Carried: self-only's 1.29 coherence; second model seed of self-only vs cross.
