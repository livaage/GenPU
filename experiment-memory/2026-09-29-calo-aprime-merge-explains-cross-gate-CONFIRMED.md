# CONFIRMED: the snap-merge explains cross's gate. Unsnapped, cross is 0.787 vs self-only 0.894

**Date** 2026-09-29 · **Commit** `2604c3f` (+ uncommitted `jobs/calo_aprime_nosnap.sh`) · **Branch** `flow-response`
**Job** 14674540 (3-task array, gpu-test) · same ckpts and settings as
[2026-09-28 A-prime entry](2026-09-28-calo-aprime-attention-gives-coherence-but-collides-4x.md) with ONLY
`--snap_cells` removed · held-out shard 5 · all 17 classes · `--max_cells 4096`

## Hypothesis (pre-registered in the job header)
Cross collides 0.130 and merges 13.1% of its points under the snap; its gate losses (`logE_p90` 0.844,
`logE_mean` 0.838, `cells_per_src` 0.695) are the direction merging pushes. If the merge is the cause, then
unsnapped those three fall to roughly self-only's (<= ~0.72 / ~0.71 / ~0.55) and cross's composite falls
BELOW self-only's. If they stay >= 0.8 / 0.8 / 0.65, the energy marginal is wrong upstream.

## Result

| per-feature AUC | cross snap | **cross no-snap** | selfonly no-snap | baseline no-snap |
|---|---|---|---|---|
| `logE_p90` | 0.844 | **0.525** | 0.600 | 0.610 |
| `logE_mean` | 0.838 | **0.576** | 0.652 | 0.524 |
| `cells_per_src` | 0.695 | **0.507** | 0.515 | 0.512 |
| `logE_std` | 0.808 | 0.518 | 0.552 | 0.707 |
| `frac_near_floor` | 0.555 | 0.586 | **0.784** | 0.508 |
| `width_std` | 0.667 | **0.666** | 0.674 | 0.695 |

| composite (12 features incl. depth, no snap) | baseline | selfonly | **cross** |
|---|---|---|---|
| `event_gate_auc` | 0.9354 | 0.8942 | **0.7866** |
| decomposition full | 0.964 | 0.919 | 0.838 |
| marginals only | 0.788 | 0.828 | **0.641** |
| copula only | 0.777 | 0.625 | **0.597** |

Sanity: baseline unsnapped 0.9354 reproduces the pre-projection 0.9357 (job 13929818).
Self-only and baseline barely move with the snap (0.905 -> 0.894, 0.945 -> 0.935), as predicted from their
2.6% / 3.6% merge.

## Verdict
**Pre-registration: "merge IS the cause" — met on every criterion**, and by a wide margin (all three below
the self-only thresholds; composite 0.787 < 0.894). Cross's snapped gate penalty (0.787 -> 0.936) is the
collision defect, and nothing else in cross's energy model needs fixing to explain it. With cells unmerged,
cross is the best calo gate recorded on the all-species multispecies set (previous best: v2 seed 1,
gate8 0.889, STATUS 2026-08-25; a different feature set, so indicative only).

**How to read 0.787:** it is what cross would score IF collisions were eliminated without disturbing
anything else. It is an upper bound, not a deliverable: the unsnapped gate scores points that the detector
would read as one cell as two (the 2026-09-15 "0.9357 was flattering" argument applies). The deliverable
number stays the snapped 0.936 until collisions are fixed upstream.

## Next
1. **Build exclusion into cross** so its cells stop landing in the same channel (repulsion loss, or generate
   on the cell lattice). Target: snapped gate approaching the unsnapped 0.787. Merging stays correct.
2. What is left in cross, unsnapped: `width_std` ~0.67 is shared by ALL THREE arms (not an attention issue,
   a standing defect); `frac_near_floor` 0.586; copula 0.597.
3. Still open from 09-28: self-only's unexplained coherence 1.29; second model seed of self-only vs cross.
