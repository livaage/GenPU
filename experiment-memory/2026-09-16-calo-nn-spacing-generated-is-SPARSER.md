# Discriminator result: generated cells are SPARSER than real (NN ratio 2.14), yet collide 120x more

**Date** 2026-09-16 · **Commit** `8db349f` · **Branch** `flow-response`
**Job** 13976736 (`jobs/calo_cooccupancy.sh`, 44s) · ckpt `multispecies_v2_s0/checkpoint_060000` · 200k showers
Follows [core-collision entry](2026-09-15-calo-collisions-are-in-the-core.md).

## Question

Collisions rise monotonically toward the shower core (reproduced here: 0.1179 at r/width < 0.25,
0.0000 beyond 5 widths; overall 0.0357 vs real 0.0003). Two readings: (a) the core is too DENSE,
(b) the density is fine and `PointCFM`'s i.i.d. per-cell draws collide where a real shower -- a set
of distinct channels -- cannot. Pre-registered: NN-spacing ratio gen/real ~1 -> (b); << 1 -> (a).

## Result

Within-shower nearest-neighbour spacing, same 4,000 showers on both sides, (d_eta, d_phi) about the core:

| | median | p10 | p90 |
|---|---|---|---|
| real | 0.00314 | 0.00128 | 0.03707 |
| generated | **0.00674** | 0.00176 | **0.10791** |

**Ratio 2.14 -- generated points are FARTHER apart than real cells, at every percentile.**
Neither pre-registered branch; the outcome ">1" was not anticipated.

## Reading

**(a) "core too dense" is not supported** -- the generated cloud is, if anything, less dense
locally. So the excess collisions happen DESPITE lower local density, which is what (b) predicts:
independent draws land on top of each other with no exclusion, while real cells tile without
overlap. For a hard-core/lattice arrangement the NN distance is LARGER than for independent points
at equal density, so a generator that were merely i.i.d. at the right density should show a ratio
BELOW 1; seeing 2.14 means it is also genuinely sparser.

## Caveat that limits how far this goes -- the statistic is 2-D

The NN distance uses (d_eta, d_phi) only. Real showers stack cells in successive layers at almost
the same transverse position, which makes their 2-D NN small for a LONGITUDINAL reason, not a
transverse-density one. So part of the 2.14 may be "real showers are aligned in depth and generated
ones are not" -- another form of the missing within-shower coherence -- rather than a transverse
density difference. This run cannot split those.

`shower_width` W/sigma is only 0.0505, so the overall extent matches; the sparser NN with matched
width points at the arrangement inside the shower, not its size. The fat generated p90 (0.108 vs
0.037) says the sparse end is much sparser -- small generated showers scatter their few points.

## Verdict

"Tune the core density" is RULED OUT as the fix. The evidence now favours the sampler: the
generator's cells do not avoid each other and are not arranged like real ones. That is the third
independent measurement pointing at the same architectural fact as the two-point coherence result.

## Next

- 3-D NN in mm, per layer (same-layer transverse spacing separately from cross-layer stacking) --
  splits the transverse-density and depth-alignment readings. Cheap: same job.
- Treat this as input to the set/attention design (option 2, still open with the user).
