# 2026-08-24 — the turning-point branch is 81-92% CALO-INTERNAL shower fragments, and that inverts the Phase 4 falsification

- **Commit / branch**: `99879a3` / `flow-response` (working tree)
- **Jobs**: 12874732 (`calo_reattribution_probe.py`), 12874782 (`calo_turning_branch_census.py`)
- Corrects [2026-08-16 Phase 4 FALSIFIED](2026-08-16-curler-tracker-coverage-PHASE4-FALSIFIED.md)
  and the turning-branch description in STATUS. Follows
  [calo over-splitting](2026-08-24-calo-shower-oversplitting.md).

## Hypothesis

J3a found 91% of calo depositors BORN INSIDE the calorimeter fall to the turning-point anchor
branch — geometrically forced, since a helix cannot be extrapolated forward to a face the particle
is already behind. If the turning branch is mostly calo-internal shower fragments rather than
curlers, then every statement conditioned on that branch is about the wrong population.

## Result

**(1) The branch is dominated by calo-internal fragments.**

| species | turning branch (of depositors) | **born INSIDE calo** | born outside |
|---|---|---|---|
| pi+ | 0.637 | **0.813** | 0.187 |
| pi- | 0.644 | **0.814** | 0.186 |
| e- | 0.736 | **0.914** | 0.086 |
| e+ | 0.732 | **0.922** | 0.078 |

**(2) That inverts the Phase 4 coverage number.** P(0 tracker hits), split:

| species | born inside (fragment) | **born outside (TRUE curler)** |
|---|---|---|
| pi+ | 0.990 | **0.045** |
| pi- | 0.989 | **0.043** |
| e- | 0.992 | **0.224** |
| e+ | 0.995 | **0.242** |

The logged "81% (pion) / 93% (e±) of curlers leave zero tracker hits" is the POOLED branch, and it
is carried almost entirely by particles created inside the calorimeter — which have no tracker hits
by construction, never having been in the tracker. **For genuine tracker-born curlers, 95.5% of
pions and ~77% of e± DO leave tracker hits.** The claim that tracks are "absent exactly where the
helix is weak" is an artifact of pooling two unrelated populations.

**(3) How the misreading happened — radius without z.** The born-inside fragments have vr median
412-470 mm, which reads like "born in the tracker". They are inside via **|vz| >= 3212 mm**: they are
ENDCAP shower fragments, at small radius and large z (the endcap spans r ~ 330-1160 mm and carries
83% of deposited energy). STATUS described the branch as "soft secondaries (median pT 0.27 GeV, born
at vr ~ 420 mm) that curl back first" — the pT and vr both match, so the description looked right
while naming the wrong physics. The genuine curlers are a separate, smaller population born at
**vr ~ 0** (beamline) with pT ~ 0.24 GeV, curling inside the tracker.

## Verdict — Phase 4 REOPENED (premise falsified, conclusion NOT yet re-established)

The 2026-08-16 entry's Q1 headline is contaminated and its conclusion does not follow. This does
**not** show Phase 4 works — only that the reason it was dismissed is invalid. Two things are now
unknown rather than settled:
- whether the track endpoint beats the helix for the ~8-19% TRUE curler population (Q3 was measured
  on the pooled branch, both on 2026-08-16 and in the corrected re-run job 12871830);
- how much of the calo core error that population actually owns.

Also note the true-curler population is SMALL (18.7% of the pion turning branch, ~8% of e±), so even
a strong result there is bounded — this reopens a narrower door than the original Phase 4 scope.

## Next

1. **Re-run Q3 restricted to born-outside depositors** (`calo_curler_track_probe.py` + a
   born-inside cut). Cheap, and it is the actual decider that has never been measured cleanly.
2. **Re-attribution (J3) separates the two populations by construction** — after it, the turning
   branch should be mostly genuine curlers, which is the regime where the tracker has signal.
3. Re-read anything conditioned on the turning branch, including Phase 0b's per-branch tightening
   table (pion 2.50x / proton 2.45x / e± 1.98x "turning pt") — those numbers pool the same two
   populations and the 2.5x "steady gain on every charged species" may be two different effects.
