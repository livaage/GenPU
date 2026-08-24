# 2026-08-16 — curlers are INVISIBLE to the tracker: the track-endpoint anchor (Phase 4) is falsified

- **Commit / branch**: `99879a3` / `flow-response` (probe untracked at run time)
- **Job**: 12471165 (`jobs/calo_curler_track_probe.sh`, CPU only, no model, no training)
- **Script**: `scripts/calo_curler_track_probe.py`
- **Data**: held-out shard 5, 400k depositing showers per species, real tracker hits + real calo cells
- **Artifacts**: `plots/calo/metrics/curler_track_{pion,electron}.{json,png}`

## Hypothesis

63% of pion / 72% of e± calo showers are anchored at the **vacuum-helix turning point** — soft
secondaries that curl back before the calo face, with their cells booked to the parent. That anchor
is worth ~2.5x vs 15-24x on the face branch, and it is a proxy: the vacuum helix knows nothing about
dE/dx, multiple scattering, nuclear interaction or early stopping.

Plan Phase 4 (`calo_tracker_coupling_plan.md` §7, billed as "the novel contribution") proposes
replacing it with the generated **track's outer state**. Refined this session to: condition on the
track's **terminal** state rather than the production vertex, which would dissolve the
barrel/endcap/turning **branch** problem entirely — the endpoint *is* the anchor, a continuous
function of the track, so the 10-15x scale discontinuity the GlobalHead cannot express disappears.

Prerequisite, and the point of this probe: **do these particles leave enough tracker hits for an
endpoint to mean anything?**

## Change

No training. On real data, per anchor branch: (Q1) tracker-hit coverage; (Q2) vacuum turning radius
vs the outermost radius actually reached; (Q3) spread of `core − anchor` using the real track
endpoint vs the current helix anchor — Phase 0b's methodology with the endpoint substituted in.
Shower core computed **exactly** as `build_calo_slice.py` defines it (unweighted mean of
particle-relative cell offsets, φ wrapped) and `core_anchor` already returns particle-relative
offsets, so both anchors sit in the same frame as every logged Phase 0b / Phase 1 number.

## Result — Q1 killed it before Q3 mattered

**Fraction with ZERO tracker hits:**

| branch | pion | e± | pion median n_hits | e± median |
|---|---|---|---|---|
| barrel | 47.0% | **98.1%** | 2 | 0 |
| endcap | 23.9% | 85.6% | 9 | 0 |
| **turning (curlers)** | **81.1%** | **93.0%** | **0** | **0** |

**81% of pion curlers and 93% of e± curlers leave no tracker hits at all.** No refinement of the
anchor definition changes that — you cannot condition on a track that does not exist. Physically
consistent: median pT 0.045–0.26 GeV, and at B = 3.07 T a 0.045 GeV track has R ≈ 49 mm.

**e± have essentially no tracker information in ANY branch** (86–98% empty). Likely because a
barrel-branch e± at 0.136 GeV has R ≈ 148 mm and can only reach r = 1259 mm if **born past ~1100 mm**
— conversions and π0 decays in the outer tracker with almost no tracker path left. Consistent with
the census (e+ deposit rate 94.5%, "e+ come from conversions and pi0 decay") and with the incidence
probe's reliance on `vr`. NOT directly measured here — median `vr` per branch was not output; a
one-line addition if it needs confirming.

**Q3 — the track endpoint is worse than the helix on every branch and both species** (gain > 1 would
favour the track):

| branch | pion φ | e± φ | pion η | e± η |
|---|---|---|---|---|
| barrel | 0.06 | 0.14 | 0.15 | 0.21 |
| endcap | 0.10 | 0.33 | 0.29 | 0.45 |
| turning | 0.63 | 0.73 | **1.04** | **1.04** |

**Q2** — helix turning radius median 570 mm vs real outermost hit 318 mm (Δr median −165 mm), and
|Δφ| median **1.5672 rad ≈ π/2** (e± 1.5709) — exactly what a *uniform* Δφ gives, i.e. the vacuum
turning point's φ is **uncorrelated** with the real endpoint's φ. Whatever the 2.4x turning-point
gain comes from, it is not φ agreement with the actual track.

### Caveats — what this does and does not rule out

- On the **face branches** the Q3 comparison is unfair to the track: the endpoint used is the raw
  outermost hit *position*, well inside r = 1259 mm, discarding the remaining bend the helix models
  correctly. A fair test would extrapolate from the last hit **and its direction** to the face.
- On the **turning branch** the comparison is fair (both are "where it stopped going outward") and
  still shows no gain — measured only on the favourable 19% / 7% that have hits at all.
- Preprocessed tracker hits are sorted **r-ascending, not by time**, so the last stored hit is the
  OUTERMOST, not the chronologically final one. The right panel of the plot shows the real
  distribution is **spiky at discrete layer radii** — the outermost *recorded* hit is quantised by
  detector layers and is a biased-low proxy for the stopping point. Genuinely testing "the track
  knows where it stopped" needs hit times from the raw source.
- The Q1 coverage number is robust to all of the above and is the decisive one.

## Verdict — ABANDONED for the curler population; Phase 4's headline motivation is falsified

**Tracks exist exactly where the helix already works (face branches, 15-24x) and are absent where
the helix is weak (curlers, 2.5x).** That is anti-complementary — the opposite of the premise. The
plan's own kill criterion ("kill the edge if generated-track conditioning is worse than the
truth-helix anchor") fires here on *real* tracks, before any exposure-bias question arises.

Second casualty: the plan's **e± caveat branch (ii)** — "if brem is the mechanism the e± core needs
the tracker, not a truth helix" — is dead too. There is no tracker signal for e± anywhere. The
stiff-e± anchor inversion (Phase 0b gain 4.52x → **0.43x**) must be fixed from truth kinematics or
not at all.

What survives is narrow: a last-hit **position + direction** extrapolation for the face-reaching
**pion** population (endcap 76% coverage, median 9 hits). But the helix already achieves 15-24x
there, so the headroom is small and it is no longer a novel-contribution story.

## Next

1. **The smooth helix↔line anchor blend for e±** is now the ONLY available fix for the brem problem.
   The hard-split `auto` anchor is the logged negative (best frame in isolation, 2.69x pooled, but
   worst residual spread of any run at 0.646 because the pT split is a discontinuity smooth `log_pt`
   conditioning cannot express). Replace with a continuous blend
   `anchor = w(pT)·line + (1−w(pT))·helix`, `w` a sigmoid calibrated on the measured per-bin
   crossover (helix 4.52/2.14/1.36/1.08/1.04/0.43 vs line 1.54/1.94/2.55/2.63/2.55/2.38, crossing
   between 0.028 and 0.047 GeV).
2. **The turning-point anchor stays a proxy and there is no better one available.** Its ~2.5x is the
   ceiling for 63-73% of showers unless something other than the tracker supplies the stopping point.
3. Phase 4 should be **rewritten, not deleted**: the surviving scope is face-reaching pions with a
   direction-aware extrapolation, and the honest framing is a small refinement rather than the
   project's novelty claim.
