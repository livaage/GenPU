# SUPERSEDES today's grid entry: the ECAL pitch is 5.1 mm EXACTLY, and the radial banding was covering a 0.01 mm error

**Date** 2026-09-15 · **Commit** `32d0cab` · **Branch** `flow-response`
**Login node, real data, no model, no job** (shard 0, 6,000-12,000 events). Corrects
[2026-09-15 grid entry](2026-09-15-calo-cell-grid-ecal-solved-hcal-open.md), which is left in place
per the append-only rule. **Where the two disagree, this one is right.**

## What triggered it

The HCAL follow-up ran ECAL as a control and the control failed in a way it should not have. With
the fitted lattice but NO radial banding, ECAL on-grid was **0.5144** — yet the job had reported
0.999. The only difference was `--r_band 200`. A model that needs arbitrary radial bins to work is
not the model; it is a fit absorbing an error.

## Diagnosis

Within one sector (det 9 layer 0, sector 8, 3,396 cells over r 317-1481 mm), fitted per 100 mm band:

| r band | 300-400 | 500-600 | 700-800 | 900-1000 | 1100-1200 | 1300-1400 |
|---|---|---|---|---|---|---|
| angle (deg) | -0.000 | -0.000 | -0.000 | -0.000 | -0.000 | -0.000 |
| origin v (mm) | -1.289 | -1.662 | -2.049 | -2.440 | +2.261 | +1.896 |

**The angle is constant to 3 dp; only the ORIGIN drifts, and it drifts SMOOTHLY and monotonically**
(the jump between -2.440 and +2.261 is the modulo wrap at pitch/2 = 2.545). Discrete modules would
give discrete jumps. A smooth linear phase drift is the signature of an assumed pitch that is
slightly wrong: over a run of n cells the phase slips by n*(p_true - p_assumed).

Solving from the drift, -1.905 mm over dr = 1000 mm, predicts p_true ~= 5.0997.

## Result — confirmed by direct refinement

Refining the pitch by MAXIMISING circular concentration over the whole sector (no histogram, no
binning, no mode):

| | pitch | whole-sector R | on-grid, ONE origin for r 317-1481 | median residual |
|---|---|---|---|---|
| mode estimate | 5.0900 | 0.8791 | 0.5289 | 0.4731 mm |
| **refined** | **5.09998** | **1.0000** | **1.0000** | **0.0011 mm** |

**The ECAL endcap transverse pitch is 5.1 mm exactly** — a round design number, as it should be.
`pitch_from_nn` took the mode of a histogram with 0.02 mm bins and returned 5.09; that 0.0100 mm
error accumulates to a full pitch over ~510 cells, which is why every fit needed radial bins narrow
enough to keep the slip below half a pitch.

**So `--r_band` was never physics.** It was scaffolding compensating for a bad constant, and I said
earlier in the session that the scaffolding parameters "don't appear in the answer". They did.

## Corrections to the superseded entry

1. ECAL pitch is **5.1000**, not 5.0900.
2. ECAL on-grid is **1.0000** with median residual **0.0011 mm**, not 0.9989-0.9997 / 0.080-0.088 mm.
   The transverse axis is now as exact as the longitudinal one (which cleared 0.0000), so the
   "residual is 0.08 mm where longitudinal cleared 0.0000, probably the pitch constant" note in that
   entry was the right suspicion and is now resolved rather than left open.
3. The snap needs **one origin per phi sector and no radial index**. The superseded entry's formula
   was right; the numbers feeding it were not.
4. The claim that ECAL is "analytic, not a LUT" **stands and is strengthened** — 32 sectors, one
   angle and one origin each, one global pitch.

## HCAL, done properly this time

The first HCAL test in this session was INVALID: it reused the ECAL shortcut (angle = 0, so u=x,
v=y) without checking, and HCAL sectors are not axis-aligned. Redone with the angle fitted per
sector, then the pitch refined in the rotated frame (det 12 layer 0, 12,000 events):

| sector | 4 | 8 | 12 | 16 | 20 | 26 |
|---|---|---|---|---|---|---|
| angle (deg) | **44.994** | **0.008** | **44.993** | **-0.059** | **44.983** | **22.465** |
| refined pitch | 29.9936 | 29.9924 | 29.9936 | 29.9930 | 29.9942 | 29.9942 |
| R | 0.9796 | 0.9736 | 0.9671 | 0.9676 | 0.9733 | 0.9745 |
| on-grid | 0.9611 | 0.9477 | 0.9495 | **0.6739** | 0.9498 | 0.8562 |
| median residual (mm) | 0.103 | 0.123 | 0.126 | **0.356** | 0.148 | 0.241 |

- **HCAL's pitch was already essentially right** (29.9930 vs the mode's 29.9900) — so HCAL's problem
  is NOT the pitch error that ECAL had. Refinement buys almost nothing here.
- **Module ORIENTATION varies by sector**: 0 deg, 22.5 deg and 45 deg all appear, and 45 deg lands on
  sectors 4, 12, 20 (= 4 mod 8) while 0 deg lands on 8, 16 (= 0 mod 8). That patterning suggests the
  true symmetry is 8-fold with alternating module orientations and **32 is an alias**, which would
  also explain why the parsimony rule settled on 32.
- Even with angle fitted and pitch refined, R caps at ~0.97 and on-grid is sector-dependent
  (0.67-0.96). **HCAL is not a single rotated square lattice.**
- Projective towers are FALSIFIED as the explanation: pitch is exactly 29.990 in every radial band
  from r=362 to r=3047 (6 bands, all identical), so cell size does not grow with radius.

## Verdict

**ECAL endcap: CLOSED.** 32 sectors, per-sector angle and origin, pitch 5.1 mm exactly, depth on the
48-plane ladder. On-grid 1.0000, residual 1 micron. 79.6% of cells / 76.0% of energy (e±).

**HCAL endcap: still open, but the question has changed** — from "what is the pitch" (answered:
29.993, and it was never the problem) to "what is the module orientation pattern", with 8-fold
alternating orientation as the leading candidate.

## Next

- Re-run the full derivation with the refined-pitch estimator (maximise concentration, do not take a
  histogram mode) and `--r_band` effectively off, to confirm 1.0000 across all ECAL layers and both
  endcaps. Expect the radial-band parameter to become irrelevant.
- **Fix `pitch_from_nn`**: the mode is fine as a SEED, but the value must then be refined by
  concentration maximisation. As it stands it silently quantises the answer to the bin width, and it
  has now produced two separate wrong results today (the sqrt2 diagonal, and this 0.01 mm slip).
- HCAL: test the 8-fold-with-alternating-orientation hypothesis directly, and check whether the bad
  sectors (16, 26) are bad or merely thin (320 cells each here).
