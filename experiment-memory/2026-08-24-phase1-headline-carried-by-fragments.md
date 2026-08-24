# 2026-08-24 — the Phase 1 "e± core 0.585 -> 0.99" headline is carried by born-inside fragments; the anchor is BETTER than documented, the model worse

- **Commit / branch**: `e069764` / `flow-response`
- **Job**: 12879155 (`calo_core_diag.py --born {all,outside,inside}`, `electron_anchor/checkpoint_060000`)
- Follows [turning-branch contamination](2026-08-24-turning-branch-contamination-phase4-REOPENED.md)
  and [calo over-splitting](2026-08-24-calo-shower-oversplitting.md).

## Hypothesis

74% of e± calo depositors are born INSIDE the calorimeter and 91% of those receive the
TURNING-POINT fallback anchor (a helix cannot be extrapolated forward to a face the particle is
already behind). The `electron_anchor` slice is 73.4% turning branch (`anchor_mode` counts
91,624 / 121,136 / 587,238). The Phase 1 headline was measured on the pooled set, so it may describe
the fallback rather than the helix anchor.

## Result — CONFIRMED, in the residual frame

**Residual-frame gen/real per-bin spread** (the quantity the mixture actually fits, and the number
STATUS quotes as "e± FIXED: 0.585 -> 0.99"):

| population | multi-cell showers | mean | min-max |
|---|---|---|---|
| all (as logged) | 600,000 | **1.00** | 0.93-1.06 |
| born INSIDE (fragments) | 544,249 | 1.05 | 0.96-1.23 |
| **born OUTSIDE (real showers)** | 190,087 | **1.86** | **1.42-2.71** |

The pooled 1.00 averages a model that fits the easy 74% and **over-disperses real showers by 1.86x**.

**BUT the physical-frame ordering REVERSES, and that is what the gate sees**: all **1.29**, inside
1.21, **outside 1.15** — born-outside is the best-modelled population physically. Reason: the anchor
is exact by construction and carries most of the physical core for real showers. Residual sigma(phi)
**0.0157** vs physical **0.0874** = **5.6x tightening**, against 0.0945 / 0.1772 = 1.9x pooled. A
1.86x error on a component that is ~18% of the physical spread does limited damage.

## Verdict — two corrections, in OPPOSITE directions

1. **STATUS's "the core is solved" / "e± FIXED: 0.585 -> 0.99" is not supported.** In the frame the
   mixture fits, real showers sit at 1.86 (worst bin 2.71). The claim describes the fallback-anchored
   fragment population.
2. **The ANCHOR is better than documented** — 5.6x phi tightening on real showers vs 1.9x pooled.
   The frame is not the problem; the model's inability to fit the tight residual is.

Net practical impact is MODERATE, not catastrophic, because of the physical-frame reversal. This is
a correction to an attribution, not a retraction of Phase 1 as the best config.

## Next

- **Strengthens re-attribution**: afterwards the slice is entirely born-outside, where the anchor
  gives a sharp 5.6x-tightened target the model currently misses by 1.86x. That is a much better
  defined learning problem than the current blend of two populations.
- Re-read every per-bin core number measured on a pooled anchored slice the same way.
- The 1.42-2.71 range across bins says the failure is conditional, not a global scale error — worth
  knowing which (charge x pT) bins are worst before choosing a fix.
