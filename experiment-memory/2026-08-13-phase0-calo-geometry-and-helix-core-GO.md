# 2026-08-13 — Phase 0a/0b: calo geometry derived, helix-core decision gate = GO (all species)

- **Date:** 2026-08-13 · **commit:** 9e0bbc2 (working tree) · **branch:** flow-response
- Plan: [calo_tracker_coupling_plan.md](../calo_tracker_coupling_plan.md). No training; login-node only.
- New: `scripts/build_calo_geometry.py`, `scripts/calo_helix_core_probe.py`, `calo_geometry.json`.

## 0a — calo geometry (was missing entirely)
`detector_geometry.py` is TRACKER layer geometry and stage2 calo hits carry only (eta, phi), so
nothing downstream knew where the calo is. Derived from the raw `calo_hits` (x, y, z, detector):

| det | subsystem | geometry | position | E frac |
|---|---|---|---|---|
| 10 | ECAL barrel | barrel | r 1259-1442 | 15.8% |
| 13 | HCAL barrel | barrel | r 1649-2248 | 0.9% |
| 9 / 11 | ECAL endcap (-z / +z) | endcap | \|z\| 3212-3420 | 27.3% / 28.4% |
| 12 / 14 | HCAL endcap (-z / +z) | endcap | \|z\| 3648-5330 | 13.4% / 14.2% |

**Front face: barrel r = 1259 mm, endcap |z| = 3212 mm, eta transition 1.666.**
- **The endcaps carry 83% of the deposited energy** — min-bias pileup is very forward — so the
  anchor must extrapolate to a PLANE for most showers, not just a cylinder. Both `helix_at_r` and
  `helix_at_z` are needed (both already exist in `helix.py`).
- Gotcha: classifying barrel vs endcap by ABSOLUTE coordinate spread misreads the deep HCAL endcap
  (|z| 3648-5330) as a barrel, because its own longitudinal depth exceeds its radial extent. Use the
  RELATIVE spread (CV) instead — that classifies all six correctly.

## 0b — DECISION GATE: how much of the core does a truth-helix extrapolation explain?
Core measured exactly as the slice builder does (mean of wrapped per-cell deltas), once relative to
the PARTICLE direction (what the model predicts today) and once relative to the HELIX prediction at
the front face (what it would predict after the frame change). Robust sigma = IQR/1.349.

| species | valid | \|core\| med: particle → helix | sigma_phi: particle → helix | **phi tightening** | eta tightening |
|---|---|---|---|---|---|
| pion | 96.8% | 1.543 → 0.377 | 1.207 → 0.265 | **4.6x** | 8.8x |
| e± | 98.3% | 0.297 → 0.043 | 0.175 → 0.0156 | **11.2x** | 4.2x |
| proton | 94.2% | 1.803 → 0.119 | 1.376 → 0.054 | **25.5x** | 17.5x |
| photon (control, straight line) | — | — → 0.009 | — → 0.0072 | 2.9x | 7.6x |

**VERDICT: GO for every species.** The gain scales as 1/pT exactly as bending physics predicts —
pion 15.8x at pT 0.12 GeV falling to 3.0x at 0.83; e± 15.2x at 0.015 falling to 5.2x at 0.25.

**The e± puzzle is resolved.** e± showed no charge asymmetry in core_phi (median ~0 for both signs)
despite a median |core| of 0.30, which looked inconsistent with bending. It was option (i) from the
plan: soft e± (pT 0.015-0.25 GeV) bend through MORE than pi, so `wrap_pi` symmetrises the signed
median. The helix computes the actual bend including >pi and explains it — 11.2x tightening.
**The bremsstrahlung hypothesis is NOT needed**, and Phase 1 does not have to branch.

**Photon control behaves correctly**: neutrals go straight, their core is already tiny (0.009 rad),
and the residual 2.9x comes from propagating the production vertex (vz spread) rather than bending.

## Caveat to carry into Phase 1
The helix fixes the BULK, not the tail: e± sigma_phi drops 11.2x on the robust measure but only
2.2x on the plain std (0.723 → 0.327); pion 4.6x robust vs 1.16x std. So the anchored core is
sharply peaked with a heavy outlier tail (wrong-crossing picks, curlers near their turning radius,
decays in flight). That is a much easier target for a quantile-normalised head with a BOUNDED
inverse than the old broad distribution, but it is not a pure Gaussian — do not expect the residual
mixture to be trivial, and keep the bounded inverse.

## Verdict / next
Phase 1 (truth-helix anchored core) is GO on the strongest evidence of the session — a 4.6-25.5x
frame tightening on the quantity that carries 83-93% of the width variance, with no training and no
exposure-bias risk. Proceed to:
- **0c: separate the per-head conditioning trunks** (still outstanding; a precondition for reading
  Phase 1's result cleanly).
- **Phase 1**: `build_calo_slice.py --core_anchor helix` storing `core - helix_pred`, with the
  anchor recomputed at generation from truth conditioning; handle the 1.7-5.8% invalid
  extrapolations explicitly rather than silently.
