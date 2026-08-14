# 2026-08-13 — Phase 0c: separate per-head conditioning trunks (hygiene fix; effect tracks data volume)

- **Date:** 2026-08-13 · **commit:** 9e0bbc2 (working tree) · **branch:** flow-response
- Jobs: 12372653 (photon+pion), 12372654 (e±), 12373095 (photon step-matched, pending).
- Plan step 0c of [calo_tracker_coupling_plan.md](../calo_tracker_coupling_plan.md).

## Hypothesis
One shared `ParticleConditioning` MLP fed all three heads, and the summed loss back-propagated every
head's gradient into it — so changing ONE head's task silently moved what the OTHERS saw. Measured
twice (width_norm, ctx_norm): GlobalHead-only changes degraded the untouched energy head identically
(e± cell_logE 0.017 → 0.081, frac_near_floor AUC 0.616 → 0.858). Giving each head its own trunk makes
that coupling impossible. Primarily EXPERIMENTAL HYGIENE — a precondition for attributing Phase 1 —
not a quality change. Caveat stated in advance: shared trunks are standard multi-task practice and
can act as a regulariser, so separation might cost quality; requirement was NO REGRESSION.

## Change
`CaloFlow(separate_trunks=True)` builds `cond_glob` / `cond_points` / `cond_energy`;
`cond_embed(cont, pdg, head=...)` dispatches; `sample_showers` uses per-head embeddings (computed
per shower, indexed by cell — no extra cost); `from_checkpoint` infers the mode from the state dict.
Trainer gains `--separate_trunks`. Cost measured: **+10,640 params** (252,735 vs 242,095, ~4%).

**Gradient isolation verified directly** — backward on the GLOBAL loss alone:
```
shared    -> cond: 1.4993
separate  -> cond_glob: 1.4213 | cond_points: 0.0000 | cond_energy: 0.0000
```

## Result — effect tracks DATA VOLUME, not the hypothesis
Same slices, same steps, same recipe as each shared-trunk reference; separation the only change.

| species | cells in slice | gate10 | gate8 | cell_logE | frac_floor AUC | verdict |
|---|---|---|---|---|---|---|
| pion | 39M | 0.993 → **0.992** | 0.809 → **0.797** | 0.020 → **0.018** | 0.666 → **0.627** | better |
| e- | 8.5M | 0.971 → 0.971 | 0.773 → **0.739** | 0.017 → **0.016** | 0.616 → **0.582** | better |
| e+ | 8.5M | 0.968 → **0.966** | 0.760 → **0.741** | 0.017 → **0.016** | 0.616 → **0.592** | better |
| photon | 4.2M | 0.805 → 0.850 | 0.558 → 0.679 | 0.016 → 0.027 | 0.510 → 0.592 | **REGRESSED** |

The two larger-data species improve; the smallest-data species regresses — the shape expected if the
shared trunk was lending statistical strength, which only pays where data is thin (photon is ~10x
thinner than pion). e± — the species that exhibited the interference twice — improves, which is the
most direct evidence that removing the coupling is the right move where there is data to support it.

**The photon regression looks like an OPTIMISATION cost, not a modelling one.** At 40k steps:
`val_cfm` 1.3933 (separate) vs 1.3936 (shared) — the point flow is unaffected; `val_gnll` **-1.226**
vs -1.081 — the global head is BETTER separated; and `val_ehl` was still descending
(0.1730 → 0.1654 → 0.1631 → 0.1547). All the damage is in the energy head, whose trunk now trains on
one loss instead of inheriting a representation three losses helped build. Step-matched test at 120k
(separate vs shared) is job 12373095.

## ADDENDUM (job 12373095) — photon step-matched test: the regression is REAL, not undertraining
| photon | gate10 | gate8 | cell_logE | frac_floor AUC |
|---|---|---|---|---|
| shared 40k | 0.805 | 0.558 | 0.016 | 0.510 |
| separate 40k | 0.850 | 0.679 | 0.027 | 0.592 |
| shared 120k | 0.810 | **0.537** | 0.017 | **0.521** |
| separate 120k | 0.853 | 0.680 | **0.016** | 0.610 |

The undertraining hypothesis was **half right**: `cell_logE` recovered fully with more steps
(0.027 → 0.016, matching shared), so that gap WAS optimisation. But the gate did not move
(0.679 → 0.680), so the overall conclusion was WRONG — the photon regression is real.

What remains is the **floor fraction** (0.610 vs 0.521) at essentially IDENTICAL likelihood
(val_ehl 0.1507 separate vs 0.1473 shared; val_cfm 1.3804 vs 1.3807; val_gnll -1.3853 vs -1.3332).
Same NLL, very different gate — a calibration the summed loss does not see.

**Mechanism that fits**: the floor-cell fraction depends on shower MULTIPLICITY, which is the
GlobalHead's variable. With a shared trunk the energy head implicitly saw multiplicity-relevant
features through the global head's gradient; separating removes that channel. Species ordering
matches exactly — photon (~4 cells/shower, threshold cells a large share) hurt most, e± (~9)
mildly helped, pion (~15) helped most.

**So separate trunks are NOT pure hygiene**: they trade a genuinely useful information channel for
isolation. Untested follow-up suggested by the mechanism: with separate trunks, condition the energy
head on `log_n` ONLY (not `total_logE`). That supplies the multiplicity signal explicitly and
cleanly, instead of via trunk bleed. Note the earlier `energy_use_glob` failure confounded three
changes at once (total_logE + log_n + qt_total, on a SHARED trunk), so it does not rule this out.

## Verdict (interim)
- **Adopt for pion / e± / multi-species**, where it is both hygienic and mildly beneficial.
- **Photon pending** the 120k step-matched test. If separate catches up → adopt globally. If it
  plateaus short → the fallback is loss-scale balancing on a shared trunk, which reduces the coupling
  without removing it.
- **Key framing for Phase 1**: separate trunks can serve as a DIAGNOSTIC INSTRUMENT (the config in
  which the helix-anchor experiment is run, so its effect is attributable) independently of whether
  it is the shipping config. A photon regression should not veto a clean read on the anchor.

## Next
Phase 1 (truth-helix anchored core), run with separate trunks so the result is attributable. The
anchor has a 4.6-25.5x frame tightening behind it from
[Phase 0](2026-08-13-phase0-calo-geometry-and-helix-core-GO.md).
