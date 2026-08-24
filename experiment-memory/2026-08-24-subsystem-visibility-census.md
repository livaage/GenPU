# 2026-08-24 — subsystem visibility census: the tracker has the SAME missing-incidence hole as the calo, and only 9.2% of particles are seen by both

- **Commit / branch**: `99879a3` / `flow-response` (working tree)
- **Job**: none — login-node probes, particle-level arrays only (no hit arrays for A/B).
- **Script**: `scripts/subsystem_visibility.py` (census + determinism + neutral mechanism)
- Real data only, no model, no training. Follows
  [the Phase 4 falsification](2026-08-16-curler-tracker-coverage-PHASE4-FALSIFIED.md), which
  measured tracker coverage for the calo's curler branch; this generalises it to all particles.

## Hypothesis

Two questions raised by the Phase 4 result:

1. The calo is known to lack an incidence head (`P(shower | deposits)`, never `P(deposits | particle)`).
   Does the **tracker** have the same hole? Suspicion: yes, unnamed, because the honest tracker gate
   is built on the same filtered population.
2. Is the tracker-only / calo-only / both outcome deterministic given truth kinematics, with a small
   stochastic tail — or genuinely random?

## Result

**(1) YES, and the tracker's hole is the larger one.** Both slice builders filter identically —
`build_count_slice_stage2.py:24` (`keep = nh >= 1`) and `build_tracker_slice.py` (`--min_hits`
default 1) — and `CountHead` is a categorical over **1..48** with `clamp(n_hits - 1, 0, 47)`, so it
structurally cannot emit zero. The tracker models `P(n_hits | particle, n >= 1)`, exactly the calo's
`P(shower | deposits)`.

Shard 0, 6,471,875 particles (**denominator caveat**: `preprocessing.py:223` already applied
`visible_mask = (n_trk > 0) | (n_cal > 0)`, so these are rates among particles visible SOMEWHERE):

| | zero trace | modelled |
|---|---|---|
| **tracker** | **0.596** | 0.404 |
| calo | 0.313 | 0.687 (matches the logged 68.7%) |

By population: charged secondaries are 83% of all particles and **63.8% leave no tracker hit**;
charged primaries are only 2.5% zero-hit. P(0 tracker hits) peaks at **0.749** in pT 0.1-0.2 GeV and
falls to 0.092 above 5 GeV — the curler/soft signature.

**THE NUMBER: `trk-only 0.313 / calo-only 0.596 / both 0.092 / neither 0.000`.** Only **9.2% of
particles are seen by both subsystems.** This is a stronger and more general statement than the
Phase 4 entry's — it is not that curlers specifically are tracker-invisible, it is that the two
subsystems see **near-disjoint populations**. There is very little overlap to connect *through*,
which is the population-level reason the tracker→calo edge came out anti-complementary.

**(2) ~94% deterministic, with a real ~5% tail.** 3-way outcome from (log pT, eta, charge, primary,
vr, |vz|, log E, pdg class), 500k subsample, 70/30 split:

| model | test accuracy |
|---|---|
| majority-class baseline | 0.595 |
| tree depth 2 | 0.817 |
| tree depth 4 | 0.875 |
| **tree depth 8** | **0.940** |
| tree depth 16 | 0.961 |

Importance: **log_E 0.53**, |vz| 0.17, pdg_class 0.14, vr 0.08, log_pt 0.04, charge 0.02. Energy
dominates; **species is only third**. Per-species purity is mostly ~0.5 (e− 0.494, π± ~0.49,
K± ~0.41) — only the stable neutrals are clean (K0L 0.999, π0 1.000, n̄ 0.997). What separates
outcomes *within* a species is energy and production vertex:

| rule | n | outcome | purity |
|---|---|---|---|
| charged, born vr > 1100 mm | 662,699 | calo-only | **0.986** |
| charged primary, pT > 1 GeV | 58,980 | both | 0.721 |
| charged, born vr < 50 mm | 715,821 | trk-only 0.476 / both 0.455 | 0.476 |

**(3) Neutral visibility is the conversion/interaction coin flip.** Raw source, 300 events, joined
**by `event_id`**:

| population | n | P(trk) | wrong-event null | P(chg daughter) | **P(trk \| daughter)** | **P(trk \| no daughter)** |
|---|---|---|---|---|---|---|
| charged | 249,990 | 0.373 | 0.096 | 0.304 | 0.385 | 0.368 |
| **γ** | 47,131 | 0.114 | 0.117 | 0.613 | **0.185** | **0.001** |
| π0 | 16,857 | 0.000 | 0.131 | 0.020 | 0.000 | 0.000 |
| K0L | 1,406 | 0.000 | 0.118 | 0.499 | 0.000 | 0.000 |
| K0S | 1,394 | 0.000 | 0.141 | 0.711 | 0.000 | 0.000 |
| n̄ | 915 | 0.002 | 0.134 | 0.508 | 0.004 | 0.000 |
| **n** | 7,258 | **0.481** | 0.095 | 0.188 | 0.070 | **0.576** |

A photon touches the tracker at **0.185 if it converted and 0.001 if it did not** — a factor ~200,
and conversion is the only thing separating them. π0/K0L/K0S/n̄ are flat zero, as they must be. The
daughter conditional does nothing for charged particles (0.385 vs 0.368), as it must not.

**Neutrons invert it**: P(trk | charged daughter) 0.070 vs P(trk | none) **0.576**, so neutron hits
are NOT from recorded charged daughters — plausibly nuclear recoil / hadronic interaction whose
secondaries fell below the truth-record threshold. Different mechanism, unresolved.

**Photon hits are real isolated deposits, not double-booked daughter tracks**: where a photon and
its daughters both have hits, **75% of the time the hit sets are entirely distinct** (25% share one
position — the conversion vertex firing the same sensor). Median **1 hit** per photon, r 32-1031 mm.

## Methodology note — a join bug caught in flight

The first version of the neutral probe joined `particles` to `tracker_hits` **by row index**. Row
order agreement between the two subsets is **0.000**, and the broken join produced a flat
P(trk hits) ~ 0.11-0.14 across *every* neutral species including π0 — physically impossible (a π0
travels ~25 nm) and a perfect imitation of a real measurement. `scripts/subsystem_visibility.py`
now carries a **wrong-event null column** so this failure mode is visible in the output rather than
inferred. This is the CLAUDE.md "join by event_id, never row index" rule, re-earned.
`preprocessing.py:271-273` uses `build_event_index` and is NOT affected — every stage2 number above
stands.

## Verdict — KEPT (measurement); reshapes the incidence-head work

Nothing was trained. Three things change:

- The **incidence head is now a two-subsystem object**, not a calo-only gap, and it is the cheapest
  open item on the board: 94% predictable with a depth-8 tree, and the residual ~5% is the physical
  conversion/interaction probability, which a **generator should SAMPLE, not predict**. That makes
  the target well-posed rather than a ceiling.
- The 9.2% overlap is the population-level statement of why the tracker→calo edge failed.
- **A decay head is NOT indicated** — see Next.

## Next

- **Do not build a decay head.** `pileup_generator_plan.md:359-379` (M0) already established that
  genuine two-body decays are a small minority and that secondary production is dominated by
  material effects, and that injection is a **generation-time** concern: current training conditions
  on TRUTH particles, which already contain every conversion daughter at its vertex. The conversion
  is an INPUT, not something to predict. This census sharpens that: the useful object is
  `P(leaves trace | E, eta, vr, pdg, has-charged-daughter)`.
- **BLOCKER for that feature**: stage2 stores `parent_id` (`particle_aux[:, 1]`) but NOT the
  particle's own id, so the parent→child graph **cannot be rebuilt downstream**
  (`preprocessing.py:233` keeps `particle_ids` in the per-event dict; it is never written to the
  npz). Adding it is a one-line preprocessing change and unlocks the single strongest neutral
  feature (0.001 → 0.185).
- **Open**: the neutron mechanism (48% tracker-visible, not via recorded daughters).
- **Open**: 44.7% of modelled charged particles have exactly 1 tracker hit (median 2). Together with
  the photon single deposits, the tracker's "track" population is majority stubs and isolated hits.
  Whether those belong in the tracker slice at all is a scope question nobody has asked.
