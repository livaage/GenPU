# Calo–tracker coupling: plan

_Written 2026-08-13, end of the calo audit session. Read `STATUS.md` and the
`experiment-memory/2026-08-13-*` entries first — this plan is built on their measurements._

## 1. Where we are

**The calo's dominant defect is the shower CORE**, not the intrinsic shower size and not energy:

| fact | value | source |
|---|---|---|
| core share of the "shower width" variance | 83% (e±), 93% (pion) | width-in-global entry |
| generated core spread vs real, per (charge × pT) bin | **0.58×** (e±), 0.94× (pion) | core diag |
| `width_std` single-feature gate AUC | 0.90 (e±), 0.96 (pion) | width-gate entry |
| median \|core\| | 0.30 rad (e±), **1.54 rad** (pion) | core diag |

Four separate attempts to fix this by changing the GlobalHead's **inputs or representation** all
failed to move the mechanism (energy-glob conditioning, cell partition, width normalisation, context
normalisation — the last improved pions slightly but left the core spread at 0.93×, and lost on e±).
In every case the 8-component Gaussian **mixture itself** stayed in place. The e± core spread has
not moved off 0.56–0.58× through any of them.

**Two structural facts learned the hard way:**
1. **The three heads share one `ParticleConditioning` MLP.** Changing only the GlobalHead's task
   degraded the *energy* head twice with an identical signature (e± `cell_logE` 0.017 → 0.081,
   `frac_near_floor` AUC 0.616 → 0.858). Every per-head experiment is confounded until this is fixed.
2. **Conditioning on SAMPLED quantities backfires; conditioning on TRUTH does not.** The energy head
   conditioned on the sampled global blew up the cell marginal. Context normalisation keyed on truth
   (charge, pT) did not have that failure mode. This distinction governs the whole plan.

## 2. Design principles (from this project's own record)

- **Change the frame, not the density model.** The two biggest wins here were coordinate changes:
  v3 surface-local (tracker gate 0.9996 → 0.80) and the v4 helix anchor (coherence fixed). Today's
  four failures were all density/normalisation patches. The core is a 1.5 m magnetic-bending
  displacement that we are asking a mixture to predict from vertex kinematics — a frame problem.
- **Measure before training.** The two things that worked today (physical bounds, the width gate
  feature) came from measurement; the four that failed came from plausible reasoning.
- **Every change gets a mechanism check**, not just a gate number. The gate moved on `pion_ctx`
  while the mechanism (core spread) did not — without the check we would have drawn the wrong lesson.
- **Truth conditioning is free; generated conditioning must be earned** (train teacher-forced,
  evaluate both ways, and report the gap).

## 3. Phase 0 — prerequisites and decision gates (cheap, no training)

**0a. Calo geometry.** `detector_geometry.py` is *tracker* layer geometry; stage2 calo hits are
(η, φ, logE, frac, detector) with **no radius**. Derive the calo surfaces from the RAW source
(`calo_hits` carries x, y, z): per `detector` value, the barrel radius / endcap z, and the ECAL front
face. Needed to know what surface to extrapolate to. → `scripts/build_calo_geometry.py`

**0b. Helix-vs-core measurement — THE DECISION GATE.** `helix.py:helix_at_r()` already exists.
For real showers, compute the helix prediction (η, φ) at the calo face from truth (pT, η, φ, charge,
vertex) and measure how much of the core it explains:
`spread(core − helix_pred)` vs `spread(core)`, per species and per pT bin.
- **Go** if the residual spread is materially smaller (target ≳2× tighter for pions).
- **No-go / branch** if it is not — see the e± caveat below.

**0c. Separate the per-head conditioning trunks** (or reweight the global NLL) and re-baseline
photon / pion / e±. Precondition for reading every later phase. Cheap.

**e± caveat to test in 0b.** e± show *no* charge asymmetry in core_φ (median ≈ 0.00 both signs)
despite a median |core| of 0.30 — inconsistent with simple bending. Two candidate explanations:
(i) soft e± bend through more than π, so `wrap_pi` symmetrises the signed median — in which case the
helix explains the core *very* well and this is the ideal case for the anchor; or (ii) the deposit
comes from **radiated bremsstrahlung photons**, which travel straight from the radiation point, so
the relevant direction is the electron's direction *where it radiated*, not at the vertex — in which
case the truth helix will explain little and the fix needs the **tracker** (Phase 4), because that is
what knows where the electron went and where it stopped. 0b distinguishes these, and either outcome
is informative.

## 4. Phase 1 — truth-helix anchored core

Store the core as a **residual from the helix prediction** at the calo face; generation adds the
(deterministic, truth-conditioned) anchor back. No exposure-bias risk — the anchor is a function of
truth conditioning only, exactly like the existing `cont` features.

- Slice: `--core_anchor helix`, storing `core − helix_pred` and the anchor for reconstruction.
- Handle non-intersecting helices (curlers with 2R < r_calo) and neutrals (straight-line limit —
  the same code path with q = 0) explicitly, and count them.
- **Mechanism check**: per-bin core spread ratio → target ~1.0 (from 0.58× / 0.93×).
- **Gate**: `event_gate_auc_width`, plus `width_std` / `width_mean` single-feature AUCs.

## 5. Phase 2 — GlobalHead: Gaussian mixture → normalizing flow

Independent of Phase 1 and targets the *other* half of the problem (mean-regression on heavy-tailed
conditionals). Literature is unambiguous: **every CaloClouds version uses a normalizing flow
("ShowerFlow") for exactly these per-shower globals**, and CaloClouds1's explicitly includes the
shower **centre of gravity** — our core. CaloClouds3 slims it to 12 affine + 2 spline couplings, so
it is cheap. Our 8-component GMM is the "lite" in CaloClouds-lite and is the piece that resisted
four patches.

Sequence Phase 1 and Phase 2 **separately** (not together) so the mechanism check can attribute.

## 6. Phase 3 — incidence head

Every calo model and gate so far is conditioned on `n_cells ≥ 1`: P(shower | deposits), never
P(deposits | particle). Only 68.7% of particles deposit, species-dependent (e⁻ 50%, e⁺ 95%, n 35%).
A logistic probe on the standard contract already reaches **AUC 0.897**. Add a Bernoulli incidence
head; required before any honest *full-event* calo gate. Later upgraded by Phase 4 (whether a
particle deposits ≈ whether its track survives the tracker).

## 7. Phase 4 — track-conditioned calo (the novel contribution)

Replace the truth-helix anchor with the **generated track's outer state** (last-hit position and
direction). This adds what a helix cannot know: multiple scattering, energy loss, nuclear
interactions, early stopping, and — if the e± caveat resolves that way — where an electron radiated.

**Structure: directional conditioning, not symmetric joint training.**
Factorise `p(track | particle) · p(shower | particle, track)`. This buys the physics correlation with
none of the interference risk, and it is also the honest answer to "should we train them together":

- We have *direct evidence* that sharing a trunk across heads with different tasks causes
  interference (§1, fact 1). Two modalities as different as an AR token model and a point flow would
  amplify it, not dilute it.
- The tracker (AR, 191 µs, gate 0.61) is the current deliverable. Joint training risks degrading it
  to help the calo — a bad trade at this stage.
- The correlation we want is *directional* (track places the shower), which the factorisation
  captures exactly. Symmetric joint training is not required for it.

So: keep separate trunks and training schedules; add the track → calo edge only.
**End-to-end joint training stays an explicit later option**, worth testing once the edge is proven
and the trunk-interference problem is understood — not before.

**Exposure-bias protocol** (the plan doc already anticipated this): train teacher-forced on REAL
tracks with conditioning augmentation (noise the track conditioning), then evaluate **both** with
real tracks and with generated tracks. The gap separates "does this information help" from "are our
tracks good enough yet" — a distinction today's failures repeatedly blurred.

## 8. Phase 5 — multi-species consolidation

One PDG-conditioned head covers all 17 classes today (γ 0.681, π 0.870, e± 0.888, p 0.749, rest
0.626) but costs ~0.12 of gate vs a dedicated head. Literature answer: **mixture-of-experts with a
gating network plus LoRA-style adapters** (Foundation Models for Calorimetry, 2026) — i.e. per-class
experts on a shared backbone. Do this last: it is an optimisation, and it is cheaper to tune once the
architecture is settled.

## 9. Evaluation roadmap

- **Now**: `event_gate_auc_width` (10 features) is the headline; the 8-feature gate is retained only
  for comparability with pre-2026-08-13 numbers. Per-feature AUCs on every run. Mechanism checks.
- **Phase 4 unlocks metrics no calo-only model can produce**: per-particle **E/p**, track–cluster
  ΔR matching, and particle-flow-style reconstruction on fully generated events. In the literature
  E/p appears as a *validation* observable, never as something a generator is built to produce
  jointly — this is the concrete novelty claim.
- **Reco-level** is where the field has moved (the Nov 2025 full-physics benchmark integrates a calo
  surrogate into a real detector with *real* tracks, via DDML, scored on di-photon separation and
  τ decays). We would be generating both sides. Biggest lift; keep as the endpoint.
- **Prior-art sweep before any writeup** — the novelty claim above rests on three web searches and
  needs a proper check.

## 10. Risks and kill criteria

| risk | mitigation / kill criterion |
|---|---|
| Helix explains little of the core (esp. e±) | Phase 0b is the gate — branch to the brem/track hypothesis rather than building on it |
| Track conditioning compounds tracker error into the calo | Phase 1 avoids it entirely; Phase 4 measures it (real vs generated tracks). Kill the edge if generated-track conditioning is worse than the truth-helix anchor |
| Joint training degrades the tracker deliverable | Not doing it; directional conditioning instead |
| Trunk interference keeps confounding results | Phase 0c, before anything else |
| Optimising a blind metric | The width gate exposed that the photon 0.557 was flattering (really 0.805). Keep adding features that physics says matter, and re-baseline when we do |

## 11. Ordering

```
0a calo geometry ─┐
0b helix-vs-core ─┼─> DECISION GATE ─> 1 helix anchor ─┐
0c split trunks  ─┘                                    ├─> 4 track-conditioned calo ─> reco-level
                    2 ShowerFlow (independent) ────────┤
                    3 incidence head ──────────────────┘
                                        5 multi-species MoE (last)
```
