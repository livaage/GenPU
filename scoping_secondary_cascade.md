# Scoping: generation-time secondary/cascade production

Prompted by the M0 decay characterization, which found that the pu0 truth is
dominated by secondaries produced by **material effects**, not two-body decays.
This doc scopes what it takes to generate the full detector response from
**Pythia primaries** (the on-demand-generation goal), before we build anything.

## 1. The problem is unavoidable

The deliverable is `p(hits | Pythia primaries)`. Physically the hits come from the
primaries **plus their entire Geant cascade** (conversions, bremsstrahlung, delta
rays, nuclear interactions, decays). Measured shares of the response that come
from **secondaries** (particle produced at vr>20mm or flagged non-primary):

| detector | share of hits from secondaries |
|---|---|
| **Tracker** | **61.4%** of hits (83.9% of hit-leaving particles are secondaries) |
| **Calo** | **84.4%** of hits AND 84.4% of energy (90.4% of hit-leaving particles) |

Production-radius bands (share of each detector's response):

```
vr band (mm)     trk-hits  calo-hits  calo-E    (% of particles)
[ -1,    1)       37.1%     15.1%     15.1%      9.0%   <- prompt/primary
[  1,   20)        1.9%      0.7%      0.7%      0.6%
[ 20,  200)       26.1%      7.9%      7.9%     12.6%   <- inner-tracker material
[200,  600)       18.7%     42.0%     43.8%     43.4%   <- bulk shower development
[600, 1100)       15.4%     18.5%     17.4%     23.2%
[1100, inf)        0.8%     15.7%     15.1%     11.2%
```

So the cascade is **the majority of both detectors' response**, not a tail. You
cannot produce a faithful event from primaries without generating it. (Requiring
Geant secondaries as input would defeat the purpose — that IS Geant.)

## 2. What the cascade is (from data)

Dominated by **material effects**, spread throughout the detector volume:
- gamma -> e+e- (conversions): ~223k displaced parents
- e± -> e/gamma (brem, delta rays): ~298k
- pi± -> nuclear interactions (avg 2.9 daughters incl. nuclei, e.g. deuteron, Si-28): ~209k
- Genuine analytic-decay species (K0S, K0L, K±, Lambda): ~30k = **~4% of the cascade**.

So the analytic "sample decay point from ctau/betagamma" scheme covers ~4%. The
other ~96% is material-geometry-dependent and must be **learned or absorbed**.

## 3. Design options

**Conditioning-unit question:** what does the response head take as its unit?

### Option A — per-primary INCLUSIVE response
Head maps each PRIMARY -> all hits from it and its whole descendant cascade.
- Training: re-attribute every hit to its ROOT primary (walk parent_id), train
  primary -> all-descendant-hits.
- Generation: run head per Pythia primary. **No injection.** Cascade absorbed.
- Matches the plan's stated intent ("model must absorb material effects").
- Cost: per-primary output is a huge, structured multi-particle signature. For
  CALO this is standard (a shower = its cascade; CaloClouds does exactly this).
  For TRACKER it means generating multiple DISPLACED tracks (conversion/interaction
  products) from one neutral/charged primary — far harder than one track.
- Our current per-DIRECT-particle heads are the wrong granularity here; tracker
  would need a substantially new inclusive generator.

### Option B — explicit learned CASCADE generator + per-direct heads
A "cascade model" `primary -> set of secondary particles (kinematics + vertex)`,
trained on truth parent->descendant data; then the per-direct response heads we
already built run on each generated secondary.
- Reuses the calo-flow / tracker-AR heads (already gated). Modular.
- The cascade model is a conditional variable-cardinality set generator — a
  generalization of the original "Stage 1" particle generator, now producing the
  full Geant cascade (with material-dependent vertices), not just primaries.
- Cost: learning Geant particle production incl. where conversions/interactions
  happen (implicit material map). Large, variable cardinality.

### Option C — per-DETECTOR hybrid (recommended)
Split by detector because the two have very different cascade character:
- **Calo: absorb the cascade (Option A for calo).** Re-attribute calo hits to the
  calo-incident particle and train the calo flow as an inclusive shower generator
  (incoming particle -> full shower cloud). This is the mature CaloClouds recipe
  and REMOVES the calo cascade problem entirely — no calo secondary injection.
  It also SIMPLIFIES our current per-direct calo training (which over-splits the
  shower into per-secondary deposits and thereby manufactures the injection
  problem for calo).
- **Tracker: explicit secondary-track generation (the genuinely novel hard part).**
  61% of tracker hits are displaced secondaries forming real tracks at r=20-1100mm.
  Need a learned model: does a conversion/interaction occur, at what radius, and
  what secondary tracks result (multiplicity + kinematics) -> then the per-direct
  tracker head generates each. This is the crux of Pythia-input tracker generation.
- **Genuine decays (~4%): analytic injection** as a cheap, exact add-on (K0S/K0L/
  K±/Lambda via ctau/betagamma), independent of the learned material model.

## 4. Implications for the current programme

- **Nothing here blocks current work.** All training/gating uses TRUTH particles as
  input, which already contain every secondary at its vertex. The per-direct heads
  already handle secondaries; cascade generation is only needed when input switches
  to Pythia primaries (the real end-to-end goal, not yet reached).
- **Calo re-framing is a latent simplification.** Moving the calo head to
  per-incident-inclusive would both fix the (future) calo cascade problem AND
  remove the current per-direct over-granularity — worth doing when we touch calo
  again, but not urgent.
- **The tracker's material-secondary generation is the single biggest open problem**
  for Pythia-input generation, and it COMPOUNDS with the tracker head's existing
  exposure-bias issue (inner-layer occupancy drift). Both point at the same fix
  direction: physics-anchored tracker generation (helix residual) that knows where
  tracks start.
- **Analytic decay injection is de-prioritised** — real but small (~4%), and easy
  to add later.

## 5. Recommended sequencing

1. Continue with TRUTH-particle input through M3 (trunk + superposition + occupancy)
   — none of it needs cascade generation, and it exercises the full assembly.
2. Treat **cascade generation as the explicit gate to true Pythia-input generation**,
   scoped as Option C: calo-inclusive (easy, reuses CaloClouds recipe) + tracker
   secondary-track model (hard, the priority research piece) + analytic decays (cheap).
3. Do NOT build the cascade generator until the per-detector heads are solid and M3
   assembly works on truth input; prototype the tracker secondary-track model first
   when we do, since it is the crux and the highest-risk piece.

**One-line takeaway:** on-demand generation from Pythia input is gated by a
material-cascade generator that is 96% material effects and ~61-84% of the response;
the calo half is a known recipe, the tracker half is the real research problem, and
neither blocks the truth-input pipeline we should build next (M3).
