# 2026-08-24 — cascade depth is SHALLOW (3 levels = 96.5% of tracker hits); primaries-in is batchable, and the speed case for it is weaker than the architecture case

- **Commit / branch**: `99879a3` / `flow-response` (working tree)
- **Job**: none — login-node probes on raw source (200 events) + stage2.
- Extends [scoping_secondary_cascade.md](../scoping_secondary_cascade.md) (Jul 8), whose spikes were
  recovered into STATUS on 2026-08-24 after being absent from the rolling state.

## Hypothesis

Decision question: for true Pythia-primaries-in generation, learn the material cascade, or call an
existing propagator (ACTS **Fatras** — available for this detector, since ColliderML is ODD via
ACTS)? The concern against Fatras was per-step cost. The concern against learning it is that a
cascade is recursive, so it might serialise just as badly.

## Result

**(1) The cascade is SHALLOW.** Parent-chain depth from each particle to its primary root, weighted
by tracker hits (200 events, 204,312 particles):

| depth | % particles | % tracker hits | cumulative hits |
|---|---|---|---|
| 0 (primary) | 26.9% | 40.5% | 40.5% |
| 1 | 28.4% | 35.7% | **76.1%** |
| 2 | 25.3% | 13.7% | **89.9%** |
| 3 | 11.7% | 6.6% | **96.5%** |
| 4 | 5.3% | 2.5% | 99.0% |
| >=5 | 2.4% | 1.0% | 100% |

Mean **3.3 particles per primary**. So a learned cascade needs ~3-4 recursive levels, and **each
level is fully batchable** (every particle at depth k generated in one call, conditioned on its
parent) — sequential depth ~4 regardless of event size. That is the throughput argument for
learning it, and it holds.

**(2) But the speed case for learning the TRACKER cascade is weak, because that is not where
Geant4's cost is.** Geant4's expense is dominated by calorimeter shower development (multiplicity
explosion). Our own data shows it indirectly: high-E photon showers have NO daughter particles in
the record because the thousands of shower particles are not kept — only hits. **The ML calo head
already replaces the expensive part.** The tracker material cascade is the cheap part (3.3
particles/primary, depth 4). Fatras is also not "a slightly faster Geant4": simplified surface-based
geometry, material effects sampled from parameterized distributions at surface crossings, and it
does not do calo showers at all.

**=> The case for primaries-in is ARCHITECTURAL** (one self-contained generator, no external
geometry dependency, a claim that stands alone), **not primarily throughput.** Worth knowing before
it goes in a paper.

**(3) Two conditioning gaps in the existing spikes.** `build_conversion_slice.py:54` conditions on
`[log_E, eta, vr, vz]`:
- **No phi.** The material map is learned phi-averaged. ODD is idealised so this may be near-exact,
  but it is untested — a one-line check of conversion radius vs phi settles it.
- **Charged particles bend.** For a photon, (start, direction) fixes the path and hence the material
  integral. For a pion the path is a helix, so material traversed depends on pT and charge; the
  nuclear spike conditions the same way, which is a cruder proxy there.

**(4) Both spikes only validated ONE level.** Photon -> direct e± daughters; pion -> direct displaced
daughters. Depth 1 is 76% of hits so that was the right first proof, but depth 2-4 (the remaining
23%) and **error compounding across levels** are untested. Compounding is the same failure mode the
tracker head already has (v3 needed scheduled sampling); the scoping doc predicted this
("COMPOUNDS with the tracker head's existing exposure-bias issue").

**(5) PRIMARY-FRACTION DISCREPANCY — RESOLVED 2026-08-24, and M0's number was wrong.**
This probe gets primaries at **26.9%** (`primary` flag); M0 (`analyze_decays.py`) reported **8.4%**
(91.6% secondaries) via "has a parent". Direct check on 227,568 raw particles:

| test | fraction |
|---|---|
| `primary` flag TRUE | **0.267** |
| `parent_id <= 0` (no parent) | **0.000** |
| `vertex_primary` TRUE | 1.000 |

**Every particle carries a `parent_id`** (range 0..6295), so "has a parent" is TRUE for ~100% of
particles and is NOT a valid secondary test. The cross-tab is degenerate: primary=True/has_parent=True
0.267, primary=False/has_parent=True 0.733, and both no-parent cells are 0.000. M0's 91.6% therefore
does not measure what it says it measures.

**Correct number: ~27% primaries.** Pythia would hand us ~27% of the particle list and the cascade
must generate the other ~73% — a **3x SMALLER** job than the M0 framing implied. Consistent with the
measured 3.3 particles per primary. Any statement anywhere in the repo resting on "91.6% of
particles are secondaries" should be re-read (it appears in `pileup_generator_plan.md:363` and was
repeated into STATUS).

## Verdict — KEPT (measurement). Direction chosen: primaries-in, learned cascade.

User decision 2026-08-24: pursue primaries-in with a learned cascade rather than calling Fatras,
on the architectural argument. Fatras is NOT discarded — it remains available as a **validation
target** for the learned cascade, which is cheaper than committing to either.

## Next

- **Prerequisite for everything**: `preprocessing.py` must store `particle_ids` (it computes them at
  line 233 and never writes them), else the parent->child graph cannot be rebuilt downstream. This
  one change unblocks the cascade work, the calo re-attribution, and the incidence head's strongest
  feature. Requires a preprocessing re-run.
- Resolve the primary-fraction discrepancy (26.9% vs 8.4%) — it is a 3x on the cascade target.
- Extend the conversion/nuclear spikes from depth 1 to depth 3-4 and **measure compounding**.
- Check conversion radius vs phi to test the phi-symmetry assumption.
- Sequencing note: `scoping_secondary_cascade.md` recommends not building the cascade generator until
  M3 assembly works on truth input. M3 (event assembly, superposition, occupancy fine-tune at
  mu=30/60/140/200) has not been started. Prioritising cascade now reorders that deliberately.
