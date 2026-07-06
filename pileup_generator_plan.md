# Pileup detector-response generator — implementation plan

> **Branch `flow-response`.** New strategy, started clean from the diffusion-era
> initial commit. The v1–v7 iteration line (tokenized AR tracker + calo
> set-prediction, best eval v5_tokenized: trk exact 71.7%, calo exact 39.4%,
> gen mean-r 356.7mm vs true 375.8) is preserved on branch `v7-caloset` and is
> not touched here. The tokenized AR **tracker head** from that line is the one
> component we intend to carry over — cherry-pick it from `v7-caloset` at M1
> rather than reimplement.
>
> **Two scope decisions taken at kickoff (differ from the original draft below):**
> 1. **M4 reco-level validation is DEFERRED.** Running ACTS reco on *generated*
>    hits is a large, separate effort we are not starting with. Until then the
>    acceptance gate is: per-species marginals (M1/M2) + a **classifier-on-hits
>    two-sample test** (train a classifier to separate Geant hits from generated
>    hits; target AUC → 0.5; report where it discriminates). Reco-level fidelity
>    remains the eventual goal, not the near-term gate.
> 2. **Data source is the pre-simulated ColliderML pu0 parquet**, not new Geant4
>    production. "Single-particle" training samples are approximated by slicing
>    *isolated* particles out of min-bias events via truth association
>    (`particle_id` / `contrib_particle_ids`) — the same association Stage 2
>    already uses. No new single-particle sim in v1.
>
> **De-risking order (do the cheap kills first):**
> - **Spike the calo flow-matching head first**, on isolated single particles,
>   and A/B it against the v5 tokenized calo BEFORE building trunk/superposition
>   pipeline around it. Continuous calo generation (voxel DDPM) already failed
>   once in v1–v4; the bet is that a CaloClouds-style per-point flow ≠ voxel
>   DDPM. If it can't beat v5/v6 calo on a controlled test, the premise is dead
>   and we stop cheap.
> - **Decay-daughter injection in preprocessing** is valuable regardless of which
>   calo approach wins — build it in parallel.

---

## Goal

A learned surrogate for the detector-simulation step of pileup production. Input: generator-level
particles from Pythia minimum-bias events (superposed to a chosen pileup level). Output: tracker
hits and calorimeter cell deposits, suitable for **hit-level overlay** onto hard-scatter events
before reconstruction. The hard-scatter itself, the particle prior, and reconstruction are all
out of scope — Pythia provides the input, standard reco consumes the output.

Design principle: factorize along the causal chain. The detector-response mechanism
p(hits | particles) is what we learn; it is invariant to pileup conditions. The pileup regime
enters only through the input particle cloud (superposition of K ~ Poisson(mu) min-bias vertices).

## Fixed scope decisions

- **Hit-level output** (overlay use case). No reconstructed-object shortcuts.
- **Generator-level input particles.** The model must absorb material effects (brem, conversions,
  nuclear interactions). **In-flight decays are handled by explicit daughter injection, NOT
  learned** — see below. Material effects remain learned and are the main residual modeling risk.
- **Decays handled outside the learned model (three-way split).** Rather than learning
  in-flight decay behavior inside the per-particle heads, decompose it:
  (a) *decay kinematics* (daughter four-momenta given a decay) — analytic / from the generator,
  not learned; (b) *decay point* — sampled analytically from the particle's lifetime ctau and
  momentum (betagamma), i.e. a survival distribution along the trajectory, optionally lightly
  learned but start analytic; (c) *detector response of the daughters* — the SAME per-particle
  response heads, applied to daughters tagged with their in-detector production vertex/radius.
  Net effect: a preprocessing step walks each unstable particle to its decay point and injects
  daughters into the input particle set; the heads only ever model response, never decay. This
  is the more modular DAG and it removes the immortal-pion failure mode by construction (survival
  is explicit, not hoped-for). Cost: deterministic bookkeeping in preprocessing that we control.
  Conversions and nuclear interactions are NOT cleanly separable this way (they are material
  effects, not two-body decays) and remain absorbed by the learned response — they stay in the
  M1 tail-validation risk.
- **No existence gate, no fakes model.** These are reconstructed-level concepts. At hit level,
  a particle leaves zero or more hits (emergent from the decoder), and efficiency / fake rates
  emerge downstream when reconstruction runs on generated hits. Electronic noise hits are added
  as a **parametric** (non-learned) process in v1.
- **Conditioning, v1:** per-particle features (four-momentum, PDG type, production vertex — vertex
  z especially) + one global scalar mu. Occupancy is otherwise implicit in the input cloud.
  Out-of-time pileup: deferred (schema must reserve a per-vertex bunch-crossing field, unused in v1).
- **In-time pileup only** for v1.

## Architecture

```
Pythia min-bias x K  ──superpose──>  particle cloud [N, F]
                                          │
                              shared trunk: permutation-equivariant
                              transformer encoder (no causal mask,
                              no positional encoding) -> h_i [N, D]
                                          │
                    ┌─────────────────────┴─────────────────────┐
             track-hit head                                calo head
      per-particle autoregressive                 per-particle flow-matching
      decoder over hit sequence                   point cloud (ShowerFlow-style
      ordered by radius; hybrid                   global structure + per-point
      discrete (layer/surface id) +               model); continuous (x,y,z,E);
      continuous (local coords, ToT/E);           project to cells afterwards
      stop token => emergent hit count
                    │                                           │
                    └────────────── scatter/merge ──────────────┘
              tracker: module-level cluster merging       calo: scatter_add
                                          │
                              + parametric noise hits
                                          │
                            event = {tracker hits, calo cells}
```

Representation rule and rationale: **tokenize where sequences are short and physically ordered
(tracker, ~10–20 hits along increasing radius); stay continuous where they are long and unordered
(calo showers, 10^2–10^3 deposits).** Calo energies span several orders of magnitude on a steeply
falling spectrum and reconstructed energies are sums over cells, so quantization error accumulates
into resolution degradation; a cell-ID vocabulary also couples the model to one geometry.
Continuous point cloud -> project-to-cells is the mature, benchmarked recipe (CaloClouds line).

Correlation between heads: v1 runs the heads **in parallel**, conditionally independent given the
trunk latents h_i (the shared trunk carries correlation implicitly). The autoregressive edge
(calo conditioned on generated tracks, for E/p and brem correlations) is a v2 experiment; if added,
train with conditioning augmentation (noise the track conditioning) to control exposure bias.

## Reference points (read before implementing)

- Tracker generation with GPT-style tokens: arXiv:2512.24254 (Open Data Detector; note their
  pion-decay failure mode — pions never decayed, inflating tracking efficiency).
- Calo point-cloud generation: CaloClouds3, arXiv:2511.01460; hadronic showers: CaloHadronic,
  arXiv:2506.21720. CaloChallenge 2022 summary for metrics conventions.
- Full-chain conditional set generation (multiplicity handling, permutation invariance):
  PIPPIN, arXiv:2406.13074.
- Hybrid continuous+discrete generation (kinematics + quantum numbers): multimodal flows,
  arXiv:2509.01736.

## Milestones

Staging principle: **learn the per-particle response first on single-particle / low-density data,
then superpose, then fine-tune for occupancy effects.** Detector response is nearly linear under
superposition (deposits add); the nonlinearity is occupancy (merged clusters, shared cells),
which is fine-tuned last, not learned first.

### M0 — Data pipeline and schema

- Simulation: **using pre-simulated ColliderML pu0 (Open Data Detector via ACTS)**; no new
  Geant4 production in v1. Isolated-particle samples are sliced from min-bias via truth
  association rather than sim'd standalone.
- Produce views: (a) isolated single particles per species (pi+-, K+-, p, n, e, gamma, mu),
  selected from pu0 by isolation cut; (b) full min-bias events; (c) a small overlaid hard-scatter
  set reserved for later. All with **truth association** hit -> parent particle.
- Schema: event-structured columnar files. Tables: particles [event, i, pdg, p4, vtx,
  bunch_dt(reserved)], tracker_hits [event, i_parent, surface_id, local_xy, energy], calo_deposits
  [event, i_parent(fractional), cell_id, xyz, energy]. Fix the fractional-energy convention for
  shared calo cells here.
- Preprocessing module: log-transform energies and pT, standardize coordinates; keep Jacobians
  explicit (matters later for density evaluation). Deterministic splits, seeded.
- **Decay preprocessing sub-step** (per the three-way split): for each unstable input particle,
  sample a decay point from its lifetime and momentum, decide whether it falls inside the
  tracking volume, and if so replace/augment the particle with its daughters tagged with
  in-detector production vertex/radius. Validate against truth: decay-radius and daughter-spectra
  distributions must match, per species, BEFORE response heads train on daughters.
- Acceptance: data QA notebook — per-species hit multiplicity distributions, energy spectra,
  truth-association coverage (>99% of deposits attributed), schema round-trip test.

### M1 — Track-hit head, single particles

- Per-particle autoregressive decoder conditioned on the particle's trunk embedding (start with
  direct features, introduce trunk in M3). **Port the tokenized AR tracker head from
  `v7-caloset`** as the starting implementation. Hit = (surface/layer token, continuous local
  coords, continuous energy/ToT); sequence ordered by radius; stop token terminates.
- Losses: cross-entropy on discrete tokens, likelihood (flow head) or regression + noise model
  on continuous features. Per-particle supervision via truth association — no set matching needed.
- Acceptance (per species, not aggregated): hit-multiplicity distributions **including tails**
  (tail risk here is material effects — nuclear interactions, conversions, punch-through — not
  decays, which M0 handles); residuals of hit positions vs truth per layer; fraction of particles
  with zero hits vs eta, pT. Cross-check that injected daughters get correct response.

### M2 — Calo head, single particles

- Flow-matching point-cloud generator per particle: global-structure flow (total E, layer energy
  fractions, point count) + per-point model, conditioned on particle features (and later trunk
  embedding). Continuous (x, y, z, E) output; separate deterministic projection onto cells.
- **This is the de-risking spike — do it first, A/B vs v5 tokenized calo on isolated particles.**
- Train and validate on **hadronic showers as first-class citizens**, not just photons/electrons —
  pileup deposits are mostly hadronic.
- Acceptance: CaloChallenge-style metrics — total energy response and resolution vs true E,
  layer-wise energy fractions, shower width/depth profiles, cell energy spectrum (check the
  low-energy tail vs threshold), point-count distributions. Per species.

### M3 — Event-level assembly and occupancy fine-tune

- Introduce the shared trunk over the full particle cloud; heads consume per-particle latents h_i.
- Superposition: scatter_add calo deposits into cells; tracker module-level merging rule for
  overlapping clusters on the same surface.
- Add parametric noise-hit process (rates from the sim's digitization config).
- Fine-tune trunk + heads on full min-bias events at several mu values (30 / 60 / 140 / 200) so
  crowded-region effects are seen in training. Global mu scalar conditioning enters here.
- Acceptance: occupancy vs eta/layer at each mu vs truth; cluster-size distributions in the
  densest regions; timing benchmark (events/sec, batched per-particle generation).

### M4 — Overlay and hit-level two-sample validation (near-term gate)

> Reco-level validation (ACTS on generated hits) is DEFERRED. Near-term gate is hit-level.

- Overlay generated pileup onto reserved hard-scatter events.
- Acceptance metrics, defined now:
  - **classifier two-sample test on full events at hit level** (train a classifier Geant-vs-
    generated; AUC ~ 0.5 is the target; report where it discriminates, not just the AUC);
  - hit-level occupancy / energy-sum / cluster-proxy distributions, generated vs Geant overlay.
- Deferred (future work, needs the ACTS chain): tracking efficiency / fake / duplicate rates,
  calo cluster multiplicity, jet energy scale/resolution, MET.

### M5 (optional) — Density estimation extension

- Track head: exact per-particle likelihoods already available (autoregressive).
- Calo head: density via the flow-matching probability-flow ODE; budget for the cost, or accept
  an ELBO-style bound. Joint log p(hits | particles) = sum of per-particle factor likelihoods
  under the v1 conditional-independence assumption — document that assumption wherever used.

## Engineering conventions

- Python, PyTorch (torch.compile where stable). Flow matching: hand-rolled or torchcfm.
- Config-driven experiments; every run reproducible from config + seed; wandb tracking.
- Batching: batch **per particle** for the heads, per event only for the trunk. Main throughput lever.
- Tests required before any training run: permutation-invariance unit test on the trunk;
  overfit-a-single-batch sanity test per head; schema golden-file tests; projection-step
  (points -> cells) exactness test.
- Keep the digitization/threshold step as an explicit, swappable module.

## Open questions

1. ~~Detector and sim framework~~ — RESOLVED: pre-simulated ColliderML pu0 (ODD + ACTS + Geant4).
2. **Tracker target: raw hits vs measurement/cluster level** (blocks M1): DECISION DEFERRED,
   data-dependent. Inspect what the pu0 tables actually provide (pre-digitization hits? cluster
   content? truth links? occupancy/cluster-size at target mu) in M0 data-QA, then record the
   choice here before M1.
3. **Fractional energy assignment for shared calo cells in truth** (blocks M0 schema): document
   the convention; carry the fraction in the schema.
4. **Pre-threshold deposits + explicit digitization vs post-threshold learning** (blocks M2):
   prefer pre-threshold + explicit digitization if the pu0 data exposes pre-threshold deposits;
   otherwise learn post-threshold and note it.
5. **Momentum spectrum for single-particle training** (blocks M0): sample from the min-bias
   spectrum (which the isolation-slice naturally gives), not flat.

## Known risks

- **Continuous calo regression to the v1–v4 failure.** Voxel DDPM plateaued at 0.53 and never
  separated subsystems. The bet is that a CaloClouds-style per-point flow is a different animal.
  Mitigation: the M2 spike A/B's against v5 tokenized calo on isolated particles before any
  pipeline is built around it.
- **Rare in-detector processes.** Decays removed from the learned model by daughter injection,
  leaving nuclear interactions, conversions, punch-through as the documented failure mode;
  mitigated by per-species tail validation in M1 and training-spectrum choice (OQ5). Fallback:
  extend the explicit-fate approach to a per-particle discrete "fate" variable
  (survive / interact / convert / punch-through) sampled before the heads.
- **Occupancy nonlinearity** at mu = 200 beyond trunk fine-tuning; mitigated by M3 densest-region
  cluster-size acceptance; fallback is local-neighborhood density conditioning per particle.
- **Exposure bias** largely designed out (Pythia input at train and generation time); re-enters
  only if the v2 track->calo edge is added — use conditioning augmentation there.
