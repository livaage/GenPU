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
- Produce views: (a) per-particle response slices per species (pi+-, K+-, p, n, e, gamma, mu)
  extracted by **truth association, NOT spatial isolation** — pu0 events are crowded (~862
  particles/event) so a spatial isolation cut discards almost everything, but `particle_id`
  (tracker) and `contrib_particle_ids` (calo) give clean per-particle slices anyway; (b) full
  min-bias events; (c) a small overlaid hard-scatter set reserved for later.
- Response is low-multiplicity-dominated (data-QA on pu0): only 30.6% of particles leave a
  tracker hit, 52% leave a calo deposit, 6.9% leave both. The heads MUST model "leaves nothing"
  as the common case (this is what the stop-token / emergent-count design is for).
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
- **M1 RESULT (charged pions, 1.62M tracks, tokenized AR on the shared contract):** ported and
  functional — trains healthily (teacher-forced val layer-CE 0.37), conditioning plumbing works
  (conditional hit-count tracks particle energy). BUT free-running generation marginals are rough:
  layer occupancy drifts to busiest layers (mean 18.9->21.0, W=2.66) and r/phi residuals narrow
  (std 0.80/0.88 vs 1.0). Diagnosis: (a) undertrained — loss still dropping at 40k (calo converged
  by 8k); (b) exposure bias — teacher-forced fine, free-running drifts. First fix (cheap): train
  longer (150k run launched). Deeper fix for occupancy drift / per-track coherence: scheduled
  sampling or the helix-residual plan below. Quality judged at M4 gate, not polished in isolation.
  NOTE: data layout is (M,5)=[layer_class,r,phi,z,time] (col0 already class), not the (M,6) I
  first assumed. No count head yet (eval conditions on truth hit-count).
- **TRACKER EVENT GATE (150k ckpt, pion, M4-lite):** AUC=0.997 — exposure-bias defects survive
  superposition. Discriminated by frac_inner 0.39->0.27 (under-populates inner layers),
  r_std 267->226 + layer_std 11.5->10.2 (residual/occupancy narrowing). Count/r_mean match
  (conditioned). So both heads now gated: CALO 0.813 (energy fixed), TRACKER 0.997 (needs a
  fix). Candidate fixes, cheapest first: sampling temperature >1 (broadens narrowed residuals);
  then scheduled sampling / the helix-residual physics anchor for the inner-layer occupancy drift.
- **TEMPERATURE SWEEP RESULT (ruled out the cheap fix):** cont_temp {1.0..1.6} x layer_temp
  {1.0,1.5} — AUC stuck at ~0.997 throughout. cont_temp does NOTHING to r_std (226->227):
  event-level r_std is set by WHICH layers are hit (barrel vs endcap radii), not within-layer
  residual. layer_temp helps only marginally. ROOT CAUSE isolated to ONE thing: the AR
  under-populates INNER layers in free-running (frac_inner 0.39->0.27), which simultaneously
  narrows r_std and layer_std. It is a directional DRIFT/BIAS (exposure bias), not an entropy
  deficit -> temperature cannot fix it. Real fix: scheduled sampling or the helix-residual
  physics anchor (pins layer/position to the analytic trajectory). A proper sub-project.
- **TRACKER FIX SOLVED (abspos + vertex-anchored seed): gate 0.997 -> 0.855.** Diagnosis chain:
  (1) per-layer-standardized residuals hide the trajectory (the AR autoregresses over near-
  independent targets); (2) the first/innermost hit was generated from an UNTRAINED BOS token
  (training never predicts hit_0; the BOS embedding row gets no gradient). Experiments:
  feeding absolute (r,phi,z) INPUT alone REGRESSED to 0.9997 (frac_inner 0.008) — position
  feedback amplified the bad first hit, PROVING first-hit is the root cause. Adding a
  vertex-anchored seed (prepend the production vertex as a trained position-0 token with its
  absolute (vr,vz), so hit_0 is predicted from a physical anchor and train==generate) FIXED it:
  AUC 0.855, every discriminating feature matched (frac_inner 0.39->0.42, layer_mean 18.6->18.2,
  r_std 267->278, layer_std 11.5->11.7 — all Δ/σ<0.3). No helix build needed. Remaining 0.855 is
  a softer multivariate residual (like the calo's 0.813). Behind use_vertex flag; abspos machinery
  in _embed_hits. Tracker now comparable to calo at event level.
- Acceptance (per species, not aggregated): hit-multiplicity distributions **including tails**
  (tail risk here is material effects — nuclear interactions, conversions, punch-through — not
  decays, which M0 handles); residuals of hit positions vs truth per layer; fraction of particles
  with zero hits vs eta, pT. Cross-check that injected daughters get correct response.

### M2 — Calo head, single particles

- Flow-matching point-cloud generator per particle: global-structure flow (total E, layer energy
  fractions, point count) + per-point model, conditioned on particle features (and later trunk
  embedding). Continuous (x, y, z, E) output; separate deterministic projection onto cells.
- **This is the de-risking spike — do it first, A/B vs v5 tokenized calo on isolated particles.**
- **SPIKE RESULT (2026-07-06, photon slice, CaloClouds-lite: Gaussian global head +
  per-point CFM, 188k params, 40k steps):** GO. Continuous flow matches shower marginals
  that v5 tokenized AR mean-collapsed on — d_eta/d_phi localisation W=0.010/0.009 (std matched
  <1%), radial profile near-perfect overlay, total shower logE W=0.105, points/shower mean
  4.43 vs 4.52. ONE expected failure: per-cell energy spectrum smears below the 50 keV floor
  (gen emits ~1e-9 GeV cells) — the predicted floor point-mass problem. FIX BEFORE PROCEEDING:
  model per-cell energy with an explicit floor (censored/clamped likelihood or at-floor
  Bernoulli), not unconstrained continuous fractions. Then repeat on hadrons (the real test).
  Code: scripts/build_calo_slice.py, src/genpu/flow/calo_flow.py, train/eval_calo_flow.py.
- **HADRON RESULT (charged pions, 2.5M showers, 15.6 pts/shower, spread 2-3x photons):**
  approach holds on the hard case. Localization matched (d_eta/d_phi std 1.06/1.18,
  W=0.03/0.06), width matched (1.77 vs 1.77), total-E W=0.13. Per-particle conditioning
  captures REAL physics with dynamic range: energy->N (6->26 across bins) and energy->width
  (2.58->0.95, more collimated) both tracked. Two residuals: (1) per-cell log-E ~0.36 low
  (both species); (2) coherent shower centroid offset under-predicted.
- **CHARGE EXPERIMENT:** adding charge to conditioning moved centroid offset 1.34->1.45
  (truth 1.73), W 0.385->0.279 — recovered ~28% of the gap. Conclusion: offset deficit is
  ~1/4 missing-input (now fixed), ~3/4 the i.i.d.-points independence limit. Closing the rest
  needs INTRA-SHOWER CORRELATION (set-transformer over points, or AR-within-shower), not more
  conditioning. Deferred: judge whether this residual matters at the M4 classifier gate before
  investing in it. M2 core (the flow-vs-tokenized bet) is validated and characterized.
- Train and validate on **hadronic showers as first-class citizens**, not just photons/electrons —
  pileup deposits are mostly hadronic.
- **EVENT-GATE LOOP RESULT (photon-only, M4-lite):** built a thin end-to-end gate (truth
  photons -> generate showers -> superpose per event -> torch-MLP two-sample test). First run
  AUC=0.994, driven ENTIRELY by the per-cell energy spectrum (frac_near_floor 0.035->0.27,
  logE_std 0.76->1.07). Fixed with a floor-mixture energy head (at-floor Bernoulli + Gaussian
  above floor). KEY LESSON: energy must condition ONLY on noise-free inputs. Conditioning it on
  cell position (v1) or the sampled global total_logE (v2) made it a sharp function of a quantity
  that is itself generated -> marginal energy blew up. Conditioning on cond (truth particle
  features) ONLY (v3) reproduces the marginal by construction. Result: AUC 0.994 -> 0.813, all
  energy features now Δ/σ<0.25. Remaining 0.81 is a softer multivariate residual (energy-tail
  shape + count), likely the i.i.d.-points structure — a separate, deeper fix.
- Acceptance: CaloChallenge-style metrics — total energy response and resolution vs true E,
  layer-wise energy fractions, shower width/depth profiles, cell energy spectrum (check the
  low-energy tail vs threshold), point-count distributions. Per species.

### M0 decay characterization (data finding — reshapes the decay-injection plan)

Ran scripts/analyze_decays.py on pu0 truth (3000 events, 2.6M particles):
- **91.6% of particles are secondaries** (have a parent / not primary). Only ~8% primaries.
- Secondary production radius spans the whole detector (median 366mm, out to 3.4m); only
  ~17% prompt (r<1mm), ~82% displaced (r>20mm).
- **The displaced daughters are DOMINATED BY MATERIAL EFFECTS, not two-body decays:** parents
  are gamma (223k -> e+e- conversions), e± (298k -> brem/delta rays), pi± (209k, 2.9 daughters
  incl. nuclei like deuteron/Si-28 -> nuclear interactions). Genuine analytic-decay species
  (K0S 5.5k, K0L 5.7k, K± 18k, Lambda) are a SMALL minority.

**Implication:** the clean "sample decay point from ctau/betagamma and inject daughters" scheme
only covers the genuine-decay minority. The dominant secondary production (conversions, brem,
nuclear interactions) is material-geometry-dependent and CANNOT be injected analytically — the
plan already flagged these as "not cleanly separable" but they turn out to be the MAJORITY, not
a tail. Also: decay/secondary injection is a GENERATION-TIME concern (Pythia-primaries-in). All
current training/gating uses TRUTH particles as input, which already contain every secondary at
its vertex — so the per-particle heads already handle secondaries and injection does not block
current work. When we do switch to Pythia input, material-effect secondary production is a large
open problem (learned cascade/multiplicity model, or response-model absorption), bigger than the
analytic decay sampler. Re-scope M0 decay accordingly.

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
2. ~~Tracker target: raw hits vs measurement/cluster level~~ — RESOLVED (M0 data-QA, 2026-07-06):
   **measurement-level**. pu0 tracker_hits carries measured `x,y,z` + pre-digitization
   `true_x,true_y,true_z` + `surface_id`/`layer_id`/`volume_id` + per-hit `particle_id`. No raw
   pre-digitization cluster content is exported. We generate measurement-level hits; truth
   positions are available for residual supervision.
3. ~~Fractional energy for shared calo cells~~ — RESOLVED: use fractional attribution; it's in the
   data as `contrib_energies` (per-cell list aligned to `contrib_particle_ids`). ~10.6% of calo
   cells have >=2 contributors (89.4% single); truth coverage effectively 100%.
4. ~~Pre-threshold vs post-threshold calo~~ — RESOLVED (M0 data-QA): **post-threshold only**.
   `total_energy` has a hard floor at 5.0e-5 GeV (50 keV) with a point mass pinned exactly at the
   floor; no pre-threshold deposits are exported. We learn post-threshold; the calo flow head must
   handle the floor point mass explicitly (see Known risks).
5. **Momentum spectrum for per-particle training** (blocks M0): the truth-association slice
   naturally reproduces the min-bias spectrum — use it as-is, do not reweight to flat.
6. **Do the pu0 particle lists include Geant secondaries?** (informs decay handling): particles
   carry `parent_id`, `primary`, and displaced `vx,vy,vz`, suggesting decay daughters ARE present
   in truth with production vertices. Confirm in M0: if so, training uses them directly and the
   decay-injection preprocessing is validated against them; injection is only needed at generation
   time (Pythia primaries in → daughters).

## Known risks

- **Continuous calo regression to the v1–v4 failure.** Voxel DDPM plateaued at 0.53 and never
  separated subsystems. The bet is that a CaloClouds-style per-point flow is a different animal.
  Mitigation: the M2 spike A/B's against v5 tokenized calo on isolated particles before any
  pipeline is built around it.
- **Calo energy floor / point mass.** Data-QA (2026-07-06): `total_energy` is post-threshold with
  a hard floor at 5.0e-5 GeV and a spike of deposits pinned exactly at the floor. A naive
  continuous flow will smear this point mass across the low-energy tail (which is where most
  deposits and the threshold interact). Handle explicitly: e.g. model log-energy above the floor
  as continuous + a separate "at-floor" Bernoulli, or a censored/clamped-energy likelihood.
  Validate the low-energy cell-spectrum tail against truth in M2 acceptance.
- **Rare in-detector processes.** Decays removed from the learned model by daughter injection,
  leaving nuclear interactions, conversions, punch-through as the documented failure mode;
  mitigated by per-species tail validation in M1 and training-spectrum choice (OQ5). Fallback:
  extend the explicit-fate approach to a per-particle discrete "fate" variable
  (survive / interact / convert / punch-through) sampled before the heads.
- **Occupancy nonlinearity** at mu = 200 beyond trunk fine-tuning; mitigated by M3 densest-region
  cluster-size acceptance; fallback is local-neighborhood density conditioning per particle.
- **Exposure bias** largely designed out (Pythia input at train and generation time); re-enters
  only if the v2 track->calo edge is added — use conditioning augmentation there.
