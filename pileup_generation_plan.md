# On-Demand Pileup Generation with Hierarchical Generative Models

## Project Summary

Build a generative ML model that produces realistic pileup detector responses (tracker hits + calorimeter deposits) on-the-fly, eliminating the need to store pre-mixed pileup samples on disk. The model should accept a user-specified number of pileup interactions ⟨μ⟩ and generate the corresponding detector-level output for overlay onto hard-scatter events.

## Motivation

At the HL-LHC, CMS needs ~200 simultaneous minimum-bias interactions overlaid per hard-scatter event. Currently these are pre-simulated and stored as "premixed" samples, creating a major storage burden — CMS projects a factor-of-7 shortfall in disk budget. Generating pileup on-demand replaces stored premixed events with a compact generative model, dramatically reducing storage while maintaining physics fidelity.

This is a largely unexplored research direction. The only prior work (Arjona Martínez et al., 2019, arXiv:1912.02748) demonstrated a proof-of-concept GAN generating particle-level four-momenta for pileup, but nobody has pushed to detector-hit-level generation or production integration.

Latency target: O(1–10 seconds) per event. This is not competing with GEANT4 on speed — it replaces disk I/O of premixed samples.

## Dataset

**CERN/ColliderML-Release-1** on HuggingFace:
- URL: https://huggingface.co/datasets/CERN/ColliderML-Release-1
- Format: Apache Parquet with list columns
- License: CC-BY-4.0
- Detector: Open Data Detector (ODD) — generic HL-LHC-style, not CMS-specific
- Simulation chain: MadGraph5 + Pythia8 → Geant4 via DD4hep → ACTS reconstruction

### Key Subset for Training

**`pileup_only_pu0`** — single soft QCD (minimum-bias) interactions simulated through the full detector. This is clean, uncontaminated pileup data with no hard-scatter process. Available data tables:

- `pileup_only_pu0_particles` — truth-level particle information (pdg_id, px, py, pz, energy, vertex info, etc.)
- `pileup_only_pu0_tracker_hits` — digitised tracker hits (x, y, z, time, volume_id, layer_id, surface_id, particle_id)
- `pileup_only_pu0_calo_hits` — calorimeter energy deposits (x, y, z, total_energy, detector, contrib_particle_ids, contrib_energies, contrib_times)
- `pileup_only_pu0_tracks` — ACTS-reconstructed tracks (d0, z0, phi, theta, qop, hit_ids)

Each event represents one minimum-bias interaction with its complete detector response. This is exactly the unit the hierarchical model should learn to generate.

### Validation Subsets

Use the pu200 overlaid samples to validate that superposed generated pileup matches genuine full-simulation pileup:
- `ttbar_pu200_*` — busy hadronic events, tests pileup in presence of hard activity
- `zmumu_pu200_*` — clean events where pileup dominates the calorimeter
- `zee_pu200_*`, `dihiggs_pu200_*`, `ggf_pu200_*` — additional physics processes for process-independence checks

The pu0 versions of these processes provide the "clean" hard-scatter baseline for overlay tests.

### Data Schema Reference

#### Particles (truth)
| Field | Type | Description |
|-------|------|-------------|
| event_id | uint32 | Event identifier |
| particle_id | list\<uint64\> | Unique particle ID within event |
| pdg_id | list\<int64\> | PDG code (211=pion, 11=electron, etc.) |
| px, py, pz | list\<float32\> | Momentum components (GeV/c) |
| energy | list\<float32\> | Total energy (GeV) |
| vx, vy, vz | list\<float32\> | Production vertex position (mm) |
| primary | list\<bool\> | Whether particle is primary |
| vertex_primary | list\<uint16\> | Primary vertex index (1=hard scatter) |
| num_tracker_hits | list\<uint16\> | Number of tracker hits |
| num_calo_hits | list\<uint16\> | Number of calorimeter hits |

#### Tracker Hits (detector-level)
| Field | Type | Description |
|-------|------|-------------|
| event_id | uint32 | Event identifier |
| x, y, z | list\<float32\> | Measured hit position (mm) |
| true_x, true_y, true_z | list\<float32\> | True position before digitisation (mm) |
| time | list\<float32\> | Hit time (ns) |
| particle_id | list\<uint64\> | Truth particle that created hit |
| volume_id | list\<uint8\> | Detector volume |
| layer_id | list\<uint16\> | Layer number |
| surface_id | list\<uint32\> | Sensor surface ID |

#### Calorimeter Hits (detector-level)
| Field | Type | Description |
|-------|------|-------------|
| event_id | uint32 | Event identifier |
| detector | list\<uint8\> | Calorimeter subsystem code |
| total_energy | list\<float32\> | Total energy in cell (GeV) |
| x, y, z | list\<float32\> | Cell centre position (mm) |
| contrib_particle_ids | list\<list\<uint64\>\> | Particles contributing to cell |
| contrib_energies | list\<list\<float32\>\> | Energy from each particle (GeV) |
| contrib_times | list\<list\<float32\>\> | Time of each contribution (ns) |

## Architecture: Hierarchical Pileup Generator

### Design Philosophy

The model mirrors the physical process: pileup consists of N independent minimum-bias pp interactions, each producing particles that leave detector signatures. The architecture factorises accordingly:

1. **Interaction-level**: Generate latent representations of particles from a single min-bias interaction
2. **Detector-level**: Decode each latent particle into tracker hits and calorimeter deposits jointly
3. **Superposition**: Combine N independently generated interactions with deterministic merging

### Why Hierarchical

- **Trivial pileup conditioning**: To generate ⟨μ⟩ = 50, 100, 200, etc., simply draw N ~ Poisson(⟨μ⟩) and invoke the single-interaction generator N times. No retraining needed for different pileup levels.
- **Joint tracker+calorimeter**: Both modalities are generated from the same latent particle representation, preserving physical correlations (same particle produces both tracker hits and calo deposits).
- **Physically motivated**: Exploits the independence of minimum-bias interactions rather than learning it from data.

### Model Components

#### Stage 1: Particle Set Generator
- **Input**: Random noise (+ optional conditioning on interaction vertex position)
- **Output**: Variable-size set of latent particle tokens, each encoding approximate (pT, η, φ, particle_type)
- **Architecture**: Set-based diffusion model or VAE that generates variable-cardinality particle sets
- **Training data**: Truth particles from `pileup_only_pu0_particles`, using kinematic features of primary particles

#### Stage 2: Joint Detector Response Decoder
- **Input**: Single latent particle token from Stage 1
- **Output**: (a) set of tracker hits on appropriate layers, (b) calorimeter energy deposit(s)
- **Architecture**: Two decoder heads sharing a common particle embedding
  - Tracker head: predicts hit positions per layer, conditioned on particle kinematics — naturally sparse (soft particles hit only a few layers)
  - Calorimeter head: predicts energy deposit and cell position, conditioned on particle kinematics — typically one or a few cells for soft pileup particles
- **Training data**: Use `contrib_particle_ids` in calo_hits and `particle_id` in tracker_hits to associate each truth particle with its detector response

#### Stage 3: Deterministic Superposition
- **Not learned** — this is a post-processing step
- Combine outputs from N single-interaction generations
- Calorimeter: sum energies in same cells, apply zero-suppression thresholds
- Tracker: merge hits on same sensor surfaces into clusters
- Overlay onto user-provided pu0 hard-scatter event using same merging rules

### Training Strategy

1. From `pileup_only_pu0` events, build per-particle training examples:
   - For each truth particle, collect its associated tracker hits (via `particle_id`) and calorimeter deposits (via `contrib_particle_ids`)
   - This gives (particle_kinematics) → (tracker_hits, calo_deposits) pairs for supervised training of Stage 2
2. Train Stage 2 (decoder) first on these per-particle examples
3. Train Stage 1 (particle set generator) on the per-event particle distributions
4. Validate end-to-end by generating full pileup events and comparing against pu200 samples

## Validation Plan

### Per-Interaction Validation (against pileup_only_pu0 held-out data)
- Particle multiplicity distribution
- pT, η, φ spectra of generated particles
- Tracker hit count per event and per layer
- Calorimeter energy sum and spatial distribution

### Aggregate Pileup Validation (against pu200 samples)
Generate N~Poisson(200) interactions, superpose, overlay onto pu0 hard-scatter events. Compare against genuine pu200 events:
- **ρ** (median pT density per unit area in η-φ) — the key pileup observable
- Calorimeter occupancy maps in (η, φ)
- Tracker hit multiplicity per layer
- Total event energy distribution
- Number of reconstructed vertices (if running ACTS reconstruction on generated events)

### Process-Independence Check
Verify that pileup generated independently produces consistent results when overlaid onto different hard-scatter processes (ttbar, zmumu, dihiggs, etc.). The pileup component should be statistically indistinguishable across processes.

## Project Phases

### Phase 0: Data Exploration
- Load `pileup_only_pu0` subsets (particles, tracker_hits, calo_hits)
- Characterise single min-bias interactions: particle multiplicities, pT spectra, hit counts, energy distributions
- Understand the detector geometry from the data: how many layers, cell structure, detector subsystem codes
- Build the particle→hits mapping using `particle_id` and `contrib_particle_ids`
- Visualise example events in (η, φ) and (x, y, z) coordinates

### Phase 1: Per-Particle Detector Response Model (Stage 2)
- Build training dataset of (particle_kinematics) → (tracker_hits, calo_deposits) pairs
- Implement and train the joint decoder
- Validate per-particle: given truth particle kinematics, does the model produce realistic hit patterns?

### Phase 2: Particle Set Generator (Stage 1)
- Train the set-level generative model on per-event particle distributions
- Validate: do generated particle sets have correct multiplicity, kinematic distributions?

### Phase 3: End-to-End Pileup Generation
- Chain Stage 1 → Stage 2 → superposition
- Generate pileup at various ⟨μ⟩ values (50, 100, 140, 200)
- Overlay onto pu0 hard-scatter events
- Compare against genuine pu200 samples from ColliderML
- Measure generation time and model size

## Key References

- Arjona Martínez et al. (2019) — "Particle GANs for full-event simulation and pileup description" (arXiv:1912.02748) — only prior work on ML pileup generation
- CaloChallenge / Krause et al. (2024) — community benchmark for calorimeter simulation, defines evaluation methodology
- CaloDiT-2 / LEMURS (Zaborowska et al., 2025) — multi-detector pre-trained calorimeter simulation, shows foundation model approach for fast sim
- CaloScore v2 (Mikuni & Nachman, 2022) — diffusion models for calorimeter showers
- OmniJet-α (Birk, Hallin, Kasieczka, 2024) — tokenisation + autoregressive generation for particle physics data
- CMS Phase-2 Computing Model Update (CDS 2815292) — documents the computing/storage challenge motivating this work
- ColliderML dataset documentation — https://huggingface.co/datasets/CERN/ColliderML-Release-1

## Technical Notes

- Python-based development (PyTorch)
- Dataset is Parquet format, loadable via HuggingFace `datasets` library or `pyarrow` directly
- The ODD detector is generic, not CMS — results are proof-of-concept, with CMS-specific follow-up as future work
- The `pileup_only_pu0` samples are individual min-bias events (no pileup overlay), each representing one soft QCD interaction through the full detector chain
