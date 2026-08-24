# PIPELINE — what the code ACTUALLY does

_Derived by reading the code, not the plan. Last verified 2026-08-24 against commit `6b18b19`._

**Why this file exists.** On 2026-08-24 a single session found four wrong recorded assumptions, two
contaminated results, and a fundamental design divergence — every one of them a gap between what a
document said and what the code did. `STATUS.md` records *what we tried*; `pileup_generator_plan.md`
records *what we intended*. Neither records *what runs*. This does.

**Rule: if this file and any other document disagree, this file is the one to re-verify first, and
whichever is wrong gets fixed the same day.** Every claim below should be checkable against a named
file and line.

---

## 0. One-line summary

Truth particles in → per-particle detector response out. **No Stage-1 particle generator**: every
result to date conditions on real (Geant) particles, including all secondaries at their true
vertices. Two independent heads: tracker (tokenised autoregressive) and calo (flow-matching point
cloud). They do not talk to each other.

---

## 1. Raw source — CERN/ColliderML-Release-1

`/scratch/gpfs/IOJALVO/lv7805/genpu_cache/datasets/CERN___collider_ml-release-1/`
Per-EVENT list columns, IPC stream format. Load with `genpu.data.load_shard(subset, i)`.

| subset | columns | on disk? |
|---|---|---|
| `particles` | event_id, particle_id, pdg_id, mass, energy, charge, vx, vy, vz, time, px, py, pz, perigee_d0, perigee_z0, vertex_primary, parent_id, primary | yes |
| `tracker_hits` | event_id, x, y, z, true_x/y/z, time, particle_id, detector, volume_id, layer_id, surface_id | yes |
| `calo_hits` | event_id, detector, total_energy, **x, y, z**, contrib_particle_ids, contrib_energies, contrib_times | yes |
| `tracks` | — | **NOT DOWNLOADED** (reco tracks; the deferred ACTS validation needs it) |

**Joins**: by `event_id`, never row index. The subsets are NOT in the same row order — measured
agreement 0.000. `genpu.data.build_event_index` is the correct tool.

**Not in the source**: the Geant4 **creation process** (conversion / brem / hadronic / decay). Any
cascade model must infer it from (parent pdg, daughter pdg set).

---

## 2. `preprocessing.py` — the input boundary, and where information is lost

Per event, attributes hits to particles by `particle_id` (`searchsorted`, not a positional join).

### What it keeps

| output | contents |
|---|---|
| `particle_features` (N,6) | log_pt, eta, phi, **pdg_class (17-way collapse)**, charge, mass |
| `particle_aux` (N,6) | primary, parent_id, vx, vy, vz, energy |
| `tracker_hits_flat` (M,5) | **layer_class (0..47)**, r, phi, z, time |
| `calo_hits_flat` (M,5) | eta, phi, log(contrib_energy), contrib_frac, detector |
| `tracker_offsets`, `calo_offsets`, `event_ids`, `particle_ids` | CSR layout + keys |

### What it DROPS — read this before assuming a quantity exists

- **calo depth.** `(x,y,z)` → `(eta,phi)`; the radial/longitudinal coordinate is discarded at
  line ~212. `detector` (6 values) is the only surviving depth proxy. **This is why the calo model
  is 2D** — see §5.
- **tracker `surface_id`** and `true_x/y/z`. `(volume,layer)` is collapsed to 48 `layer_class`
  values, so module identity is gone (the v3 surface-local work had to rebuild it separately via
  `module_geometry.py`).
- **particles with no hits.** `visible_mask = (n_tracker_hits > 0) | (n_calo_hits > 0)`. Drops
  **24.2%** of particles and **2,586 of 10,000 events** entirely. Consequence: the parent→child
  graph CANNOT be rebuilt from stage2 — 16.1% of visible secondaries have an invisible direct
  parent and 48.6% of chains to a primary cross an invisible node. Use the cascade graph (§3).
- **pdg identity beyond 17 classes.** Everything unusual becomes class 16 — 6% of particles,
  9,561 distinct masses up to 27.9 GeV (nuclei). `mass` is the only surviving discriminator there.

### Units of the stored calo record

`calo_hits_flat` rows are **(cell, particle) CONTRIBUTION pairs, not cells.** A cell with 3
contributors appears 3 times, each carrying that particle's share. Mean 1.206 contributors per cell;
89.3% of cells have one. This is why nine of the ten calo gate features are partition-dependent (§6).

---

## 3. Derived artifacts

| artifact | built by | contents |
|---|---|---|
| `shard_XXXX_stage2.npz` (4 shards: **0,1,2,5**) | `preprocessing.py` | above |
| `shard_XXXX_graph.npz` | `build_cascade_graph.py` | **every RAW particle** (visible or not): 17 source columns + n_tracker_hits, n_calo_hits, calo_energy_sum, r_innermost, r_outermost, depth, root_primary_id, visible |
| `shard_XXXX_pids.npz` | same | join key, stage2 row order |
| `tracker_slice/*.npz` | `build_tracker_slice.py` | cont (S,7), pdg, hits (P,5), offsets |
| `calo_slice/*.npz` | `build_calo_slice.py` | cont (S,7), glob (S,4), points_flat (P,3), offsets, anchor, anchor_mode |

**Shards are 0, 1, 2, 5** — 3 and 4 were never preprocessed.

---

## 4. Slice builders — every one filters to n >= 1

- `build_tracker_slice.py` — `--min_hits` default 1. **Sorts hits r-ascending here**, NOT in
  preprocessing (stage2 order is arbitrary: Spearman(index, r) = 0.01).
- `build_calo_slice.py` — `>= 1` calo deposit, one pdg class. Groups by **direct depositing
  particle**, so 64.8% of "showers" are fragments born inside the calorimeter (§6).
- `build_count_slice_stage2.py` — `keep = nh >= 1`. Computes `d0` **geometrically**
  (`vx*sin(phi) - vy*cos(phi)`) because the source `perigee_d0` is non-finite for 56% of charged
  particles.

Shared conditioning: `CONT_FEATURES = [log_pt, eta, log_E, charge, mass, vr, vz]`
(`genpu/conditioning.py`). phi excluded — response is phi-symmetric.

---

## 5. Models — what they actually generate

| model | file | generates |
|---|---|---|
| Tracker v1/v3 | `models/tracker_ar.py` | layer_class + standardised (r,phi,z,time) residual bins, inner→outer |
| Tracker v2 | `models/tracker_state_ar.py` | + direction state, STOP head |
| Count head | `models/count_head.py` | **categorical over 1..48** — structurally cannot emit 0 |
| Calo flow | `flow/calo_flow.py` | GlobalHead (total_logE, log_n, core_eta, core_phi) + PointFlow + EnergyHead |
| Incidence | `train_incidence_head.py` | P(tracker trace), P(calo trace) — shared trunk, 2 Bernoullis |

**The calo model is 2D.** `points_flat` is (P,3) = `(d_eta, d_phi, logE)`; `PointFlow` runs CFM on
`pts[:, :2]`; `sample_showers` returns `pos (P,2)` + `logE`. It generates no depth and no
`detector`.

---

## 6. Metrics — what they measure, and what they cannot

`calo_metrics.py` event gate: pools an event's cells and computes 10 features. Partition-invariance:

| feature | invariant? |
|---|---|
| `log_totE` | **yes** |
| `n_cells`, `logE_mean/std/max/p90`, `frac_near_floor`, `cells_per_src`, `width_mean/std` | **no** — computed over contributions, or per-shower |

So any change to how particles are partitioned moves 9 of 10 features regardless of model quality.

---

## 7. KNOWN GAPS — specified but NOT built

1. **Calo depth / projection-to-cells.** Plan (`pileup_generator_plan.md:257`) specifies
   "Continuous (x, y, z, E) output; separate deterministic projection onto cells". Neither exists.
   Consequence: the plan's own acceptance metrics — layer-wise energy fractions, shower depth
   profiles (line 356) — **have never been computable**, and every calo result is a 2D projection.
2. **Cell-level metric.** Blocked on (1).
3. **Tracker incidence** — the head exists (§5) but is not wired into any generation path.
4. **Stage-1 / cascade generator.** Architecture settled (one recursive model on
   `[log_E, eta, vr, vz, pdg]`, ~3-4 batched levels) but not built.
5. **M3 event assembly** — superposition, noise process, mu conditioning. Not started.
6. **M4 reco-level (ACTS)** — deferred; also needs the `tracks` subset, not downloaded.

---

## 8. Maintaining this file

Update it in the same commit as any change to a data contract, a model output, or a metric
definition. When a docstring and the code disagree, fix the docstring and note it here — several
were stale as of 2026-08-24 (`build_calo_slice.py` documents `cond (S,4)` / `glob (S,2)`; the real
shapes are `cont (S,7)` / `glob (S,4)`).
