# GenPU — project guide for Claude

Full-event pileup **generator** for the ColliderML detector (tracker + calorimeter),
trained on single min-bias (pileup-only, PU0) events. Two-stage: particle-level
generation → per-particle detector hits (tracker AR head + calo flow-matching head).

## Environment

Micromamba env **`genpu2`** (the old `genpu` env is broken — torch import fails; do not use it).

```bash
export MAMBA_ROOT_PREFIX=/scratch/gpfs/IOJALVO/gnn-tracking/object_condensation/micromamba
export MAMBA_EXE=/home/lv7805/bin/micromamba
export HF_HOME=/scratch/gpfs/IOJALVO/lv7805/genpu_cache            # HF library cache root
export GENPU_DATA_DIR=/scratch/gpfs/IOJALVO/lv7805/genpu_cache/datasets  # what the loader globs

# run anything in the env:
$MAMBA_EXE run -n genpu2 python <script>
```

- Python 3.12, torch 2.5.1 (cuda build 12.4), pyarrow 24.
- Package is a **src layout**: `import genpu...` needs `sys.path.insert(0, "src")` (or install -e).
- **No GPU on the login node** (`torch.cuda.is_available()` is False there). Run training/eval/
  sweeps as SLURM jobs — use the **submit-to-grid** skill. Keep only light schema peeks / small
  slices on the login node.
- Home has a tight quota — all caches, data, checkpoints live on `/scratch/gpfs/IOJALVO/lv7805/`.

## Dataset — CERN/ColliderML-Release-1 (HuggingFace)

### Raw source (HF arrow shards, per-EVENT list columns)
`/scratch/gpfs/IOJALVO/lv7805/genpu_cache/datasets/CERN___collider_ml-release-1/`
- 1000 shards × ~7k events. Load via `genpu.data.load_shard(subset, i)` (IPC stream format —
  **not** `pyarrow.dataset`, which errors with "Not an Arrow file").
- Subsets (`PU0_SUBSETS` in [src/genpu/data.py](src/genpu/data.py)): `particles`,
  `tracker_hits`, `calo_hits`, `tracks`. Join source tables **by `event_id`, never row index**.
- **`tracker_hits` schema** (list<> per event): `event_id, x, y, z, true_x/y/z, time,
  particle_id, detector, volume_id, layer_id, surface_id`.
  - `x,y,z` = measured global position (mm); `true_x/y/z` = pre-smearing truth (smearing is tiny,
    ratio ~0.98 — see findings doc). No per-hit momentum exists in the source.
  - Discrete geometry: `volume_id` (9) → `layer_id` → `surface_id` (~3.4k raw values, **reused
    across layers** — the module key is the `(volume,layer,surface)` composite, ~18k modules).
- `particles`: kinematics + `vx,vy,vz` (production vertex), `energy`, `primary` flag, `parent_id`.

### Preprocessed stage2 (training-ready npz)
`/scratch/gpfs/IOJALVO/lv7805/genpu_data/preprocessed/shard_XXXX_stage2.npz` — **4 shards,
numbered 0, 1, 2 and 5** (3 and 4 were never preprocessed; verified 2026-08-24. Shard 5 is
the held-out one used by the metric suite).
Flat arrays + offsets (CSR-style; hit `i` of particle `p` is `flat[offset[p]:offset[p+1]]`):
- `particle_features (N,6)` = **[logpt, eta, phi, pdg_class, charge, mass]**
- `particle_aux (N,6)`      = **[primary, parent_id, vx, vy, vz, energy]**
- `tracker_hits_flat (M,5)` = **[layer_class(0..47), r, phi, z, time]** — note: preprocessing
  collapses `(volume,layer)` → one of **48 layer classes** and **drops `surface_id`**.
- `calo_hits_flat`, `calo_offsets`, `tracker_offsets`, `event_ids`.
- **Hit order in stage2 is ARBITRARY** (grouped by `particle_id`, not sorted). Verified
  2026-08-24: Spearman(index, r) = 0.01, ~50% of steps decrease in r, median max drop 158 mm.
  The **r-ascending (inner→outer) sort happens in `build_tracker_slice.py:85`**, so it holds for
  the tracker SLICES only. Anything reading `tracker_hits_flat` directly must sort by r itself —
  "the last stored hit" is NOT the outermost hit.
- Column-index constants live at the top of the tracker scripts (`PF_*`, `AUX_*`, `TH_*`).

### Tracker training slices
`/scratch/gpfs/IOJALVO/lv7805/genpu_data/tracker_slice/*.npz` — e.g. `multispecies.npz`
(all species), `primary.npz` / `secondary.npz` (truth-flag population split), `pion.npz`,
`v2_pion.npz`. Each carries `cont_mean`/`cont_std` normalization. Built by
[scripts/build_tracker_slice.py](scripts/build_tracker_slice.py).

## Models

- **Tracker v1** — [src/genpu/models/tracker_ar.py](src/genpu/models/tracker_ar.py): tokenized
  autoregressive hit generator. Predicts `layer_class (48) + per-layer-standardized residual bins
  (r,φ,z 512-bin, time 64-bin)`, inner→outer. Flags: `use_vertex`, `use_helix`, `use_mom_feat`,
  `seed_weight`, `jitter`.
- **Tracker v2** — [src/genpu/models/tracker_state_ar.py](src/genpu/models/tracker_state_ar.py):
  adds a supervised direction state + a STOP head (no truth `n_hits`). Same residual position head.
- **Wrapper/trainer**: [tracker_model.py](src/genpu/models/tracker_model.py),
  [scripts/train_tracker.py](scripts/train_tracker.py). Count head:
  [src/genpu/models/count_head.py](src/genpu/models/count_head.py).

## Evaluation

Quality metric is a **two-sample event gate**: an MLP classifier separating generated vs real
events on per-event features (rank-AUC; **0.5 = indistinguishable**, 1.0 = trivially separable).
`scripts/tracker_honest_gate.py`. Current honest tracker: **base-8 gate 0.77**. See
[TRACKER_GATE_FINDINGS.md](TRACKER_GATE_FINDINGS.md) for the open drift problem and the
surface-representation lever.

## Working conventions

- Research-memory workflow: **STATUS.md** (rolling) + **experiment-memory/** (append-only, one
  file per run). Read STATUS.md before proposing experiments; maintain via start-experiment /
  log-experiment skills. Regenerate diagnostic plots after any retrain.
