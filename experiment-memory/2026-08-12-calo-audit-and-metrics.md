# 2026-08-12 — Calo audit + held-out metrics (publication readiness)

- **Branch:** flow-response. First time the calo work is recorded in experiment-memory (it lived only
  in git + `/scratch` checkpoints until now).

## What the calo is
CaloClouds-lite per-particle shower generator (`src/genpu/flow/calo_flow.py`): `GlobalHead`
(Gaussian mixture over `[total_logE, log_n]`) + `PointCFM` (conditional flow-matching, 50-step Euler
ODE, 2-D `(d_eta,d_phi)` offsets) + `EnergyHead` (floor-Bernoulli + Gaussian mixture per cell).
Conditions on the 7 CONT_FEATURES + PDG embedding. Best photon recipe = quantile normalization +
count-dither (`photon_qtd_v1`). Deliberately replaced a voxel-DDPM calo that failed in v1–v4.

## Held-out metrics (shard 5, `scripts/calo_metrics.py`)
**Photon (`photon_qtd_v1/checkpoint_040000`) — works, generalizes:**
- Event gate **AUC 0.557** held-out ≈ 0.546 in-sample → no overfitting.
- Energy: linearity ⟨E_reco/E_true⟩ tracks real across energy bins; total-E W/σ 0.027, cell-E 0.016,
  cells/shower 0.01 — energy is well modelled.
- **Weak spot: lateral shape** — `shower_width` W/σ **0.44**, `d_eta` 0.47 (gen showers too wide/narrow).
- Speed **14.5 µs/particle** (50 ODE steps) — 13× FASTER than the tracker AR (191 µs); full-event
  per-particle cost is tracker-dominated, not calo-dominated.

**Pion — retrained with current code** (`pion_qtd_v2`, fresh `pion_ctr_v2` slice, quantile+dither,
40k steps; old checkpoint+slice were both stale — 5-dim `cond`, 3-D point head). Held-out shard 5:
- Gate **AUC 0.809** (first pion number; harder than photon — ~15 cells/shower vs 4).
- **Positions/width GOOD** (the per-shower core helps): `d_eta` W/σ 0.006, `shower_width` 0.32 (vs
  photon 0.44). Core mechanism validated.
- **Energy is the weak spot**: resolution too broad (reso_gen ~1.5-1.7 vs real ~1.0) + a
  catastrophic low-E outlier (resp_gen 4.7 vs 0.016 — the energy mixture head samples unphysically
  huge cells, a heavy-tail failure). This drives the 0.81.
- Speed 27.8 µs/particle (2× photon, 7× faster than tracker).

**Photon vs pion fail in OPPOSITE ways** — photon weak on width, pion weak on energy. Unified fix:
apply the core to photon (fixes width) + bound the energy-head tail like the position quantile inverse
(stops the huge-cell blow-ups).

## Reproducibility issues surfaced (fix before release)
- `preprocessing.py` emits tracker hits as 6-col `[r,phi,z,time,vol,layer]`, but the on-disk training
  shards 0-2 are 5-col `[layer_class,r,phi,z,time]` — freshly-preprocessed data doesn't match training
  format (converted shard 5 by hand for the held-out eval). Calo format DID match (5-col).
- No held-out preprocessed shard existed — all prior gates (tracker + calo) were in-sample. Built
  shard 5 as the held-out set.

## Verdict / next
Photon calo = publication-grade single-species result (held-out, energy-faithful, fast; lateral width
the honest limitation). Pion/multi-species = a retrain, now in progress. Full-event fast-sim framing
(condition on real particles; no Stage-1 needed — that's Pythia/data, not learned).
