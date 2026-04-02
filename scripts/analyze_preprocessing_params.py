#!/usr/bin/env python
"""Determine preprocessing parameters from data using vectorized processing.

Computes:
- N_max_tracker, N_max_calo (padding sizes from 99th/99.9th percentile)
- Visible particles per event (particles with >= 1 hit)
- Feature ranges for normalization

Uses the vectorized process_event_vectorized(), so ~500 events is feasible.
Submit as a short SLURM job (submit_analyze_params.slurm).
"""

import json
import sys
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from genpu.data import load_shard, explode_list_columns, build_event_index
from genpu.preprocessing import process_event_vectorized

print("Loading shard 0...")
p_table = load_shard("particles", 0)
t_table = load_shard("tracker_hits", 0)
c_table = load_shard("calo_hits", 0)

p_idx = build_event_index(p_table)
t_idx = build_event_index(t_table)
c_idx = build_event_index(c_table)
common = sorted(set(p_idx) & set(t_idx) & set(c_idx))

N_EVENTS = min(500, len(common))
print(f"Analysing {N_EVENTS} common events...\n")

# Collectors
trk_hits_per_particle = []
calo_hits_per_particle = []
visible_per_event = []
total_per_event = []

# Feature collectors (from visible particles only)
all_features = []  # (log_pt, eta, phi, pdg_class, charge, mass)
all_trk_features = []  # (r, phi, z, time, volume_id, layer_id)
all_calo_features = []  # (eta, phi, log_energy, contrib_frac, detector)

for eid in tqdm(common[:N_EVENTS], desc="Events"):
    p_ev = explode_list_columns(p_table, p_idx[eid])
    t_ev = explode_list_columns(t_table, t_idx[eid])
    c_ev = explode_list_columns(c_table, c_idx[eid])

    result = process_event_vectorized(p_ev, t_ev, c_ev)

    vis = result["visible_mask"]
    vis_idx = np.where(vis)[0]

    trk_hits_per_particle.append(result["n_tracker_hits"][vis_idx])
    calo_hits_per_particle.append(result["n_calo_hits"][vis_idx])
    visible_per_event.append(vis.sum())
    total_per_event.append(len(result["particle_ids"]))

    if vis.sum() > 0:
        all_features.append(result["particle_features"][vis_idx])
        for j in vis_idx:
            if result["tracker_hits"][j].shape[0] > 0:
                all_trk_features.append(result["tracker_hits"][j])
            if result["calo_hits"][j].shape[0] > 0:
                all_calo_features.append(result["calo_hits"][j])

# Concatenate
trk_hits_per_particle = np.concatenate(trk_hits_per_particle)
calo_hits_per_particle = np.concatenate(calo_hits_per_particle)
visible_per_event = np.array(visible_per_event)
total_per_event = np.array(total_per_event)
all_features = np.concatenate(all_features)
all_trk_features = np.concatenate(all_trk_features) if all_trk_features else np.empty((0, 6))
all_calo_features = np.concatenate(all_calo_features) if all_calo_features else np.empty((0, 5))

# ── Results ──

print("\n" + "=" * 70)
print("PADDING SIZES (hits per visible particle)")
print("=" * 70)
trk_nonzero = trk_hits_per_particle[trk_hits_per_particle > 0]
calo_nonzero = calo_hits_per_particle[calo_hits_per_particle > 0]

for name, arr in [("Tracker hits/particle (nonzero)", trk_nonzero),
                  ("Calo hits/particle (nonzero)", calo_nonzero)]:
    print(f"\n  {name}:")
    print(f"    mean={arr.mean():.1f}, median={np.median(arr):.0f}, std={arr.std():.1f}")
    print(f"    min={arr.min()}, max={arr.max()}")
    for p in [90, 95, 99, 99.5, 99.9]:
        print(f"    P{p}={np.percentile(arr, p):.0f}")

print("\n" + "=" * 70)
print("VISIBLE PARTICLES PER EVENT")
print("=" * 70)
print(f"  mean={visible_per_event.mean():.1f}, median={np.median(visible_per_event):.0f}, "
      f"std={visible_per_event.std():.1f}")
print(f"  min={visible_per_event.min()}, max={visible_per_event.max()}")
for p in [90, 95, 99, 99.9]:
    print(f"  P{p}={np.percentile(visible_per_event, p):.0f}")
print(f"  Visible fraction: {visible_per_event.mean()/total_per_event.mean():.1%}")

print("\n" + "=" * 70)
print("FEATURE RANGES — particle features (log_pt, eta, phi, pdg_class, charge, mass)")
print("=" * 70)
feat_names = ["log_pt", "eta", "phi", "pdg_class", "charge", "mass"]
for i, name in enumerate(feat_names):
    col = all_features[:, i]
    print(f"  {name:>10s}: mean={col.mean():.4f}, std={col.std():.4f}, "
          f"min={col.min():.4f}, max={col.max():.4f}")

print("\n" + "=" * 70)
print("FEATURE RANGES — tracker hits (r, phi, z, time, volume_id, layer_id)")
print("=" * 70)
trk_names = ["r", "phi", "z", "time", "volume_id", "layer_id"]
for i, name in enumerate(trk_names):
    col = all_trk_features[:, i]
    print(f"  {name:>10s}: mean={col.mean():.4f}, std={col.std():.4f}, "
          f"min={col.min():.4f}, max={col.max():.4f}")

print("\n" + "=" * 70)
print("FEATURE RANGES — calo hits (eta, phi, log_energy, contrib_frac, detector)")
print("=" * 70)
calo_names = ["eta", "phi", "log_energy", "contrib_frac", "detector"]
for i, name in enumerate(calo_names):
    col = all_calo_features[:, i]
    print(f"  {name:>10s}: mean={col.mean():.4f}, std={col.std():.4f}, "
          f"min={col.min():.4f}, max={col.max():.4f}")

# Save as JSON for programmatic use
output = {
    "n_events_analyzed": int(N_EVENTS),
    "padding": {
        "tracker_p99": int(np.percentile(trk_nonzero, 99)),
        "tracker_p999": int(np.percentile(trk_nonzero, 99.9)),
        "tracker_max": int(trk_nonzero.max()),
        "calo_p99": int(np.percentile(calo_nonzero, 99)),
        "calo_p999": int(np.percentile(calo_nonzero, 99.9)),
        "calo_max": int(calo_nonzero.max()),
    },
    "visible_particles": {
        "mean": float(visible_per_event.mean()),
        "median": float(np.median(visible_per_event)),
        "p99": int(np.percentile(visible_per_event, 99)),
        "max": int(visible_per_event.max()),
        "fraction": float(visible_per_event.mean() / total_per_event.mean()),
    },
    "normalization": {
        "particle": {
            name: {"mean": float(all_features[:, i].mean()), "std": float(all_features[:, i].std())}
            for i, name in enumerate(feat_names) if name not in ("pdg_class",)
        },
        "tracker": {
            name: {"mean": float(all_trk_features[:, i].mean()), "std": float(all_trk_features[:, i].std())}
            for i, name in enumerate(trk_names) if name not in ("volume_id", "layer_id")
        },
        "calo": {
            name: {"mean": float(all_calo_features[:, i].mean()), "std": float(all_calo_features[:, i].std())}
            for i, name in enumerate(calo_names) if name not in ("detector",)
        },
    },
}

out_path = Path(__file__).resolve().parent.parent / "preprocessing_params.json"
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nSaved parameters to {out_path}")
