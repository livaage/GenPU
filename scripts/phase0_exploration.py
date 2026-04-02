#!/usr/bin/env python
"""Phase 0: Data Exploration — ColliderML pileup_only_pu0

Reads directly from Arrow cache — no HF library or internet needed.
Uses shard 0 of each subset (~10k events) for all distributions.

Goals:
1. Characterise single min-bias interactions: particle multiplicities, pT spectra, hit counts
2. Understand detector geometry: layers, cells, subsystem codes
3. Build the particle → hits mapping
4. Visualise example events in (η, φ) and (x, y, z)

Saves all plots to plots/ directory.
"""

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from genpu.data import (
    load_shard,
    explode_list_columns,
    get_num_events,
    get_particle_hit_map,
    compute_eta_phi,
    build_event_index,
)

PLOT_DIR = Path(__file__).resolve().parent.parent / "plots"
PLOT_DIR.mkdir(exist_ok=True)

plt.rcParams.update({"figure.figsize": (10, 6), "font.size": 12})


# ── 1. Load Data (shard 0 only) ────────────────────────────────────────────

print("Loading shard 0 of each subset...")
particles_table = load_shard("particles", 0)
print(f"  particles:    {get_num_events(particles_table)} events, columns: {particles_table.column_names}")

tracker_table = load_shard("tracker_hits", 0)
print(f"  tracker_hits: {get_num_events(tracker_table)} events, columns: {tracker_table.column_names}")

calo_table = load_shard("calo_hits", 0)
print(f"  calo_hits:    {get_num_events(calo_table)} events, columns: {calo_table.column_names}")


# ── 2. Particle Multiplicity & Kinematics ───────────────────────────────────

n_events = get_num_events(particles_table)
n_sample = min(n_events, 5000)

multiplicities = []
all_pt = []
all_eta = []
all_pdg = []
primary_fracs = []

for i in tqdm(range(n_sample), desc="Scanning particles"):
    ev = explode_list_columns(particles_table, i)
    n_part = len(ev["particle_id"])
    multiplicities.append(n_part)

    pt, eta, phi = compute_eta_phi(ev["px"], ev["py"], ev["pz"])
    all_pt.append(pt)
    all_eta.append(eta)
    all_pdg.append(ev["pdg_id"])

    if "primary" in ev:
        primary_fracs.append(ev["primary"].sum() / max(n_part, 1))

multiplicities = np.array(multiplicities)
all_pt_flat = np.concatenate(all_pt)
all_eta_flat = np.concatenate(all_eta)
all_pdg_flat = np.concatenate(all_pdg)

# ── Plot: particle distributions ──

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].hist(multiplicities, bins=80, edgecolor="black", alpha=0.7)
axes[0, 0].set_xlabel("Particle multiplicity per event")
axes[0, 0].set_ylabel("Events")
axes[0, 0].set_title(
    f"Particle multiplicity (mean={multiplicities.mean():.1f}, std={multiplicities.std():.1f})"
)

axes[0, 1].hist(all_pt_flat, bins=np.logspace(-3, 2, 100), edgecolor="black", alpha=0.7)
axes[0, 1].set_xscale("log")
axes[0, 1].set_yscale("log")
axes[0, 1].set_xlabel("pT [GeV/c]")
axes[0, 1].set_ylabel("Particles")
axes[0, 1].set_title("Transverse momentum spectrum")

axes[1, 0].hist(all_eta_flat[np.isfinite(all_eta_flat)], bins=100, edgecolor="black", alpha=0.7)
axes[1, 0].set_xlabel("η (pseudorapidity)")
axes[1, 0].set_ylabel("Particles")
axes[1, 0].set_title("Pseudorapidity distribution")

pdg_counts = Counter(all_pdg_flat.astype(int))
top_pdg = pdg_counts.most_common(15)
pdg_labels = [str(p) for p, _ in top_pdg]
pdg_vals = [c for _, c in top_pdg]
axes[1, 1].barh(pdg_labels, pdg_vals, edgecolor="black", alpha=0.7)
axes[1, 1].set_xlabel("Count")
axes[1, 1].set_title("Top 15 particle species (PDG ID)")

plt.tight_layout()
plt.savefig(PLOT_DIR / "phase0_particle_distributions.png", dpi=150, bbox_inches="tight")
plt.close()

print(f"\nParticle summary ({n_sample} events from shard 0):")
print(
    f"  Multiplicity: mean={multiplicities.mean():.1f}, "
    f"median={np.median(multiplicities):.0f}, "
    f"min={multiplicities.min()}, max={multiplicities.max()}"
)
print(f"  pT: median={np.median(all_pt_flat):.3f} GeV, mean={all_pt_flat.mean():.3f} GeV")
if primary_fracs:
    print(f"  Primary particle fraction: {np.mean(primary_fracs):.2%}")


# ── 3. Tracker Hit Characterisation & Detector Geometry ─────────────────────

n_tracker_sample = min(get_num_events(tracker_table), 2000)

hit_counts = []
all_volume_ids = []
all_layer_ids = []
sample_xyz = []

for i in tqdm(range(n_tracker_sample), desc="Scanning tracker hits"):
    ev = explode_list_columns(tracker_table, i)
    n_hits = len(ev["x"])
    hit_counts.append(n_hits)
    all_volume_ids.append(ev["volume_id"])
    all_layer_ids.append(ev["layer_id"])
    if i < 5:
        sample_xyz.append((ev["x"], ev["y"], ev["z"]))

hit_counts = np.array(hit_counts)
all_volume_flat = np.concatenate(all_volume_ids)
all_layer_flat = np.concatenate(all_layer_ids)

# ── Plot: tracker geometry ──

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].hist(hit_counts, bins=60, edgecolor="black", alpha=0.7)
axes[0, 0].set_xlabel("Tracker hits per event")
axes[0, 0].set_ylabel("Events")
axes[0, 0].set_title(f"Tracker hits per event (mean={hit_counts.mean():.1f})")

vol_unique, vol_counts = np.unique(all_volume_flat, return_counts=True)
axes[0, 1].bar([str(v) for v in vol_unique], vol_counts, edgecolor="black", alpha=0.7)
axes[0, 1].set_xlabel("Volume ID")
axes[0, 1].set_ylabel("Hits")
axes[0, 1].set_title("Hits per detector volume")
axes[0, 1].tick_params(axis="x", rotation=45)

layer_unique, layer_counts = np.unique(all_layer_flat, return_counts=True)
axes[1, 0].bar([str(l) for l in layer_unique], layer_counts, edgecolor="black", alpha=0.7)
axes[1, 0].set_xlabel("Layer ID")
axes[1, 0].set_ylabel("Hits")
axes[1, 0].set_title(f"Hits per layer ({len(layer_unique)} unique layers)")
axes[1, 0].tick_params(axis="x", rotation=45)

x0, y0, z0 = sample_xyz[0]
axes[1, 1].scatter(x0, y0, s=1, alpha=0.5)
axes[1, 1].set_xlabel("x [mm]")
axes[1, 1].set_ylabel("y [mm]")
axes[1, 1].set_title("Tracker hits x-y (event 0)")
axes[1, 1].set_aspect("equal")

plt.tight_layout()
plt.savefig(PLOT_DIR / "phase0_tracker_geometry.png", dpi=150, bbox_inches="tight")
plt.close()

print(f"\nDetector geometry summary:")
print(f"  Volume IDs: {sorted(vol_unique.tolist())}")
print(f"  Layer IDs: {sorted(layer_unique.tolist())}")
print(f"  Tracker hits/event: mean={hit_counts.mean():.1f}, std={hit_counts.std():.1f}")

# ── Plot: tracker r-z view ──

fig, ax = plt.subplots(figsize=(14, 5))
for x, y, z in sample_xyz[:3]:
    r = np.sqrt(x**2 + y**2)
    ax.scatter(z, r, s=0.5, alpha=0.3)
ax.set_xlabel("z [mm]")
ax.set_ylabel("r [mm]")
ax.set_title("Tracker r-z view (3 events overlaid)")
plt.tight_layout()
plt.savefig(PLOT_DIR / "phase0_tracker_rz.png", dpi=150, bbox_inches="tight")
plt.close()


# ── 4. Calorimeter Hit Characterisation ─────────────────────────────────────

n_calo_sample = min(get_num_events(calo_table), 2000)

calo_hit_counts = []
all_calo_energy = []
all_calo_detector = []
event_total_energy = []
sample_calo_xyz = []

for i in tqdm(range(n_calo_sample), desc="Scanning calo hits"):
    ev = explode_list_columns(calo_table, i)
    n_hits = len(ev["x"])
    calo_hit_counts.append(n_hits)
    all_calo_energy.append(ev["total_energy"])
    all_calo_detector.append(ev["detector"])
    event_total_energy.append(ev["total_energy"].sum())
    if i < 5:
        sample_calo_xyz.append((ev["x"], ev["y"], ev["z"], ev["total_energy"]))

calo_hit_counts = np.array(calo_hit_counts)
all_calo_energy_flat = np.concatenate(all_calo_energy)
all_calo_det_flat = np.concatenate(all_calo_detector)
event_total_energy = np.array(event_total_energy)

# ── Plot: calo distributions ──

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].hist(calo_hit_counts, bins=60, edgecolor="black", alpha=0.7)
axes[0, 0].set_xlabel("Calorimeter hits per event")
axes[0, 0].set_ylabel("Events")
axes[0, 0].set_title(f"Calo hits/event (mean={calo_hit_counts.mean():.1f})")

axes[0, 1].hist(
    all_calo_energy_flat[all_calo_energy_flat > 0],
    bins=np.logspace(-5, 2, 100),
    edgecolor="black",
    alpha=0.7,
)
axes[0, 1].set_xscale("log")
axes[0, 1].set_yscale("log")
axes[0, 1].set_xlabel("Cell energy [GeV]")
axes[0, 1].set_ylabel("Cells")
axes[0, 1].set_title("Calorimeter cell energy spectrum")

det_unique, det_counts = np.unique(all_calo_det_flat, return_counts=True)
axes[1, 0].bar([str(d) for d in det_unique], det_counts, edgecolor="black", alpha=0.7)
axes[1, 0].set_xlabel("Detector code")
axes[1, 0].set_ylabel("Hits")
axes[1, 0].set_title(f"Hits per calo subsystem ({len(det_unique)} subsystems)")

axes[1, 1].hist(event_total_energy, bins=60, edgecolor="black", alpha=0.7)
axes[1, 1].set_xlabel("Total calorimeter energy [GeV]")
axes[1, 1].set_ylabel("Events")
axes[1, 1].set_title(f"Total calo energy/event (mean={event_total_energy.mean():.2f} GeV)")

plt.tight_layout()
plt.savefig(PLOT_DIR / "phase0_calo_distributions.png", dpi=150, bbox_inches="tight")
plt.close()

# ── Plot: calo η-φ occupancy ──

x0, y0, z0, e0 = sample_calo_xyz[0]
r0 = np.sqrt(x0**2 + y0**2)
eta0 = np.arctanh(
    np.clip(z0 / (np.sqrt(x0**2 + y0**2 + z0**2) + 1e-10), -1 + 1e-7, 1 - 1e-7)
)
phi0 = np.arctan2(y0, x0)

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

sc = axes[0].scatter(eta0, phi0, c=e0, s=5, cmap="hot", norm=LogNorm(vmin=max(e0.min(), 1e-5)))
plt.colorbar(sc, ax=axes[0], label="Energy [GeV]")
axes[0].set_xlabel("η")
axes[0].set_ylabel("φ")
axes[0].set_title("Calo hits η-φ (event 0, color=energy)")

sc2 = axes[1].scatter(z0, r0, c=e0, s=5, cmap="hot", norm=LogNorm(vmin=max(e0.min(), 1e-5)))
plt.colorbar(sc2, ax=axes[1], label="Energy [GeV]")
axes[1].set_xlabel("z [mm]")
axes[1].set_ylabel("r [mm]")
axes[1].set_title("Calo hits r-z (event 0)")

plt.tight_layout()
plt.savefig(PLOT_DIR / "phase0_calo_etaphi.png", dpi=150, bbox_inches="tight")
plt.close()


# ── 5. Particle → Hit Mapping (joined by event_id) ─────────────────────────

print("\n── Building event_id indices for shard 0 ──")
p_idx = build_event_index(particles_table)
t_idx = build_event_index(tracker_table)
c_idx = build_event_index(calo_table)
common_ids = sorted(set(p_idx) & set(t_idx) & set(c_idx))
print(f"  Particles: {len(p_idx)} events, Tracker: {len(t_idx)} events, Calo: {len(c_idx)} events")
print(f"  Common events in shard 0: {len(common_ids)}")

# Use the first common event for the detailed mapping
eid = common_ids[0]
print(f"\n── Particle → Hit Mapping (event_id={eid}) ──")

p_ev = explode_list_columns(particles_table, p_idx[eid])
t_ev = explode_list_columns(tracker_table, t_idx[eid])
c_ev = explode_list_columns(calo_table, c_idx[eid])

hit_map = get_particle_hit_map(p_ev, t_ev, c_ev)

n_with_tracker = sum(1 for v in hit_map.values() if len(v["tracker_hits"]) > 0)
n_with_calo = sum(1 for v in hit_map.values() if len(v["calo_hits"]) > 0)
n_with_both = sum(
    1
    for v in hit_map.values()
    if len(v["tracker_hits"]) > 0 and len(v["calo_hits"]) > 0
)
n_with_neither = sum(
    1
    for v in hit_map.values()
    if len(v["tracker_hits"]) == 0 and len(v["calo_hits"]) == 0
)

print(f"  Total particles: {len(hit_map)}")
print(f"  With tracker hits: {n_with_tracker} ({n_with_tracker/len(hit_map):.1%})")
print(f"  With calo hits: {n_with_calo} ({n_with_calo/len(hit_map):.1%})")
print(f"  With both: {n_with_both} ({n_with_both/len(hit_map):.1%})")
print(f"  With neither: {n_with_neither} ({n_with_neither/len(hit_map):.1%})")

# ── Plot: hits per particle ──

trk_per_particle = [len(v["tracker_hits"]) for v in hit_map.values()]
calo_per_particle = [len(v["calo_hits"]) for v in hit_map.values()]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(
    trk_per_particle,
    bins=range(0, max(trk_per_particle) + 2),
    edgecolor="black",
    alpha=0.7,
)
axes[0].set_xlabel("Tracker hits per particle")
axes[0].set_ylabel("Particles")
axes[0].set_title("Tracker hits per particle")

axes[1].hist(
    calo_per_particle,
    bins=range(0, min(max(calo_per_particle) + 2, 50)),
    edgecolor="black",
    alpha=0.7,
)
axes[1].set_xlabel("Calo hits per particle")
axes[1].set_ylabel("Particles")
axes[1].set_title("Calo hits per particle")

plt.tight_layout()
plt.savefig(PLOT_DIR / "phase0_hits_per_particle.png", dpi=150, bbox_inches="tight")
plt.close()

print(f"  Tracker hits/particle: mean={np.mean(trk_per_particle):.2f}, max={max(trk_per_particle)}")
print(f"  Calo hits/particle: mean={np.mean(calo_per_particle):.2f}, max={max(calo_per_particle)}")

# ── Plot: tracker hits vs pT ──

pts_with_hits = []
nhits_with_hits = []
for pid, v in hit_map.items():
    if len(v["tracker_hits"]) > 0:
        px = v["kinematics"].get("px", 0)
        py = v["kinematics"].get("py", 0)
        pt = np.sqrt(px**2 + py**2)
        pts_with_hits.append(pt)
        nhits_with_hits.append(len(v["tracker_hits"]))

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(pts_with_hits, nhits_with_hits, s=3, alpha=0.3)
ax.set_xlabel("pT [GeV/c]")
ax.set_ylabel("Number of tracker hits")
ax.set_title("Tracker hit count vs particle pT")
ax.set_xscale("log")
plt.tight_layout()
plt.savefig(PLOT_DIR / "phase0_nhits_vs_pt.png", dpi=150, bbox_inches="tight")
plt.close()


# ── Done ────────────────────────────────────────────────────────────────────

print(f"\nAll plots saved to {PLOT_DIR}/")
print("Done.")
