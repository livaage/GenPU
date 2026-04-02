#!/usr/bin/env python
"""Verify that matching event_ids across tables actually correspond to the same event.

Reads directly from Arrow cache — no HF library or internet needed.

Checks:
1. For shared event_ids, do particle_ids in tracker_hits appear in the particles table?
2. Do contrib_particle_ids in calo_hits overlap with particle_ids from particles?
3. Are hit positions spatially consistent? (tracker hits should be at smaller r than calo hits)
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from genpu.data import load_shard, explode_list_columns, build_event_index

# Load just the first shard of each subset (~10k events each)
print("Loading first shard of each subset...")
p_table = load_shard("particles", 0)
t_table = load_shard("tracker_hits", 0)
c_table = load_shard("calo_hits", 0)

print(f"  particles:    {p_table.num_rows} events")
print(f"  tracker_hits: {t_table.num_rows} events")
print(f"  calo_hits:    {c_table.num_rows} events")

# Find common event_ids in these shards
p_idx = build_event_index(p_table)
t_idx = build_event_index(t_table)
c_idx = build_event_index(c_table)
common = sorted(set(p_idx) & set(t_idx) & set(c_idx))
print(f"Common event_ids in shard 0: {len(common)} (first 10: {common[:10]})")

N_CHECK = min(20, len(common))

print(f"\n{'='*70}")
print(f"Checking {N_CHECK} common events")
print(f"{'='*70}")

for eid in common[:N_CHECK]:
    p_ev = explode_list_columns(p_table, p_idx[eid])
    t_ev = explode_list_columns(t_table, t_idx[eid])
    c_ev = explode_list_columns(c_table, c_idx[eid])

    # Particle IDs from each table
    p_pids = set(int(x) for x in p_ev["particle_id"])
    t_pids = set(int(x) for x in t_ev["particle_id"])

    # Calo contributing particle IDs (flattened)
    c_pids = set()
    for cpid_list in c_ev["contrib_particle_ids"]:
        c_pids.update(int(x) for x in cpid_list)

    # Overlaps
    t_in_p = len(t_pids & p_pids)
    t_not_in_p = len(t_pids - p_pids)
    c_in_p = len(c_pids & p_pids)
    c_not_in_p = len(c_pids - p_pids)

    # Spatial check: tracker r range vs calo r range
    t_r = np.sqrt(t_ev["x"]**2 + t_ev["y"]**2)
    c_r = np.sqrt(c_ev["x"]**2 + c_ev["y"]**2)

    print(
        f"\n  event_id={eid}:  "
        f"particles={len(p_pids)}, tracker_hits={len(t_ev['x'])}, calo_hits={len(c_ev['x'])}"
    )
    print(
        f"    Tracker PIDs in particles: {t_in_p}/{len(t_pids)}  "
        f"(not in particles: {t_not_in_p})"
    )
    print(
        f"    Calo PIDs in particles:    {c_in_p}/{len(c_pids)}  "
        f"(not in particles: {c_not_in_p})"
    )
    print(
        f"    Tracker r range: [{t_r.min():.0f}, {t_r.max():.0f}] mm  "
        f"Calo r range: [{c_r.min():.0f}, {c_r.max():.0f}] mm"
    )

print(f"\n{'='*70}")
print("Done.")
