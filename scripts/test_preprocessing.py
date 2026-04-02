#!/usr/bin/env python
"""Quick test: process 1 event with the vectorized pipeline, compare to old method."""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from genpu.data import (
    load_shard, explode_list_columns, build_event_index, get_particle_hit_map,
)
from genpu.preprocessing import process_event_vectorized

print("Loading shard 0...")
p_table = load_shard("particles", 0)
t_table = load_shard("tracker_hits", 0)
c_table = load_shard("calo_hits", 0)

p_idx = build_event_index(p_table)
t_idx = build_event_index(t_table)
c_idx = build_event_index(c_table)
common = sorted(set(p_idx) & set(t_idx) & set(c_idx))

# Test on a few events
for eid in common[:5]:
    p_ev = explode_list_columns(p_table, p_idx[eid])
    t_ev = explode_list_columns(t_table, t_idx[eid])
    c_ev = explode_list_columns(c_table, c_idx[eid])

    n_particles = len(p_ev["particle_id"])
    n_tracker = len(t_ev["x"])
    n_calo = len(c_ev["x"])

    # Vectorized version
    t0 = time.time()
    result = process_event_vectorized(p_ev, t_ev, c_ev)
    t_vec = time.time() - t0

    # Old version
    t0 = time.time()
    hit_map = get_particle_hit_map(p_ev, t_ev, c_ev)
    t_old = time.time() - t0

    # Compare
    n_vis_vec = result["visible_mask"].sum()
    n_vis_old = sum(
        1 for v in hit_map.values()
        if len(v["tracker_hits"]) > 0 or len(v["calo_hits"]) > 0
    )

    trk_total_vec = result["n_tracker_hits"].sum()
    trk_total_old = sum(len(v["tracker_hits"]) for v in hit_map.values())

    calo_total_vec = result["n_calo_hits"].sum()
    calo_total_old = sum(len(v["calo_hits"]) for v in hit_map.values())

    match = (n_vis_vec == n_vis_old and trk_total_vec == trk_total_old
             and calo_total_vec == calo_total_old)

    print(f"\nevent_id={eid}: {n_particles} particles, {n_tracker} tracker, {n_calo} calo hits")
    print(f"  Vectorized: {t_vec:.3f}s  |  Old: {t_old:.3f}s  |  Speedup: {t_old/max(t_vec,0.001):.1f}x")
    print(f"  Visible particles: vec={n_vis_vec}, old={n_vis_old}  {'OK' if n_vis_vec == n_vis_old else 'MISMATCH!'}")
    print(f"  Total tracker hits: vec={trk_total_vec}, old={trk_total_old}  {'OK' if trk_total_vec == trk_total_old else 'MISMATCH!'}")
    print(f"  Total calo hits:    vec={calo_total_vec}, old={calo_total_old}  {'OK' if calo_total_vec == calo_total_old else 'MISMATCH!'}")
    print(f"  Max tracker hits/particle: {result['n_tracker_hits'].max()}")
    print(f"  Max calo hits/particle: {result['n_calo_hits'].max()}")

    if not match:
        print("  *** VALIDATION FAILED ***")
        sys.exit(1)

print("\nAll events validated. Vectorized version matches original.")
