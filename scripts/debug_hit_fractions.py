#!/usr/bin/env python
"""Break down particle-hit mapping by PDG ID to understand the fractions."""

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from genpu.data import (
    load_shard,
    explode_list_columns,
    get_particle_hit_map,
    build_event_index,
)

PDG_NAMES = {
    11: "e-", -11: "e+", 13: "mu-", -13: "mu+",
    12: "nu_e", -12: "nu_e~", 14: "nu_mu", -14: "nu_mu~",
    16: "nu_tau", -16: "nu_tau~",
    22: "photon", 111: "pi0", 211: "pi+", -211: "pi-",
    321: "K+", -321: "K-", 130: "K0L", 310: "K0S",
    2112: "neutron", -2112: "antineutron",
    2212: "proton", -2212: "antiproton",
}

print("Loading shard 0...")
p_table = load_shard("particles", 0)
t_table = load_shard("tracker_hits", 0)
c_table = load_shard("calo_hits", 0)

p_idx = build_event_index(p_table)
t_idx = build_event_index(t_table)
c_idx = build_event_index(c_table)
common = sorted(set(p_idx) & set(t_idx) & set(c_idx))

# Aggregate over several events
N_EVENTS = min(50, len(common))
print(f"Analysing {N_EVENTS} events...\n")

# Per-PDG counters: pdg -> [total, has_tracker, has_calo, has_both, has_neither]
pdg_stats = defaultdict(lambda: [0, 0, 0, 0, 0])

for eid in common[:N_EVENTS]:
    p_ev = explode_list_columns(p_table, p_idx[eid])
    t_ev = explode_list_columns(t_table, t_idx[eid])
    c_ev = explode_list_columns(c_table, c_idx[eid])
    hit_map = get_particle_hit_map(p_ev, t_ev, c_ev)

    for pid, v in hit_map.items():
        pdg = int(v["kinematics"].get("pdg_id", 0))
        has_trk = len(v["tracker_hits"]) > 0
        has_cal = len(v["calo_hits"]) > 0
        s = pdg_stats[pdg]
        s[0] += 1
        s[1] += has_trk
        s[2] += has_cal
        s[3] += (has_trk and has_cal)
        s[4] += (not has_trk and not has_cal)

# Sort by total count
sorted_pdg = sorted(pdg_stats.items(), key=lambda x: -x[1][0])

print(f"{'PDG':>8s}  {'Name':>12s}  {'Total':>6s}  {'Tracker':>8s}  {'Calo':>8s}  {'Both':>8s}  {'Neither':>8s}")
print("-" * 80)
for pdg, (total, trk, cal, both, neither) in sorted_pdg[:25]:
    name = PDG_NAMES.get(pdg, "?")
    print(
        f"{pdg:>8d}  {name:>12s}  {total:>6d}  "
        f"{trk:>5d} ({trk/total:>4.0%})  "
        f"{cal:>5d} ({cal/total:>4.0%})  "
        f"{both:>5d} ({both/total:>4.0%})  "
        f"{neither:>5d} ({neither/total:>4.0%})"
    )

# Summary
total_all = sum(s[0] for s in pdg_stats.values())
total_neither = sum(s[4] for s in pdg_stats.values())
neutrino_pdgs = {12, -12, 14, -14, 16, -16}
neutrino_neither = sum(pdg_stats[p][4] for p in neutrino_pdgs if p in pdg_stats)
neutrino_total = sum(pdg_stats[p][0] for p in neutrino_pdgs if p in pdg_stats)
print(f"\nTotal particles: {total_all}")
print(f"Total with neither: {total_neither} ({total_neither/total_all:.1%})")
print(f"  of which neutrinos: {neutrino_neither} ({neutrino_neither/max(total_neither,1):.1%} of 'neither')")
