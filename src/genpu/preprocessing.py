"""Vectorized preprocessing: build per-particle training data from Arrow shards.

The key bottleneck in the old get_particle_hit_map() was Python loops over
every calo hit checking `pid in contrib_particle_ids`. This module replaces
that with vectorized numpy operations:

1. Tracker hits: grouped by particle_id using np.searchsorted (O(n log n))
2. Calo hits: build a flat (calo_hit_idx, particle_id) table from the
   variable-length contrib_particle_ids lists, then group by particle_id.

Output format: one .npz file per shard, containing flat arrays that can be
directly loaded into PyTorch datasets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from tqdm.auto import tqdm

from genpu.data import (
    load_shard,
    explode_list_columns,
    build_event_index,
    compute_eta_phi,
)
from .detector_geometry import hits_to_layer_class


# PDG class mapping: group ~4000 PDG codes into a small set of classes
# for the model to predict
PDG_TO_CLASS = {
    11: 0, -11: 1,       # e-, e+
    22: 2,                # photon
    211: 3, -211: 4,      # pi+, pi-
    321: 5, -321: 6,      # K+, K-
    2212: 7, -2212: 8,    # proton, antiproton
    2112: 9, -2112: 10,   # neutron, antineutron
    13: 11, -13: 12,      # mu-, mu+
    130: 13,              # K0L
    310: 14,              # K0S
    111: 15,              # pi0
}
N_PDG_CLASSES = 17  # 0-15 + 16 for "other"
PDG_OTHER = 16


def pdg_to_class(pdg_ids: np.ndarray) -> np.ndarray:
    """Map PDG IDs to class indices."""
    result = np.full(len(pdg_ids), PDG_OTHER, dtype=np.int8)
    for pdg, cls in PDG_TO_CLASS.items():
        result[pdg_ids == pdg] = cls
    return result


def _empty_tracker() -> dict[str, np.ndarray]:
    """Return an empty tracker event dict."""
    return {
        "particle_id": np.array([], dtype=np.int64),
        "x": np.array([], dtype=np.float32),
        "y": np.array([], dtype=np.float32),
        "z": np.array([], dtype=np.float32),
        "time": np.array([], dtype=np.float32),
        "volume_id": np.array([], dtype=np.float32),
        "layer_id": np.array([], dtype=np.float32),
    }


def _empty_calo() -> dict[str, np.ndarray]:
    """Return an empty calo event dict."""
    return {
        "x": np.array([], dtype=np.float32),
        "y": np.array([], dtype=np.float32),
        "z": np.array([], dtype=np.float32),
        "total_energy": np.array([], dtype=np.float32),
        "detector": np.array([], dtype=np.float32),
        "contrib_particle_ids": [],
        "contrib_energies": [],
    }


def process_event_vectorized(
    p_ev: dict[str, np.ndarray],
    t_ev: dict[str, np.ndarray],
    c_ev: dict[str, np.ndarray],
) -> dict:
    """Process one event into per-particle training arrays, vectorized.

    Returns dict with:
        particle_features: (N_particles, 6) — log_pt, eta, phi, pdg_class, charge, mass
        particle_aux: (N_particles, 6) — primary, parent_id, vx, vy, vz, energy
        n_tracker_hits: (N_particles,) — int
        n_calo_hits: (N_particles,) — int
        tracker_hits: list of (n_hits, 6) arrays — r, phi, z, time, volume_id, layer_id
        calo_hits: list of (n_hits, 5) arrays — eta, phi, log_energy, contrib_energy_frac, detector
        visible_mask: (N_particles,) — bool, True if particle has any hits
    """
    particle_ids = p_ev["particle_id"].astype(np.int64)
    n_particles = len(particle_ids)

    # ── Particle kinematics ──
    px = p_ev["px"].astype(np.float32)
    py = p_ev["py"].astype(np.float32)
    pz = p_ev["pz"].astype(np.float32)
    pt, eta, phi = compute_eta_phi(px, py, pz)
    log_pt = np.log(np.clip(pt, 1e-6, None))

    pdg_class = pdg_to_class(p_ev["pdg_id"].astype(np.int64))
    charge = p_ev["charge"].astype(np.float32) if "charge" in p_ev else np.zeros(n_particles, dtype=np.float32)
    mass = p_ev["mass"].astype(np.float32) if "mass" in p_ev else np.zeros(n_particles, dtype=np.float32)

    particle_features = np.stack([log_pt, eta, phi, pdg_class.astype(np.float32), charge, mass], axis=1)

    # ── Auxiliary particle info (for future use) ──
    primary = p_ev["primary"].astype(np.float32) if "primary" in p_ev else np.zeros(n_particles, dtype=np.float32)
    parent_id = p_ev["parent_id"].astype(np.float32) if "parent_id" in p_ev else np.full(n_particles, -1, dtype=np.float32)
    vx = p_ev["vx"].astype(np.float32) if "vx" in p_ev else np.zeros(n_particles, dtype=np.float32)
    vy = p_ev["vy"].astype(np.float32) if "vy" in p_ev else np.zeros(n_particles, dtype=np.float32)
    vz = p_ev["vz"].astype(np.float32) if "vz" in p_ev else np.zeros(n_particles, dtype=np.float32)
    energy = p_ev["energy"].astype(np.float32) if "energy" in p_ev else np.zeros(n_particles, dtype=np.float32)

    particle_aux = np.stack([primary, parent_id, vx, vy, vz, energy], axis=1)

    # ── Tracker hits: group by particle_id using sorting ──
    trk_pid = t_ev["particle_id"].astype(np.int64)
    trk_x = t_ev["x"].astype(np.float32)
    trk_y = t_ev["y"].astype(np.float32)
    trk_z = t_ev["z"].astype(np.float32)
    trk_time = t_ev["time"].astype(np.float32) if "time" in t_ev else np.zeros_like(trk_x)
    trk_vol = t_ev["volume_id"].astype(np.float32)
    trk_layer = t_ev["layer_id"].astype(np.float32)

    trk_r = np.sqrt(trk_x**2 + trk_y**2)
    trk_phi = np.arctan2(trk_y, trk_x)

    # Stack into (n_tracker_hits, 5): layer_class, r, phi, z, time.
    # NOTE 2026-08-24: this file previously emitted 6 columns (r, phi, z, time, volume_id,
    # layer_id), which does NOT match the stage2 npz the whole training pipeline reads
    # (5 columns, col 0 = layer_class 0..47, verified against shard_0000). Every slice builder
    # indexes TH_LAYER=0, TH_R=1, ... so a re-run with the old columns would have silently
    # produced unusable data. (volume, layer) -> layer_class is the documented collapse in
    # detector_geometry.hits_to_layer_class, which this module was never calling.
    trk_layer_class = hits_to_layer_class(
        trk_vol.astype(np.int64), trk_layer.astype(np.int64)).astype(np.float32)
    trk_features = np.stack([trk_layer_class, trk_r, trk_phi, trk_z, trk_time], axis=1)

    # Sort tracker hits by particle_id for fast grouping
    # STABLE: np.argsort defaults to quicksort, which is NOT stable, so ties (hits sharing a
    # particle_id) come out in arbitrary order and the within-particle row order is not
    # reproducible across runs. Found 2026-08-24 when a calo sidecar failed its alignment assert
    # while being identical as a SET. Existing shards were written with the unstable sort, so
    # consumers aligning to them must match by VALUE, not by row index.
    trk_sort = np.argsort(trk_pid, kind="stable")
    trk_pid_sorted = trk_pid[trk_sort]
    trk_features_sorted = trk_features[trk_sort]

    # ── Calo hits: flatten contrib_particle_ids for vectorized lookup ──
    calo_x = c_ev["x"].astype(np.float32)
    calo_y = c_ev["y"].astype(np.float32)
    calo_z = c_ev["z"].astype(np.float32)
    calo_total_e = c_ev["total_energy"].astype(np.float32)
    calo_det = c_ev["detector"].astype(np.float32)
    calo_contrib_pids = c_ev["contrib_particle_ids"]  # list of arrays
    calo_contrib_energies = c_ev["contrib_energies"]   # list of arrays

    calo_r_3d = np.sqrt(calo_x**2 + calo_y**2 + calo_z**2)
    calo_eta = np.arctanh(np.clip(calo_z / (calo_r_3d + 1e-10), -1 + 1e-7, 1 - 1e-7))
    calo_phi = np.arctan2(calo_y, calo_x)

    # Build flat (calo_hit_idx, particle_id, contrib_energy) table
    flat_calo_idx = []
    flat_calo_pid = []
    flat_calo_contrib_e = []
    for j in range(len(calo_x)):
        cpids = np.asarray(calo_contrib_pids[j], dtype=np.int64)
        cenergies = np.asarray(calo_contrib_energies[j], dtype=np.float32)
        n = len(cpids)
        if n > 0:
            flat_calo_idx.append(np.full(n, j, dtype=np.int32))
            flat_calo_pid.append(cpids)
            flat_calo_contrib_e.append(cenergies)

    if flat_calo_pid:
        flat_calo_idx = np.concatenate(flat_calo_idx)
        flat_calo_pid = np.concatenate(flat_calo_pid)
        flat_calo_contrib_e = np.concatenate(flat_calo_contrib_e)

        # Sort by particle_id
        calo_sort = np.argsort(flat_calo_pid, kind="stable")   # see the tracker note above
        flat_calo_pid_sorted = flat_calo_pid[calo_sort]
        flat_calo_idx_sorted = flat_calo_idx[calo_sort]
        flat_calo_contrib_e_sorted = flat_calo_contrib_e[calo_sort]
    else:
        flat_calo_pid_sorted = np.array([], dtype=np.int64)
        flat_calo_idx_sorted = np.array([], dtype=np.int32)
        flat_calo_contrib_e_sorted = np.array([], dtype=np.float32)

    # ── Assign hits to particles using searchsorted ──
    n_tracker_hits = np.zeros(n_particles, dtype=np.int32)
    n_calo_hits = np.zeros(n_particles, dtype=np.int32)
    tracker_hit_list = []
    calo_hit_list = []
    calo_rz_list = []

    for i, pid in enumerate(particle_ids):
        # Tracker: binary search in sorted array
        lo = np.searchsorted(trk_pid_sorted, pid, side="left")
        hi = np.searchsorted(trk_pid_sorted, pid, side="right")
        n_trk = hi - lo
        n_tracker_hits[i] = n_trk
        if n_trk > 0:
            tracker_hit_list.append(trk_features_sorted[lo:hi])
        else:
            tracker_hit_list.append(np.empty((0, 5), dtype=np.float32))

        # Calo: binary search in sorted flat table
        lo_c = np.searchsorted(flat_calo_pid_sorted, pid, side="left")
        hi_c = np.searchsorted(flat_calo_pid_sorted, pid, side="right")
        n_cal = hi_c - lo_c
        n_calo_hits[i] = n_cal
        if n_cal > 0:
            cidxs = flat_calo_idx_sorted[lo_c:hi_c]
            cenergies = flat_calo_contrib_e_sorted[lo_c:hi_c]
            # (eta, phi, log_contrib_energy, contrib_energy_fraction, detector)
            # (r, z) of the same cells, in the SAME row order as `hits`. Kept RAW rather than
            # reduced to a "depth" here: collapsing (x,y,z) -> (eta,phi) at this boundary is
            # exactly what lost the longitudinal dimension for months (see PIPELINE.md §2), so
            # store the coordinates and let downstream choose the depth convention.
            calo_rz_list.append(np.stack([
                np.hypot(calo_x[cidxs], calo_y[cidxs]), calo_z[cidxs]], axis=1).astype(np.float32))
            hits = np.stack([
                calo_eta[cidxs],
                calo_phi[cidxs],
                np.log(np.clip(cenergies, 1e-8, None)),
                cenergies / np.clip(calo_total_e[cidxs], 1e-8, None),
                calo_det[cidxs],
            ], axis=1)
            calo_hit_list.append(hits)
        else:
            calo_hit_list.append(np.empty((0, 5), dtype=np.float32))
            calo_rz_list.append(np.empty((0, 2), dtype=np.float32))

    visible_mask = (n_tracker_hits > 0) | (n_calo_hits > 0)

    return {
        "particle_features": particle_features.astype(np.float32),
        "particle_aux": particle_aux.astype(np.float32),
        "n_tracker_hits": n_tracker_hits,
        "n_calo_hits": n_calo_hits,
        "tracker_hits": tracker_hit_list,
        "calo_hits": calo_hit_list,
        # (r, z) per calo contribution, row-aligned with `calo_hits`. ADDITIVE — no saved file
        # shape changes; scripts/build_calo_depth.py sidecars it onto existing stage2 shards.
        "calo_rz": calo_rz_list,
        "visible_mask": visible_mask,
        "particle_ids": particle_ids,
    }


def process_shard(
    shard_idx: int,
    output_dir: str | Path,
    cache_dir: Optional[str | Path] = None,
) -> dict:
    """Process one Arrow shard into preprocessed .npz files.

    Storage format: flat arrays with offsets (no padding, no wasted space).
    Padding to fixed sizes happens at batch time in the DataLoader.

    Produces two files:
        {output_dir}/shard_{shard_idx:04d}_stage2.npz — per-particle data:
            particle_features: (N_visible, 6) float32
            tracker_hits_flat: (total_trk_hits, 6) float32 — r,phi,z,time,vol,layer
            tracker_offsets: (N_visible+1,) int32 — particle i owns hits [off[i]:off[i+1]]
            calo_hits_flat: (total_calo_hits, 5) float32 — eta,phi,log_e,frac,det
            calo_offsets: (N_visible+1,) int32
            event_ids: (N_visible,) int32

        {output_dir}/shard_{shard_idx:04d}_stage1.npz — per-event particle sets:
            particle_features: (total_visible, 6) float32
            offsets: (N_events+1,) int32 — event i owns particles [off[i]:off[i+1]]
            event_ids: (N_events,) int32

    Returns summary statistics dict.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the three tables for this shard
    p_table = load_shard("particles", shard_idx, cache_dir)
    t_table = load_shard("tracker_hits", shard_idx, cache_dir)
    c_table = load_shard("calo_hits", shard_idx, cache_dir)

    p_idx = build_event_index(p_table)
    t_idx = build_event_index(t_table)
    c_idx = build_event_index(c_table)

    # Process ALL particle events, not just common ones.
    # Events without tracker/calo are real soft QCD interactions (~26%)
    # that produce zero detector hits — important for correct pileup rates.
    all_p_eids = sorted(p_idx.keys())

    if not all_p_eids:
        print(f"  Shard {shard_idx}: no particle events, skipping")
        return {"shard": shard_idx, "n_events": 0, "n_visible_particles": 0}

    # Stage 2 accumulators (per visible particle)
    all_particle_features = []
    all_particle_aux = []
    all_particle_ids = []
    all_tracker_hits = []    # list of (n_hits, 5) arrays
    all_calo_hits = []       # list of (n_hits, 5) arrays
    all_event_ids_s2 = []

    # Stage 1 accumulators (per event — only events with visible particles)
    event_visible_sets = []
    event_visible_aux = []
    event_ids_s1 = []

    stats = {"trk_max": 0, "cal_max": 0, "n_null_events": 0}

    for eid in tqdm(all_p_eids, desc=f"Shard {shard_idx}", leave=False):
        p_ev = explode_list_columns(p_table, p_idx[eid])

        # If tracker/calo data exists for this event, use it; otherwise empty
        has_tracker = eid in t_idx
        has_calo = eid in c_idx

        if has_tracker and has_calo:
            t_ev = explode_list_columns(t_table, t_idx[eid])
            c_ev = explode_list_columns(c_table, c_idx[eid])
        elif has_tracker:
            t_ev = explode_list_columns(t_table, t_idx[eid])
            c_ev = _empty_calo()
        elif has_calo:
            t_ev = _empty_tracker()
            c_ev = explode_list_columns(c_table, c_idx[eid])
        else:
            # No detector hits at all — null event (~26% of min-bias).
            # Skip entirely; the null fraction is stored in the summary
            # and applied at generation time.
            stats["n_null_events"] += 1
            continue

        result = process_event_vectorized(p_ev, t_ev, c_ev)

        vis = result["visible_mask"]
        n_vis = int(vis.sum())
        if n_vis == 0:
            continue

        vis_idx = np.where(vis)[0]
        vis_features = result["particle_features"][vis_idx]
        vis_aux = result["particle_aux"][vis_idx]

        # Stage 1: visible particle set for this event
        event_visible_sets.append(vis_features)
        event_visible_aux.append(vis_aux)
        event_ids_s1.append(eid)

        # Stats
        vis_n_trk = result["n_tracker_hits"][vis_idx]
        vis_n_cal = result["n_calo_hits"][vis_idx]
        if len(vis_n_trk) > 0:
            stats["trk_max"] = max(stats["trk_max"], int(vis_n_trk.max()))
        if len(vis_n_cal) > 0:
            stats["cal_max"] = max(stats["cal_max"], int(vis_n_cal.max()))

        # Stage 2: per-particle flat hits
        all_particle_features.append(vis_features)
        all_particle_aux.append(vis_aux)
        # own particle_id, so (event_id, particle_id) keys the parent->child graph downstream.
        # particle_aux carries parent_id but WITHOUT this the graph cannot be rebuilt at all.
        all_particle_ids.append(result["particle_ids"][vis_idx].astype(np.int64))
        all_event_ids_s2.append(np.full(n_vis, eid, dtype=np.int32))

        for j in vis_idx:
            all_tracker_hits.append(result["tracker_hits"][j])  # (n, 5) or (0, 5)
            all_calo_hits.append(result["calo_hits"][j])        # (n, 5) or (0, 5)

    # ── Save Stage 2: flat with offsets ──
    if all_particle_features:
        pf = np.concatenate(all_particle_features)
        paux = np.concatenate(all_particle_aux)
        eids = np.concatenate(all_event_ids_s2)

        # Build tracker offset array
        trk_lengths = np.array([h.shape[0] for h in all_tracker_hits], dtype=np.int32)
        trk_offsets = np.concatenate([[0], np.cumsum(trk_lengths)]).astype(np.int32)
        trk_flat = np.concatenate(all_tracker_hits) if trk_offsets[-1] > 0 else np.empty((0, 5), dtype=np.float32)

        # Build calo offset array
        cal_lengths = np.array([h.shape[0] for h in all_calo_hits], dtype=np.int32)
        cal_offsets = np.concatenate([[0], np.cumsum(cal_lengths)]).astype(np.int32)
        cal_flat = np.concatenate(all_calo_hits) if cal_offsets[-1] > 0 else np.empty((0, 5), dtype=np.float32)

        s2_path = output_dir / f"shard_{shard_idx:04d}_stage2.npz"
        np.savez_compressed(
            s2_path,
            particle_features=pf,
            particle_aux=paux,
            tracker_hits_flat=trk_flat,
            tracker_offsets=trk_offsets,
            calo_hits_flat=cal_flat,
            calo_offsets=cal_offsets,
            event_ids=eids,
            particle_ids=np.concatenate(all_particle_ids),
        )

    # ── Save Stage 1: flat with offsets ──
    if event_visible_sets:
        s1_offsets = np.concatenate([[0], np.cumsum([len(s) for s in event_visible_sets])]).astype(np.int32)
        s1_path = output_dir / f"shard_{shard_idx:04d}_stage1.npz"
        np.savez_compressed(
            s1_path,
            particle_features=np.concatenate(event_visible_sets),
            particle_aux=np.concatenate(event_visible_aux),
            offsets=s1_offsets,
            event_ids=np.array(event_ids_s1, dtype=np.int32),
        )

    n_visible_total = sum(len(s) for s in event_visible_sets)
    n_with_hits = len(all_p_eids) - stats["n_null_events"]
    summary = {
        "shard": shard_idx,
        "n_events_total": len(all_p_eids),
        "n_events_with_hits": n_with_hits,
        "n_null_events": stats["n_null_events"],
        "null_fraction": stats["n_null_events"] / max(len(all_p_eids), 1),
        "n_visible_particles": n_visible_total,
        "trk_max_hits": stats["trk_max"],
        "cal_max_hits": stats["cal_max"],
    }
    return summary
