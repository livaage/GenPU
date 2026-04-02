"""Data loading utilities for the ColliderML dataset.

Reads directly from the HuggingFace datasets Arrow cache on disk.
No network access or HF library required at load time.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc

DATASET_HF_ID = "CERN/ColliderML-Release-1"

# Default HF cache root (set via GENPU_DATA_DIR or HF_HOME env var)
DEFAULT_CACHE_DIR = Path(
    os.environ.get(
        "GENPU_DATA_DIR",
        os.environ.get("HF_HOME", "/scratch/gpfs/IOJALVO/genpu_cache"),
    )
)

# Subset names for single min-bias (pileup-only, no overlay)
PU0_SUBSETS = {
    "particles": "pileup_only_pu0_particles",
    "tracker_hits": "pileup_only_pu0_tracker_hits",
    "calo_hits": "pileup_only_pu0_calo_hits",
    "tracks": "pileup_only_pu0_tracks",
}

# Validation subsets (pu200 overlaid)
PU200_PROCESSES = ["ttbar", "zmumu", "zee", "dihiggs", "ggf"]


def _find_arrow_shards(subset: str, cache_dir: Optional[str | Path] = None) -> list[str]:
    """Locate Arrow shard files for a subset in the HF cache."""
    cache_dir = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    config = PU0_SUBSETS.get(subset, subset)

    # HF cache layout: {cache}/CERN___collider_ml-release-1/{config}/0.0.0/{hash}/*.arrow
    pattern = str(cache_dir / f"CERN___collider_ml-release-1/{config}/0.0.0/*/*.arrow")
    shards = sorted(glob.glob(pattern))
    if not shards:
        raise FileNotFoundError(
            f"No Arrow shards found for '{subset}'. Looked in: {pattern}\n"
            f"Run the HF download first on a node with internet access."
        )
    return shards


def _read_arrow_shard(path: str) -> pa.Table:
    """Read a single HF-format Arrow shard (IPC stream format)."""
    return ipc.open_stream(pa.memory_map(path, "r")).read_all()


def load_shard(subset: str, shard_idx: int = 0, cache_dir: Optional[str | Path] = None) -> pa.Table:
    """Load a single Arrow shard for a subset.

    Each shard contains ~10k events. Use this to avoid loading the full dataset.
    """
    shards = _find_arrow_shards(subset, cache_dir)
    if shard_idx >= len(shards):
        raise IndexError(f"Shard index {shard_idx} out of range (have {len(shards)} shards)")
    return _read_arrow_shard(shards[shard_idx])


def load_table(subset: str, cache_dir: Optional[str | Path] = None) -> pa.Table:
    """Load a full ColliderML subset by concatenating all Arrow shards.

    Warning: this loads the entire subset into memory. For per-event access,
    prefer load_shard() or load_event().
    """
    shards = _find_arrow_shards(subset, cache_dir)
    tables = [_read_arrow_shard(s) for s in shards]
    return pa.concat_tables(tables)


def load_event(
    subset: str,
    event_id: int,
    cache_dir: Optional[str | Path] = None,
    _index: Optional[dict] = None,
) -> dict[str, np.ndarray]:
    """Load a single event by event_id, scanning shards as needed.

    Parameters
    ----------
    subset : str
        One of 'particles', 'tracker_hits', 'calo_hits', 'tracks'.
    event_id : int
        The event_id value to find.
    cache_dir : str or Path, optional
        Cache directory override.
    _index : dict, optional
        Pre-built index from build_shard_index() for fast lookup.

    Returns
    -------
    dict[str, np.ndarray]
        Column name -> numpy array for that event.
    """
    if _index is not None:
        shard_path, row_idx = _index[event_id]
        table = _read_arrow_shard(shard_path)
        return explode_list_columns(table, row_idx)

    # Linear scan through shards
    for shard_path in _find_arrow_shards(subset, cache_dir):
        table = _read_arrow_shard(shard_path)
        event_ids = table.column("event_id")
        for i in range(table.num_rows):
            if event_ids[i].as_py() == event_id:
                return explode_list_columns(table, i)

    raise KeyError(f"event_id {event_id} not found in {subset}")


def build_shard_index(
    subset: str,
    cache_dir: Optional[str | Path] = None,
) -> dict[int, tuple[str, int]]:
    """Build an index mapping event_id -> (shard_path, row_index).

    Call once, then pass to load_event(_index=...) for fast repeated lookups.
    """
    index = {}
    for shard_path in _find_arrow_shards(subset, cache_dir):
        table = _read_arrow_shard(shard_path)
        event_ids = table.column("event_id")
        for i in range(table.num_rows):
            index[event_ids[i].as_py()] = (shard_path, i)
    return index


def get_common_event_ids(
    cache_dir: Optional[str | Path] = None,
    subsets: tuple[str, ...] = ("particles", "tracker_hits", "calo_hits"),
) -> list[int]:
    """Return sorted event_ids present in all given subsets."""
    id_sets = []
    for subset in subsets:
        ids = set()
        for shard_path in _find_arrow_shards(subset, cache_dir):
            table = _read_arrow_shard(shard_path)
            event_col = table.column("event_id")
            ids.update(event_col.to_pylist())
        id_sets.append(ids)
    common = id_sets[0]
    for s in id_sets[1:]:
        common &= s
    return sorted(common)


def explode_list_columns(table, event_idx: int = 0) -> dict[str, np.ndarray]:
    """Extract arrays for a single event from a list-column Arrow table.

    ColliderML stores each event as one row where columns like 'px', 'py' etc.
    are list<float32>. This function extracts the arrays for one event as numpy.

    Parameters
    ----------
    table : pyarrow.Table
        Table with list-type columns and an 'event_id' column.
    event_idx : int
        Row index to extract (not event_id value).

    Returns
    -------
    dict[str, np.ndarray]
        Column name -> numpy array for that event.
    """
    result = {}
    for col_name in table.column_names:
        col = table.column(col_name)
        val = col[event_idx].as_py()
        if isinstance(val, list):
            # Nested lists (e.g. contrib_particle_ids) stay as lists of arrays
            if len(val) > 0 and isinstance(val[0], list):
                result[col_name] = [np.array(v) for v in val]
            else:
                result[col_name] = np.array(val)
        else:
            result[col_name] = val
    return result


def get_num_events(table) -> int:
    """Return the number of events (rows) in a table."""
    return table.num_rows


def build_event_index(table) -> dict[int, int]:
    """Build a mapping from event_id value to row index within a table.

    Returns
    -------
    dict[int, int]
        event_id -> row index in the table.
    """
    event_ids = table.column("event_id")
    return {int(event_ids[i].as_py()): i for i in range(table.num_rows)}


def get_particle_hit_map(
    particles: dict[str, np.ndarray],
    tracker_hits: dict[str, np.ndarray],
    calo_hits: dict[str, np.ndarray],
) -> dict[int, dict]:
    """Build a mapping from particle_id to its detector response.

    For each truth particle, collects:
    - Tracker hits where tracker_hits['particle_id'] == pid
    - Calorimeter hits where pid is in calo_hits['contrib_particle_ids']

    Parameters
    ----------
    particles : dict
        Exploded particle arrays for one event.
    tracker_hits : dict
        Exploded tracker hit arrays for one event.
    calo_hits : dict
        Exploded calorimeter hit arrays for one event.

    Returns
    -------
    dict[int, dict]
        pid -> {'kinematics': {...}, 'tracker_hits': [...], 'calo_hits': [...]}
    """
    particle_ids = particles["particle_id"]

    # Index tracker hits by particle_id
    tracker_pid = tracker_hits["particle_id"]

    # Index calo hits: each calo hit can have multiple contributing particles
    calo_contrib_pids = calo_hits["contrib_particle_ids"]

    result = {}
    for i, pid in enumerate(particle_ids):
        pid_int = int(pid)

        # Kinematics
        kin = {}
        for k in ["px", "py", "pz", "energy", "pdg_id"]:
            if k in particles:
                kin[k] = float(particles[k][i])

        # Tracker hits for this particle
        trk_mask = tracker_pid == pid_int
        trk_hits = []
        if trk_mask.any():
            for idx in np.where(trk_mask)[0]:
                hit = {}
                for k in ["x", "y", "z", "time", "volume_id", "layer_id", "surface_id"]:
                    if k in tracker_hits:
                        hit[k] = float(tracker_hits[k][idx])
                trk_hits.append(hit)

        # Calo hits where this particle contributed
        cal_hits = []
        for j, cpids in enumerate(calo_contrib_pids):
            if pid_int in cpids:
                hit = {}
                for k in ["x", "y", "z", "total_energy", "detector"]:
                    if k in calo_hits:
                        hit[k] = float(calo_hits[k][j])
                # Find this particle's energy contribution
                contrib_idx = np.where(cpids == pid_int)[0]
                if len(contrib_idx) > 0 and "contrib_energies" in calo_hits:
                    hit["contrib_energy"] = float(
                        calo_hits["contrib_energies"][j][contrib_idx[0]]
                    )
                cal_hits.append(hit)

        result[pid_int] = {
            "kinematics": kin,
            "tracker_hits": trk_hits,
            "calo_hits": cal_hits,
        }

    return result


def compute_eta_phi(px: np.ndarray, py: np.ndarray, pz: np.ndarray):
    """Compute pseudorapidity (eta) and azimuthal angle (phi) from momentum components."""
    pt = np.sqrt(px**2 + py**2)
    p = np.sqrt(px**2 + py**2 + pz**2)
    # eta = -ln(tan(theta/2)) = arctanh(pz/p)
    with np.errstate(divide="ignore", invalid="ignore"):
        eta = np.arctanh(np.clip(pz / (p + 1e-10), -1 + 1e-7, 1 - 1e-7))
    phi = np.arctan2(py, px)
    return pt, eta, phi
