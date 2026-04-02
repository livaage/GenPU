#!/usr/bin/env python
"""Diagnostic script: check event_id alignment across particles, tracker_hits, calo_hits tables.

Hypothesis under test
---------------------
Row index N in the particles table does NOT necessarily correspond to row index N
in tracker_hits / calo_hits.  Each table stores events independently, so the only
reliable join key is the event_id column value, not the row position.

If this hypothesis is correct, explode_list_columns(table, event_idx=0) pulls row 0
from all three tables, but those rows may belong to *different* events, causing
get_particle_hit_map to find zero matching hits.

Checks performed
----------------
1. Print event_id for rows 0–4 in each table — reveals whether the tables share
   the same ordering.
2. Compare event_ids at each row position across all three tables — quantifies how
   often row-index alignment would accidentally be correct.
3. For row 0 of the particles table, print the particle_ids present and then look
   up tracker_hits rows whose event_id matches that particle row's event_id (not
   just row 0 of tracker_hits) — confirms whether hits exist when the join is done
   by event_id value.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure the genpu package is importable regardless of install state.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pyarrow.compute as pc

from genpu.data import load_from_hf, explode_list_columns

CACHE_DIR = os.environ.get("HF_HOME", None)
N_PREVIEW = 5  # number of rows to inspect


def load_tables():
    print("Loading tables from HuggingFace (this may use the local cache) …")
    tables = {}
    for name in ("particles", "tracker_hits", "calo_hits"):
        print(f"  Loading {name} …", flush=True)
        tables[name] = load_from_hf(name, split="train", cache_dir=CACHE_DIR)
        print(f"    -> {tables[name].num_rows} rows, columns: {tables[name].column_names}")
    return tables


# ---------------------------------------------------------------------------
# CHECK 1 — print the first N event_ids from each table
# ---------------------------------------------------------------------------
def check_first_event_ids(tables: dict, n: int = N_PREVIEW) -> None:
    print(f"\n{'='*60}")
    print(f"CHECK 1: event_id values for the first {n} rows of each table")
    print("='*60")

    for name, tbl in tables.items():
        ids = [tbl.column("event_id")[i].as_py() for i in range(min(n, tbl.num_rows))]
        print(f"  {name:20s}: {ids}")


# ---------------------------------------------------------------------------
# CHECK 2 — compare event_ids at the same row index across tables
# ---------------------------------------------------------------------------
def check_row_alignment(tables: dict, n: int = N_PREVIEW) -> None:
    print(f"\n{'='*60}")
    print(f"CHECK 2: Do event_ids match at the same row index? (first {n} rows)")
    print("='*60")

    names = list(tables.keys())
    n_check = min(n, *(tbl.num_rows for tbl in tables.values()))

    mismatches = 0
    for i in range(n_check):
        row_ids = {name: tables[name].column("event_id")[i].as_py() for name in names}
        values = list(row_ids.values())
        aligned = all(v == values[0] for v in values)
        status = "OK   " if aligned else "MISMATCH"
        print(f"  row {i}: {row_ids}  [{status}]")
        if not aligned:
            mismatches += 1

    if mismatches == 0:
        print(f"\n  -> All {n_check} rows have matching event_ids by row index.")
        print("     Row-index alignment appears accidental or deliberate — inspect further.")
    else:
        print(f"\n  -> {mismatches}/{n_check} rows have MISMATCHED event_ids.")
        print("     Root cause confirmed: tables are NOT aligned by row index.")


# ---------------------------------------------------------------------------
# CHECK 3 — find tracker hits for event 0 particles using event_id join
# ---------------------------------------------------------------------------
def check_particle_hit_join(tables: dict) -> None:
    print(f"\n{'='*60}")
    print("CHECK 3: particle_ids in row-0 of particles, then join tracker_hits by event_id")
    print("='*60")

    particles_tbl = tables["particles"]
    tracker_tbl = tables["tracker_hits"]

    # Row 0 of the particles table.
    event_data = explode_list_columns(particles_tbl, event_idx=0)

    particles_event_id = particles_tbl.column("event_id")[0].as_py()
    particle_ids = event_data.get("particle_id", [])

    print(f"\n  Particles row 0 — event_id : {particles_event_id}")
    print(f"  Number of particles        : {len(particle_ids)}")
    print(f"  First 10 particle_ids      : {list(particle_ids[:10])}")

    # Locate the tracker_hits row whose event_id equals particles_event_id.
    tracker_event_ids = tracker_tbl.column("event_id")
    match_mask = pc.equal(tracker_event_ids, particles_event_id)
    matching_indices = [i for i, m in enumerate(match_mask.to_pylist()) if m]

    if not matching_indices:
        print(f"\n  [!] No tracker_hits row found with event_id == {particles_event_id}.")
        print("      The event may not exist in the tracker_hits table, or the id type differs.")
        # Show what event_id row 0 of tracker_hits actually holds.
        print(f"      tracker_hits row 0 event_id: {tracker_tbl.column('event_id')[0].as_py()}")
        return

    tracker_row_idx = matching_indices[0]
    print(f"\n  tracker_hits row with matching event_id: row index {tracker_row_idx}")

    tracker_data = explode_list_columns(tracker_tbl, event_idx=tracker_row_idx)
    tracker_pids = tracker_data.get("particle_id", [])
    print(f"  Number of tracker hits in that event   : {len(tracker_pids)}")

    import numpy as np
    particle_ids_set = set(int(p) for p in particle_ids)
    tracker_pids_arr = np.asarray(tracker_pids, dtype=int)
    overlap = particle_ids_set & set(tracker_pids_arr.tolist())

    print(f"  Unique particle_ids in tracker_hits    : {len(set(tracker_pids_arr.tolist()))}")
    print(f"  Overlap with particles table           : {len(overlap)} particle(s)")

    if overlap:
        print(f"  Sample overlapping particle_ids        : {sorted(overlap)[:10]}")
        print("\n  -> Hits ARE found when joining by event_id value.")
        print("     Confirm: the bug is using row index instead of event_id for the join.")
    else:
        print("\n  [!] Still no overlap even after event_id join.")
        print("      Possible secondary issues: particle_id encoding mismatch, wrong column name,")
        print("      or the particles and tracker_hits use different id namespaces.")

    # Bonus: compare what row 0 of tracker_hits gives (the broken path)
    tracker_data_row0 = explode_list_columns(tracker_tbl, event_idx=0)
    tracker_pids_row0 = tracker_data_row0.get("particle_id", [])
    overlap_row0 = particle_ids_set & set(int(p) for p in tracker_pids_row0)
    print(f"\n  [Baseline] If we (wrongly) use tracker_hits row 0:")
    print(f"    tracker_hits row 0 event_id           : {tracker_tbl.column('event_id')[0].as_py()}")
    print(f"    Number of tracker hits                : {len(tracker_pids_row0)}")
    print(f"    Overlap with particles row 0          : {len(overlap_row0)}")
    if len(overlap_row0) == 0:
        print("    -> Zero hits, reproducing the reported bug.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("GenPU — event alignment diagnostic")
    print(f"HF_HOME / cache_dir: {CACHE_DIR}")

    tables = load_tables()

    check_first_event_ids(tables)
    check_row_alignment(tables)
    check_particle_hit_join(tables)

    print(f"\n{'='*60}")
    print("Diagnostic complete.")


if __name__ == "__main__":
    main()
