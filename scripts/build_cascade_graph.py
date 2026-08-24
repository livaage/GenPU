"""J2 — build the COMPLETE per-shard particle graph from the RAW source.

WHY NOT A STAGE2 SIDECAR (measured 2026-08-24, 150 events):
`preprocessing.visible_mask = (n_tracker_hits > 0) | (n_calo_hits > 0)` keeps only particles that
left a hit somewhere — 6.47M of 8.54M raw particles on shard 0 (24.2% dropped), and 2,586 of 10,000
events vanish entirely. The dropped particles are NOT leaves: 16.1% of visible secondaries have an
INVISIBLE direct parent, and 48.6% of chains to the primary cross at least one invisible particle.
So a parent->child graph built on stage2 is broken at 1 in 6 edges and silently skips a generation
in half of all root walks — it would look complete and not be. The cascade generator (and the calo
re-attribution, which must walk to the calo-incident ancestor) therefore need the RAW particle set,
where every node exists.

WHAT THIS WRITES, per shard:
  shard_XXXX_graph.npz  — one row per RAW particle (visible or not)
      raw:      event_id particle_id parent_id primary vertex_primary pdg_id charge mass energy
                vx vy vz time px py pz perigee_d0 perigee_z0
      derived:  n_tracker_hits n_calo_hits calo_energy_sum r_innermost r_outermost
                depth root_primary_id visible
  shard_XXXX_pids.npz   — particle_id aligned ROW-FOR-ROW with the existing stage2 npz, purely as a
                          join key from existing slices into the graph above.

FIELD NOTES:
- `mass` is kept even though it is constant within each named pdg_class, because `pdg_class`
  collapses everything unusual into class 16 — 6% of particles, 9,561 distinct masses up to
  27.9 GeV (nuclei; Si-28). Mass is the only surviving carrier of nuclear identity there, and 49
  pdg_id values have non-constant mass, so a pdg->mass lookup would be lossy.
- `perigee_d0`/`perigee_z0` are the SOURCE's impact parameters. build_count_slice_stage2.py currently
  approximates d0 as vx*sin(phi) - vy*cos(phi); this is the real thing.
- Derived hit summaries are stored because recomputing them means re-reading the ~1.9 GB calo shard,
  and because `r_outermost` retires a whole bug class: stage2 hit order is ARBITRARY, not
  r-ascending (Spearman(index, r) = 0.01) — an assumption that already produced one wrong result.
- NOT AVAILABLE IN THE SOURCE: the Geant4 creation process (conversion / brem / hadronic / decay).
  The cascade generator must infer it from (parent pdg, daughter pdg set).

VALIDATION: for every event also present in stage2, the recomputed `particle_features` for visible
particles are asserted equal ROW-FOR-ROW to the existing file. That both guarantees the join key is
aligned and validates the 2026-08-24 preprocessing.py fix across all shards (it previously emitted
6 tracker-hit columns where the pipeline reads 5).
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from genpu.data import load_shard, build_event_index, explode_list_columns  # noqa: E402
from genpu.preprocessing import (process_event_vectorized, _empty_tracker,  # noqa: E402
                                 _empty_calo)

RAW_COLS = ["particle_id", "parent_id", "primary", "vertex_primary", "pdg_id", "charge", "mass",
            "energy", "vx", "vy", "vz", "time", "px", "py", "pz", "perigee_d0", "perigee_z0"]
MAX_DEPTH = 32


def chain(pid, parent, primary):
    """depth and root primary id for every particle in ONE event, on the COMPLETE node set."""
    pos = {int(p): k for k, p in enumerate(pid)}
    depth = np.full(len(pid), -1, np.int16)
    root = pid.astype(np.int64).copy()
    for k in range(len(pid)):
        if depth[k] >= 0:
            continue
        path = []
        cur = k
        for _ in range(MAX_DEPTH):
            if depth[cur] >= 0 or primary[cur]:
                break
            path.append(cur)
            nxt = pos.get(int(parent[cur]))
            if nxt is None or nxt == cur:
                break
            cur = nxt
        base_d = 0 if depth[cur] < 0 else int(depth[cur])
        base_r = int(pid[cur]) if depth[cur] < 0 else int(root[cur])
        if depth[cur] < 0:
            depth[cur] = 0
            root[cur] = base_r
        for i, node in enumerate(reversed(path)):
            depth[node] = base_d + i + 1
            root[node] = base_r
    return depth, root


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--preproc_dir", default="/scratch/gpfs/IOJALVO/lv7805/genpu_data/preprocessed")
    ap.add_argument("--out_dir", default="/scratch/gpfs/IOJALVO/lv7805/genpu_data/cascade_graph")
    ap.add_argument("--max_events", type=int, default=0,
                    help="smoke test: stop after N events (0 = all). Output is tagged _smoke so a "
                         "partial file can never be mistaken for a full shard.")
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)

    print(f"[shard {a.shard}] loading raw tables", flush=True)
    t, tr, ca = (load_shard("particles", a.shard), load_shard("tracker_hits", a.shard),
                 load_shard("calo_hits", a.shard))
    pi, ti, ci = build_event_index(t), build_event_index(tr), build_event_index(ca)

    s2p = Path(a.preproc_dir) / f"shard_{a.shard:04d}_stage2.npz"
    s2 = np.load(s2p)
    pf_ex, ev_ex = s2["particle_features"], s2["event_ids"]
    ex_rows = {int(e): np.where(ev_ex == e)[0] for e in np.unique(ev_ex)}
    print(f"[shard {a.shard}] stage2 {pf_ex.shape[0]:,} visible rows / {len(ex_rows):,} events", flush=True)

    cols = {c: [] for c in RAW_COLS}
    ev_l, ntrk_l, ncal_l, ce_l, ri_l, ro_l, dep_l, root_l, vis_l = ([] for _ in range(9))
    pid_by_event = {}
    n_checked = n_missing_s2 = 0

    ev_list = sorted(pi)[:a.max_events] if a.max_events else sorted(pi)
    for n_done, eid in enumerate(ev_list):
        p_ev = explode_list_columns(t, pi[eid])
        # the 2,586 events with no hit tables are REAL soft-QCD interactions, not errors —
        # process_shard feeds them the empty sentinels, so do the same (None is not accepted).
        t_ev = explode_list_columns(tr, ti[eid]) if eid in ti else _empty_tracker()
        c_ev = explode_list_columns(ca, ci[eid]) if eid in ci else _empty_calo()
        res = process_event_vectorized(p_ev, t_ev, c_ev)
        npart = len(res["particle_ids"])
        if npart == 0:
            continue

        vis = res["visible_mask"]
        # ---- validation + join key ----
        if eid in ex_rows:
            mine, theirs = res["particle_features"][vis], pf_ex[ex_rows[eid]]
            if mine.shape != theirs.shape or not np.allclose(mine, theirs, atol=1e-5, equal_nan=True):
                raise SystemExit(f"MISMATCH event {eid}: recomputed {mine.shape} vs stage2 "
                                 f"{theirs.shape} — refusing to write a misaligned join key")
            pid_by_event[int(eid)] = res["particle_ids"][vis].astype(np.int64)
            n_checked += 1
        else:
            n_missing_s2 += 1

        for c in RAW_COLS:
            cols[c].append(np.asarray(p_ev[c]))
        ev_l.append(np.full(npart, eid, np.int32))
        ntrk_l.append(res["n_tracker_hits"].astype(np.int32))
        ncal_l.append(res["n_calo_hits"].astype(np.int32))
        vis_l.append(vis.astype(bool))

        ce = np.zeros(npart, np.float32); ri = np.full(npart, np.nan, np.float32)
        ro = np.full(npart, np.nan, np.float32)
        for k in range(npart):
            ch = res["calo_hits"][k]
            if len(ch):                       # col 2 = log(contrib energy)
                ce[k] = np.exp(ch[:, 2].astype(np.float64)).sum()
            th = res["tracker_hits"][k]
            if len(th):                       # col 1 = r (5-col layout: layer_class, r, phi, z, time)
                ri[k] = th[:, 1].min(); ro[k] = th[:, 1].max()
        ce_l.append(ce); ri_l.append(ri); ro_l.append(ro)

        d, r = chain(res["particle_ids"], np.asarray(p_ev["parent_id"]).astype(np.int64),
                     np.asarray(p_ev["primary"]) > 0.5)
        dep_l.append(d); root_l.append(r)
        if n_done % 1000 == 0:
            print(f"  event {n_done}/{len(ev_list)}", flush=True)

    g = {c: np.concatenate(cols[c]) for c in RAW_COLS}
    g.update(event_id=np.concatenate(ev_l), n_tracker_hits=np.concatenate(ntrk_l),
             n_calo_hits=np.concatenate(ncal_l), calo_energy_sum=np.concatenate(ce_l),
             r_innermost=np.concatenate(ri_l), r_outermost=np.concatenate(ro_l),
             depth=np.concatenate(dep_l), root_primary_id=np.concatenate(root_l),
             visible=np.concatenate(vis_l))
    sfx = "_smoke" if a.max_events else ""
    np.savez_compressed(out / f"shard_{a.shard:04d}_graph{sfx}.npz", **g)

    # join key in EXACT stage2 row order — only meaningful on a FULL pass, since a partial run
    # would leave un-touched rows at 0 and silently look like a valid key.
    if not a.max_events:
        pids = np.zeros(len(pf_ex), np.int64)
        for e, rows in ex_rows.items():
            pids[rows] = pid_by_event[e]
        np.savez_compressed(out / f"shard_{a.shard:04d}_pids.npz",
                            particle_ids=pids, event_ids=ev_ex)
    else:
        print("  (smoke run: join key NOT written)")

    n = len(g["particle_id"])
    print(f"\n[shard {a.shard}] wrote {n:,} raw particles  visible {g['visible'].mean():.3f}")
    print(f"  events validated against stage2: {n_checked:,}   raw-only (no stage2): {n_missing_s2:,}")
    hw = np.bincount(np.clip(g["depth"], 0, 8), weights=g["n_tracker_hits"], minlength=9)
    print("  tracker-hit fraction by depth: " +
          "  ".join(f"{i}:{v/hw.sum():.3f}" for i, v in enumerate(hw)))


if __name__ == "__main__":
    main()
