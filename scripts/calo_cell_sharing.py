"""Do calo cells have MULTIPLE contributing particles, and are those contributors the same shower?

`calo_hits.contrib_particle_ids` is a LIST per cell, and preprocessing books the cell to EVERY
contributor. Two consequences with opposite fixes:

  WITHIN a shower  — a calo-incident particle and its own descendants (born inside the calo) both
                     touch the same cells. Re-attribution must DEDUPLICATE these, not sum. J3a
                     reported cells/shower 10.49 -> 20.72 by summing merged fragments' counts, so
                     that figure is an UPPER BOUND and this probe measures the real one.
  BETWEEN showers  — two independent incident particles overlapping in a dense event. That is
                     genuine superposition (M3's scatter_add), not double counting.

Distinguishes them by walking every contributor up to its calo-incident ancestor and asking whether
the contributors of a cell share one.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from genpu.data import load_shard, build_event_index, explode_list_columns  # noqa: E402
from genpu.calo_geom import load_front_face  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--n_events", type=int, default=400)
ap.add_argument("--out", default="/home/lv7805/genpu/plots/calo/metrics/cell_sharing.json")
a = ap.parse_args()
R, Z = load_front_face(None)

t, ca = load_shard("particles", a.shard), load_shard("calo_hits", a.shard)
pi, ci = build_event_index(t), build_event_index(ca)
eids = sorted(set(pi) & set(ci))[:a.n_events]

ncontrib = []          # contributors per cell
same_anc = 0; diff_anc = 0; single = 0
per_shower_sum = []    # naive summed cell count (what J3a did)
per_shower_uniq = []   # deduplicated cell count (the truth)
for eid in eids:
    p = explode_list_columns(t, pi[eid]); c = explode_list_columns(ca, ci[eid])
    P = p["particle_id"].astype(np.int64); PAR = p["parent_id"].astype(np.int64)
    vr = np.hypot(p["vx"], p["vy"]); vz = np.abs(p["vz"])
    inside = (vr >= R) | (vz >= Z)
    pos = {int(x): k for k, x in enumerate(P)}
    # calo-incident ancestor of every particle
    anc = {}
    for k in range(len(P)):
        cur = k
        for _ in range(32):
            if not inside[cur]:
                break
            nxt = pos.get(int(PAR[cur]))
            if nxt is None or nxt == cur:
                break
            cur = nxt
        anc[int(P[k])] = int(P[cur])
    cells = c["contrib_particle_ids"]
    shower_cells = {}
    for j, lst in enumerate(cells):
        ids = np.asarray(lst, np.int64)
        ncontrib.append(len(ids))
        if len(ids) == 1:
            single += 1
        else:
            A = {anc.get(int(x), int(x)) for x in ids}
            if len(A) == 1: same_anc += 1
            else: diff_anc += 1
        for x in ids:
            shower_cells.setdefault(anc.get(int(x), int(x)), []).append(j)
    for aid, js in shower_cells.items():
        per_shower_sum.append(len(js)); per_shower_uniq.append(len(set(js)))

nc = np.array(ncontrib)
tot = single + same_anc + diff_anc
print(f"events {len(eids)}   cells {len(nc):,}")
print(f"\ncontributors per cell: mean {nc.mean():.3f}  median {np.median(nc):.0f}  "
      f"p90 {np.percentile(nc,90):.0f}  max {nc.max()}")
print(f"  1 contributor            {single/tot:.4f}")
print(f"  >1, SAME calo ancestor   {same_anc/tot:.4f}   <- within-shower, must DEDUPLICATE")
print(f"  >1, DIFFERENT ancestors  {diff_anc/tot:.4f}   <- true superposition (M3 scatter_add)")
s, u = np.array(per_shower_sum, float), np.array(per_shower_uniq, float)
print(f"\nre-attributed shower cell count: naive SUM {s.mean():.2f}  "
      f"DEDUPLICATED {u.mean():.2f}   inflation {s.mean()/max(u.mean(),1e-9):.3f}x")
json.dump(dict(mean_contrib=float(nc.mean()), single=single/tot, same_anc=same_anc/tot,
               diff_anc=diff_anc/tot, sum_mean=float(s.mean()), uniq_mean=float(u.mean())),
          open(a.out, "w"), indent=2)
print(f"wrote {a.out}")
