"""Is `frac_near_floor` measuring physics, or the PARTITION?

preprocessing.py books, per (cell, particle) pair, that particle's CONTRIBUTION to the cell --
not the cell's total. calo_metrics.py then computes every energy feature over those contributions.
But a calorimeter reads out the CELL. A cell where three particles each deposited a third gives
three near-floor contributions instead of one comfortably-above-floor cell.

`frac_near_floor` has been the top gate discriminator in nearly every run since 2026-08-13, so if
the two differ materially, that thread has been partly chasing a partitioning artifact.

Compares, on REAL data only:
  contributions : every contrib_energy       (what the gate scores today)
  cells         : total_energy per cell      (what a detector reports)
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from genpu.data import load_shard, build_event_index, explode_list_columns  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--n_events", type=int, default=600)
ap.add_argument("--floor", type=float, default=5e-5, help="calo_metrics.py uses log(5e-5) + 0.5")
ap.add_argument("--out", default="/home/lv7805/genpu/plots/calo/metrics/floor_contrib_vs_cell.json")
a = ap.parse_args()
LF = np.log(a.floor) + 0.5

ca = load_shard("calo_hits", a.shard)
ci = build_event_index(ca)
contrib, cell = [], []
for eid in sorted(ci)[:a.n_events]:
    c = explode_list_columns(ca, ci[eid])
    tot = np.asarray(c["total_energy"], np.float64)
    cell.append(tot)
    for lst in c["contrib_energies"]:
        v = np.asarray(lst, np.float64)
        if len(v):
            contrib.append(v)
contrib = np.concatenate(contrib); cell = np.concatenate(cell)
lc, lk = np.log(np.clip(contrib, 1e-30, None)), np.log(np.clip(cell, 1e-30, None))

f_contrib = float((lc < LF).mean()); f_cell = float((lk < LF).mean())
print(f"events {a.n_events}   contributions {len(contrib):,}   distinct cells {len(cell):,}   "
      f"inflation {len(contrib)/len(cell):.3f}x")
print(f"\nfrac_near_floor  (logE < log({a.floor:g}) + 0.5)")
print(f"  over CONTRIBUTIONS (what the gate scores today) : {f_contrib:.4f}")
print(f"  over CELLS         (what a detector reports)    : {f_cell:.4f}")
print(f"  ratio {f_contrib/max(f_cell,1e-12):.3f}x   absolute difference {f_contrib-f_cell:+.4f}")
print(f"\nlog-energy distribution")
for nm, v in [("contributions", lc), ("cells", lk)]:
    print(f"  {nm:>14}: mean {v.mean():+.3f}  std {v.std():.3f}  "
          f"p10 {np.percentile(v,10):+.3f}  p50 {np.percentile(v,50):+.3f}  p90 {np.percentile(v,90):+.3f}")
json.dump(dict(frac_contrib=f_contrib, frac_cell=f_cell, n_contrib=len(contrib), n_cell=len(cell),
               inflation=len(contrib)/len(cell)), open(a.out, "w"), indent=2)
print(f"\nwrote {a.out}")
