"""J4a — can ONE recursive model serve every cascade level, and does the material map need phi?

The primaries-in design generates the cascade in ~3-4 batched levels (depth 0/1/2/3 carry
40.5/35.7/13.7/6.6% of tracker hits). Two design questions have to be answered from data BEFORE any
model is built, because they decide the architecture:

  Q1 SELF-SIMILARITY — is the depth-2 parent->daughter relation the SAME function as depth-1?
     If yes, one model applied recursively covers all levels. If depth-2 has different multiplicity,
     species mix or radius behaviour at matched parent kinematics, each level needs its own model
     (or an explicit depth input), which is a much bigger build.
     Both July spikes (conversion, nuclear) were trained on DIRECT daughters only = depth 1.

  Q2 PHI SYMMETRY — build_conversion_slice.py conditions on [log_E, eta, vr, vz] with NO phi, i.e.
     it learns a phi-AVERAGED material map. ODD is idealised so this may be near-exact, but it has
     never been checked. If the interaction radius or rate modulates with phi, the conditioning is
     missing a variable and will not extrapolate.

Measurement only, on the complete raw graph. No training, no model.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--graph_dir", default="/scratch/gpfs/IOJALVO/lv7805/genpu_data/cascade_graph")
ap.add_argument("--out", default="/home/lv7805/genpu/plots/calo/metrics/cascade_depth_char.json")
a = ap.parse_args()

g = np.load(f"{a.graph_dir}/shard_{a.shard:04d}_graph.npz")
ev, pid, par, depth = g["event_id"], g["particle_id"], g["parent_id"], g["depth"]
pdg, E = g["pdg_id"], g["energy"]
vx, vy, vz = g["vx"], g["vy"], g["vz"]
vr = np.hypot(vx, vy); vphi = np.arctan2(vy, vx)
n = len(pid)
print(f"nodes {n:,}   depth hist {np.bincount(np.clip(depth,0,6))}")

# parent row for every node, via (event, id) key
key = (ev.astype(np.int64) << 32) | pid.astype(np.int64)
o = np.argsort(key); ks = key[o]
pk = (ev.astype(np.int64) << 32) | par.astype(np.int64)
ip = np.clip(np.searchsorted(ks, pk), 0, len(ks) - 1)
prow = np.where(ks[ip] == pk, o[ip], -1)
has_p = prow >= 0
print(f"nodes with a resolvable parent: {has_p.mean():.4f}")

# ---- Q1: daughter multiplicity / displacement, per PARENT DEPTH, at matched parent kinematics ----
ndau = np.bincount(prow[has_p], minlength=n)
disp = np.full(n, np.nan)
disp[has_p] = np.hypot(vr[has_p] - vr[prow[has_p]], vz[has_p] - vz[prow[has_p]])

out = {}
print("\nQ1 SELF-SIMILARITY — parents that produced >=1 daughter, binned by parent log-E")
for code, nm in [(22, "gamma"), (11, "e-"), (-11, "e+"), (211, "pi+"), (-211, "pi-")]:
    sel = np.where((pdg == code) & (ndau >= 1))[0]
    if len(sel) < 5000:
        continue
    lo = np.log(np.clip(E[sel], 1e-6, None))
    edges = np.quantile(lo, [0, .25, .5, .75, 1.0])
    print(f"\n  {nm}: parents with daughters {len(sel):,}")
    print(f"    {'parent depth':>13} {'n':>9} {'mean n_dau':>11} {'med disp[mm]':>13} "
          f"{'med logE bin0':>14} {'mean n_dau b0':>14} {'mean n_dau b3':>14}")
    rows = {}
    for d in range(4):
        m = sel[depth[sel] == d]
        if len(m) < 500:
            continue
        b = np.clip(np.digitize(np.log(np.clip(E[m], 1e-6, None)), edges[1:-1]), 0, 3)
        nb = [float(ndau[m[b == k]].mean()) if (b == k).sum() > 50 else float("nan") for k in range(4)]
        rows[d] = dict(n=int(len(m)), mean_ndau=float(ndau[m].mean()),
                       med_disp=float(np.nanmedian(disp[m])) if np.isfinite(disp[m]).any() else None,
                       ndau_by_Ebin=nb)
        print(f"    {d:>13} {len(m):>9,} {ndau[m].mean():>11.3f} "
              f"{np.nanmedian(np.hypot(vr[m]-vr[prow[m]], vz[m]-vz[prow[m]])) if depth[m].min()>0 else float('nan'):>13.1f} "
              f"{'':>14} {nb[0]:>14.3f} {nb[3]:>14.3f}")
    out[f"selfsim_{nm}"] = rows

# ---- Q2: does the material map depend on PHI? ----
print("\nQ2 PHI SYMMETRY — interaction rate and radius vs production phi")
NPHI = 12
pe = np.linspace(-np.pi, np.pi, NPHI + 1)
for code, nm in [(22, "gamma"), (211, "pi+"), (-211, "pi-")]:
    sel = np.where((pdg == code) & (vr < 1100))[0]          # produced inside the tracker volume
    if len(sel) < 20000:
        continue
    b = np.clip(np.digitize(vphi[sel], pe[1:-1]), 0, NPHI - 1)
    inter = (ndau[sel] >= 1).astype(np.float64)
    rate = np.array([inter[b == k].mean() for k in range(NPHI)])
    dsel = sel[ndau[sel] >= 1]
    bd = np.clip(np.digitize(vphi[dsel], pe[1:-1]), 0, NPHI - 1)
    rad = np.array([np.median(vr[dsel[bd == k]]) if (bd == k).sum() > 100 else np.nan
                    for k in range(NPHI)])
    spread = (rate.max() - rate.min()) / max(rate.mean(), 1e-9)
    rspread = (np.nanmax(rad) - np.nanmin(rad)) / max(np.nanmean(rad), 1e-9)
    print(f"  {nm:>6}: n={len(sel):,}  interact rate mean {rate.mean():.4f} "
          f"min {rate.min():.4f} max {rate.max():.4f}  -> RELATIVE SPREAD {spread:.4f}")
    print(f"          production r median {np.nanmean(rad):7.1f} mm  spread {rspread:.4f}")
    out[f"phi_{nm}"] = dict(rate_rel_spread=float(spread), radius_rel_spread=float(rspread),
                            rate_by_phi=rate.tolist())
print("\n  (relative spread << 0.05 => phi-averaged conditioning is safe; >~0.1 => phi is a "
      "missing variable in build_conversion_slice.py)")
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
json.dump(out, open(a.out, "w"), indent=2, default=float)
print(f"\nwrote {a.out}")
