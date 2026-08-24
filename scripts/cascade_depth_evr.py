"""Does DEPTH still predict daughter multiplicity once production POSITION is controlled?

J4a (job 12879196) found mean daughters per parent falls sharply with depth (pions 4.05 -> 2.14 ->
1.69 -> 1.55) and that it survives matching on parent ENERGY. But depth correlates with where the
particle was born, and `vr`/`vz` are ALREADY conditioning variables in build_conversion_slice.py.
So the J4a test could not distinguish "depth matters" from "position matters, and depth is a proxy".

This bins jointly on (log E x vr) and asks whether depth still moves multiplicity INSIDE a cell.

  depth spread ~ 1.0 inside (E, vr) cells  => depth is REDUNDANT. One recursive model conditioned on
                                              [log_E, eta, vr, vz, pdg] serves every level.
  depth spread still large                 => depth carries information position does not; add an
                                              explicit depth input (still far cheaper than
                                              per-level models).

The marginal (E-only) spread is printed alongside as the reference J4a effectively measured.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--graph_dir", default="/scratch/gpfs/IOJALVO/lv7805/genpu_data/cascade_graph")
ap.add_argument("--nbin", type=int, default=4)
ap.add_argument("--min_cell", type=int, default=300)
ap.add_argument("--out", default="/home/lv7805/genpu/plots/calo/metrics/cascade_depth_evr.json")
a = ap.parse_args()

g = np.load(f"{a.graph_dir}/shard_{a.shard:04d}_graph.npz")
ev, pid, par, depth, pdg, E = (g["event_id"], g["particle_id"], g["parent_id"], g["depth"],
                               g["pdg_id"], g["energy"])
vr = np.hypot(g["vx"], g["vy"])
n = len(pid)
key = (ev.astype(np.int64) << 32) | pid.astype(np.int64)
o = np.argsort(key); ks = key[o]
pk = (ev.astype(np.int64) << 32) | par.astype(np.int64)
ip = np.clip(np.searchsorted(ks, pk), 0, len(ks) - 1)
prow = np.where(ks[ip] == pk, o[ip], -1)
ndau = np.bincount(prow[prow >= 0], minlength=n)

out = {}
for code, nm in [(22, "gamma"), (11, "e-"), (-11, "e+"), (211, "pi+"), (-211, "pi-")]:
    sel = np.where((pdg == code) & (ndau >= 1))[0]
    if len(sel) < 20000:
        continue
    lo = np.log(np.clip(E[sel], 1e-9, None)); rr = vr[sel]
    eb = np.clip(np.digitize(lo, np.quantile(lo, np.linspace(0, 1, a.nbin + 1))[1:-1]), 0, a.nbin - 1)
    rb = np.clip(np.digitize(rr, np.quantile(rr, np.linspace(0, 1, a.nbin + 1))[1:-1]), 0, a.nbin - 1)
    d = np.clip(depth[sel], 0, 3)

    # marginal: E only (what J4a effectively did)
    marg = []
    for e_ in range(a.nbin):
        v = [ndau[sel[(eb == e_) & (d == k)]].mean() for k in range(1, 4)
             if ((eb == e_) & (d == k)).sum() >= a.min_cell]
        if len(v) >= 2: marg.append(max(v) / max(min(v), 1e-9))
    # joint: E x vr
    joint = []; cells = 0
    for e_ in range(a.nbin):
        for r_ in range(a.nbin):
            m = (eb == e_) & (rb == r_)
            v = [ndau[sel[m & (d == k)]].mean() for k in range(1, 4)
                 if (m & (d == k)).sum() >= a.min_cell]
            if len(v) >= 2:
                joint.append(max(v) / max(min(v), 1e-9)); cells += 1
    if not joint:
        continue
    mm, jj = float(np.median(marg)) if marg else float("nan"), float(np.median(joint))
    out[nm] = dict(marginal_spread=mm, joint_spread=jj, cells=cells,
                   joint_p90=float(np.percentile(joint, 90)))
    print(f"{nm:>6}: parents {len(sel):>9,}   depth spread (max/min over depths 1-3)")
    print(f"         E only      median {mm:.3f}")
    print(f"         E x vr      median {jj:.3f}   p90 {np.percentile(joint,90):.3f}   "
          f"({cells} cells)   -> {'REDUNDANT' if jj < 1.25 else 'depth still matters'}")

Path(a.out).parent.mkdir(parents=True, exist_ok=True)
json.dump(out, open(a.out, "w"), indent=2)
print(f"\nrule of thumb: joint spread < ~1.25 => one recursive model on [log_E, eta, vr, vz, pdg]")
print(f"wrote {a.out}")
