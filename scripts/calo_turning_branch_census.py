"""Who is actually IN the turning-point anchor branch?

STATUS describes it as "soft secondaries (median pT 0.27 GeV, born at vr ~ 420 mm) that curl back
before the calo face". But J3a (job 12874732) found 91% of depositors BORN INSIDE the calorimeter
fall to that branch, because a helix cannot be extrapolated forward to a face the particle is
already behind. Those are shower fragments, not curlers.

This conditions the other way -- given the turning branch, what is the population? -- and reports
the vr/pT distributions per sub-population, so the two readings can be told apart.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from genpu.calo_geom import core_anchor, load_front_face  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--graph_dir", default="/scratch/gpfs/IOJALVO/lv7805/genpu_data/cascade_graph")
ap.add_argument("--out", default="/home/lv7805/genpu/plots/calo/metrics/turning_branch_census.json")
a = ap.parse_args()

R, Z = load_front_face(None)
g = np.load(f"{a.graph_dir}/shard_{a.shard:04d}_graph.npz")
pdg, ncal, ntrk = g["pdg_id"], g["n_calo_hits"], g["n_tracker_hits"]
vx, vy, vz, px, py, pz, chg = (g[k] for k in ["vx", "vy", "vz", "px", "py", "pz", "charge"])
vr = np.hypot(vx, vy); pt = np.hypot(px, py)
phi0 = np.arctan2(py, px); eta = np.arcsinh(np.clip(pz/np.clip(pt, 1e-9, None), -30, 30))
inside = (vr >= R) | (np.abs(vz) >= Z)

out = {}
for code, nm in [(211, "pi+"), (-211, "pi-"), (11, "e-"), (-11, "e+")]:
    sel = np.where((ncal >= 1) & (pdg == code))[0]
    if len(sel) < 5000: continue
    if len(sel) > 500000:
        sel = np.random.default_rng(0).choice(sel, 500000, replace=False)
    _, _, mode = core_anchor(pt[sel], phi0[sel], eta[sel], chg[sel], vx[sel], vy[sel], vz[sel], R, Z)
    turn = sel[mode == 2]
    ins = inside[turn]
    print(f"\n=== {nm} ===  depositors {len(sel):,}  turning branch {len(turn):,} "
          f"({len(turn)/len(sel):.3f} of depositors)")
    print(f"  OF THE TURNING BRANCH: born INSIDE calo {ins.mean():.3f}   born outside {1-ins.mean():.3f}")
    rows = {}
    for lab, m in [("born INSIDE (shower fragment)", ins), ("born outside (true curler?)", ~ins)]:
        t = turn[m]
        if not len(t): continue
        rows[lab] = dict(n=int(len(t)), vr_median=float(np.median(vr[t])),
                         pt_median=float(np.median(pt[t])),
                         zero_tracker_hits=float(np.mean(ntrk[t] == 0)))
        print(f"  {lab:32s} n={len(t):>8,}  vr med {np.median(vr[t]):>7.1f} mm  "
              f"pT med {np.median(pt[t]):.3f} GeV  P(0 tracker hits) {np.mean(ntrk[t]==0):.3f}")
    out[nm] = dict(turning_frac=float(len(turn)/len(sel)), inside_frac=float(ins.mean()), pops=rows)
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
json.dump(out, open(a.out, "w"), indent=2)
print(f"\nwrote {a.out}")
