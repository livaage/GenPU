"""Does a single-hit particle's ONE hit sit at its production vertex?
If yes (for secondaries), we can place single-hit debris directly at the birth vertex instead of
running the AR (which mis-seeds them to the beampipe layer 7). Split by primary/secondary, because
a single-hit PRIMARY is born at the center and its hit is NOT at the vertex (born r~0, hit further out),
whereas a single-hit SECONDARY is born in material and should leave its hit right there.
Pure data check on stage2 (event-joined)."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from genpu.detector_geometry import LAYER_MEANS, N_LAYERS, LAYER_IS_BARREL

PF_LOGPT, PF_ETA, PF_PHI, PF_PDG, PF_CHARGE, PF_MASS = range(6)
AUX_PRIMARY, AUX_PARENT, AUX_VX, AUX_VY, AUX_VZ, AUX_ENERGY = range(6)
TH_LAYER, TH_R, TH_PHI, TH_Z, TH_TIME = range(5)
CLASS = {0: "other", 1: "?1", 2: "photon", 3: "pi+", 4: "pi-", 5: "K+", 6: "K-", 7: "p",
         8: "pbar", 9: "n", 10: "nbar", 11: "e-", 12: "e+", 13: "mu-", 14: "mu+"}
# barrel layer nominal radii (for the "snap hit to nearest layer at vertex radius" recipe)
BARREL_R = np.sort(LAYER_MEANS[LAYER_IS_BARREL, 0])


def pct(x):
    p = np.percentile(x, [50, 75, 90, 99])
    return f"med {p[0]:7.1f}  p75 {p[1]:7.1f}  p90 {p[2]:7.1f}  p99 {p[3]:7.1f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--preproc_dir", default="/scratch/gpfs/IOJALVO/lv7805/genpu_data/preprocessed")
    args = ap.parse_args()
    d = np.load(Path(args.preproc_dir) / f"shard_{args.shard:04d}_stage2.npz")
    pf, aux, th, off = d["particle_features"], d["particle_aux"], d["tracker_hits_flat"], d["tracker_offsets"]
    nh = np.diff(off)
    one = np.where(nh == 1)[0]                     # single-hit particles
    a = off[one]                                   # their single hit index
    prim = aux[one, AUX_PRIMARY] > 0.5
    cls = pf[one, PF_PDG].astype(int)

    vx, vy, vz = aux[one, AUX_VX], aux[one, AUX_VY], aux[one, AUX_VZ]
    vr = np.hypot(vx, vy)
    hr, hphi, hz = th[a, TH_R], th[a, TH_PHI], th[a, TH_Z]
    hx, hy = hr * np.cos(hphi), hr * np.sin(hphi)
    dxy = np.hypot(hx - vx, hy - vy)               # transverse hit->vertex distance
    dz = hz - vz
    d3d = np.sqrt(dxy ** 2 + dz ** 2)
    dr = hr - vr                                   # radial: hit r vs birth r
    # "snap to nearest barrel layer at the vertex radius" residual (the actual placement recipe)
    snap_r = BARREL_R[np.abs(BARREL_R[None, :] - vr[:, None]).argmin(1)]
    d_snap = np.abs(hr - snap_r)

    tot = len(nh)
    print("=" * 82)
    print(f"SINGLE-HIT particles (stage2 shard {args.shard}): n={len(one)} = {len(one)/tot*100:.1f}% of all {tot}")
    print(f"  of single-hit: primary {prim.mean()*100:.1f}%   secondary {(~prim).mean()*100:.1f}%")
    print("=" * 82)
    for lab, m in [("SECONDARY single-hit", ~prim), ("PRIMARY single-hit", prim)]:
        if m.sum() < 100:
            continue
        print(f"\n{lab}  (n={m.sum()}, {m.sum()/len(one)*100:.1f}% of single-hit)")
        print(f"  |hit - vertex| 3D:   {pct(d3d[m])}")
        print(f"  |hit_r - vertex_r|:  {pct(np.abs(dr[m]))}   (signed median {np.median(dr[m]):+.1f})")
        print(f"  |hit_z - vertex_z|:  {pct(np.abs(dz[m]))}")
        print(f"  vertex_r itself:     {pct(vr[m])}")
        print(f"  snap-to-layer resid |hit_r - nearest_barrel_layer(vertex_r)|: {pct(d_snap[m])}")
        # what the AR does instead: default seed ~layer 7 (r=32.5); error vs truth hit
        print(f"  [AR default would place at layer7 r=32.5 -> error vs truth hit_r: med {np.median(np.abs(hr[m]-32.5)):.1f}]")
    print("\nspecies of SECONDARY single-hit:")
    sc = cls[~prim]
    for c in sorted(set(sc)):
        f = (sc == c).mean()
        if f > 0.01:
            print(f"  {CLASS.get(c,str(c)):8s} {f*100:5.1f}%   |hit-vtx|3D med={np.median(d3d[~prim][sc==c]):.1f}")


if __name__ == "__main__":
    main()
