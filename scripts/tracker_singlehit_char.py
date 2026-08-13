"""What IS the real single-hit r distribution, and which group does my placement recipe get wrong?
Real single-hit r is left-skewed (mean<median) -> a low-r tail placement misses. Split single-hit
by group (charged/fragment, photon, neutron), and for each compare REAL hit (r,z) vs the PLACEMENT
recipe (charged->production vertex; neutral->straight-line flight, L sampled from held-out shard).
Also characterize the low-r (<300mm) single-hit population directly. Pure data, no model."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

PF_LOGPT, PF_ETA, PF_PHI, PF_PDG, PF_CHARGE, PF_MASS = range(6)
AUX_PRIMARY, AUX_PARENT, AUX_VX, AUX_VY, AUX_VZ, AUX_ENERGY = range(6)
TH_LAYER, TH_R, TH_PHI, TH_Z, TH_TIME = range(5)
CLASS = {0: "other", 1: "cls1", 2: "photon", 9: "neutron", 10: "nbar"}


def mp(x):
    if len(x) == 0:
        return "empty"
    p = np.percentile(x, [10, 50, 90])
    return f"n={len(x):8d}  med={p[1]:6.0f} mean={x.mean():6.0f}  p10={p[0]:6.0f} p90={p[2]:6.0f}  frac<300={np.mean(x<300):.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--ref_shard", type=int, default=1)
    ap.add_argument("--preproc_dir", default="/scratch/gpfs/IOJALVO/lv7805/genpu_data/preprocessed")
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    # flight reference (held-out)
    dr = np.load(Path(args.preproc_dir) / f"shard_{args.ref_shard:04d}_stage2.npz")
    nhr = np.diff(dr["tracker_offsets"]); oner = np.where(nhr == 1)[0]; ar = dr["tracker_offsets"][oner]
    thr, auxr, pfr = dr["tracker_hits_flat"], dr["particle_aux"], dr["particle_features"]
    hxr = thr[ar, TH_R]*np.cos(thr[ar, TH_PHI]); hyr = thr[ar, TH_R]*np.sin(thr[ar, TH_PHI])
    dispr = np.sqrt((hxr-auxr[oner, AUX_VX])**2 + (hyr-auxr[oner, AUX_VY])**2 + (thr[ar, TH_Z]-auxr[oner, AUX_VZ])**2)
    clsr = pfr[oner, PF_PDG].astype(int)
    ref = {"photon": dispr[clsr == 2], "neutron": dispr[np.isin(clsr, [9, 10])]}

    d = np.load(Path(args.preproc_dir) / f"shard_{args.shard:04d}_stage2.npz")
    pf, aux, th, off = d["particle_features"], d["particle_aux"], d["tracker_hits_flat"], d["tracker_offsets"]
    nh = np.diff(off); one = np.where(nh == 1)[0]; a = off[one]
    cls = pf[one, PF_PDG].astype(int)
    vx, vy, vz = aux[one, AUX_VX], aux[one, AUX_VY], aux[one, AUX_VZ]
    vr = np.hypot(vx, vy); eta, ph = pf[one, PF_ETA], pf[one, PF_PHI]
    hr = th[a, TH_R]; hz = th[a, TH_Z]

    groups = {"charged/frag": ~np.isin(cls, [2, 9, 10]), "photon": cls == 2, "neutron": np.isin(cls, [9, 10])}

    print("=" * 96)
    print(f"REAL single-hit r by group (shard {args.shard}); overall real: {mp(hr)}")
    print("=" * 96)
    for g, m in groups.items():
        print(f"\n[{g}]  ({m.mean()*100:.1f}% of single-hit)")
        print(f"  REAL  hit_r:  {mp(hr[m])}")
        print(f"        vtx_r:  {mp(vr[m])}   |hit_r-vtx_r| med={np.median(np.abs(hr[m]-vr[m])):.1f}")
        # placement recipe for this group
        if g == "charged/frag":
            pr = vr[m]
        else:
            ce = np.cosh(eta[m]); nx = np.cos(ph[m])/ce; ny = np.sin(ph[m])/ce
            L = rng.choice(ref["photon" if g == "photon" else "neutron"], size=int(m.sum()))
            pr = np.hypot(vx[m] + L*nx, vy[m] + L*ny)
        print(f"        PLACED hit_r: {mp(pr)}   <- recipe vs REAL above")

    print("\n" + "=" * 96)
    print("LOW-r single hits (real hit_r < 300): what are they?")
    lo = hr < 300
    print(f"  n={lo.sum()} ({lo.mean()*100:.1f}% of single-hit)   primary frac={np.mean(aux[one[lo],AUX_PRIMARY]>0.5):.3f}")
    print(f"  vtx_r of these: {mp(vr[lo])}")
    print(f"  |hit-vtx|3D med={np.median(np.sqrt((th[a[lo],TH_R]*np.cos(th[a[lo],TH_PHI])-vx[lo])**2 + (th[a[lo],TH_R]*np.sin(th[a[lo],TH_PHI])-vy[lo])**2 + (hz[lo]-vz[lo])**2)):.1f}")
    lc = cls[lo]; u, c = np.unique(lc, return_counts=True)
    print("  species:", {CLASS.get(int(t), str(int(t))): round(float(c[i]/lo.sum()), 3) for i, t in enumerate(u) if c[i]/lo.sum() > 0.02})


if __name__ == "__main__":
    main()
