"""Understand the tracking, ON CORRECT DATA (preprocessed stage2 = event_id-joined).
Redoes on valid data everything the buggy source join got wrong: per-species survival,
primary vs secondary (TRUTH flag), and length predictability incl. d0."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np

PF_LOGPT, PF_ETA, PF_PHI, PF_PDG, PF_CHARGE, PF_MASS = range(6)
AUX_PRIMARY, AUX_PARENT, AUX_VX, AUX_VY, AUX_VZ, AUX_ENERGY = range(6)
TH_LAYER, TH_R = 0, 1
CLASS = {2: "photon", 3: "pi+", 4: "pi-", 5: "K+", 6: "K-", 7: "p", 8: "pbar",
         9: "n", 10: "nbar", 11: "e-", 12: "e+", 13: "mu-", 14: "mu+"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--preproc_dir", default="/scratch/gpfs/IOJALVO/lv7805/genpu_data/preprocessed")
    args = ap.parse_args()
    d = np.load(Path(args.preproc_dir) / f"shard_{args.shard:04d}_stage2.npz")
    pf, aux, th, off = d["particle_features"], d["particle_aux"], d["tracker_hits_flat"], d["tracker_offsets"]
    nh = np.diff(off)
    keep = nh >= 1                                        # particles with >=1 tracker hit
    cls = pf[:, PF_PDG].astype(int); pt = np.exp(pf[:, PF_LOGPT]); aeta = np.abs(pf[:, PF_ETA])
    phi = pf[:, PF_PHI]; prim = aux[:, AUX_PRIMARY] > 0.5
    d0 = np.abs(aux[:, AUX_VX] * np.sin(phi) - aux[:, AUX_VY] * np.cos(phi))
    avz = np.abs(aux[:, AUX_VZ]); vr = np.hypot(aux[:, AUX_VX], aux[:, AUX_VY])
    last_r = th[off[1:] - 1, TH_R]                        # outermost hit radius (hits sorted by r)

    print("=" * 66); print(f"CORRECT-DATA TRACKING ANALYSIS (stage2 shard {args.shard}, {keep.sum()} particles w/ hits)"); print("=" * 66)
    print("1. PARTICLE TYPE (survival) — do muons penetrate, hadrons interact?")
    print(f"  {'species':8s} {'n':>8s} {'nh mean':>8s} {'median':>7s} {'>=7':>6s} {'>=15':>6s}")
    for c in sorted(set(cls[keep])):
        s = keep & (cls == c)
        if s.sum() < 300:
            continue
        print(f"  {CLASS.get(c,str(c)):8s} {s.sum():8d} {nh[s].mean():8.2f} {np.median(nh[s]):7.0f} "
              f"{np.mean(nh[s]>=7):6.2f} {np.mean(nh[s]>=15):6.2f}")

    print("\n2. PRIMARY vs SECONDARY (truth flag)")
    for lab, m in [("PRIMARY", keep & prim), ("SECONDARY", keep & ~prim)]:
        print(f"  {lab:10s} {m.sum():8d} ({m.sum()/keep.sum()*100:4.1f}%)  nh mean={nh[m].mean():.2f} "
              f"median={np.median(nh[m]):.0f}  >=7={np.mean(nh[m]>=7):.2f}  vr med={np.median(vr[m]):.0f}  last_r med={np.median(last_r[m]):.0f}")

    print("\n3. LENGTH PREDICTABILITY  corr(n_hits, .)")
    for lab, m in [("all w/hit", keep), ("pions", keep & np.isin(cls, [3, 4])), ("primaries", keep & prim)]:
        if m.sum() < 500:
            continue
        cc = lambda v: np.corrcoef(nh[m], v[m])[0, 1]
        print(f"  {lab:10s}: logpt={cc(np.log(pt)):+.3f}  |eta|={cc(aeta):+.3f}  |d0|={cc(d0):+.3f}  "
              f"log|d0|={np.corrcoef(nh[m], np.log(d0[m]+.1))[0,1]:+.3f}  |vz|={cc(avz):+.3f}  primary={np.corrcoef(nh[m], prim[m].astype(float))[0,1]:+.3f}")

    print("\n4. d0 detail (pions): n_hits vs |d0|")
    s = keep & np.isin(cls, [3, 4])
    for lo, hi in [(0, .5), (.5, 1.5), (1.5, 5), (5, 20), (20, 1e9)]:
        m = s & (d0 >= lo) & (d0 < hi)
        if m.sum() < 30:
            continue
        print(f"  |d0| [{lo:.1f}-{hi:.0f}): n={m.sum():7d} ({m.sum()/s.sum()*100:4.1f}%)  nh mean={nh[m].mean():5.2f} med={np.median(nh[m]):.0f} >=7={np.mean(nh[m]>=7):.2f}")


if __name__ == "__main__":
    main()
