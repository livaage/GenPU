"""Does the impact parameter d0 predict track length? d0 is THE variable ATLAS uses to
separate primaries (|d0|<1.5mm) from secondaries (|d0|>=5mm removes >99% of primaries).
We never used it. Test corr(n_hits, |d0|) and n_hits split by the standard d0 cuts."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from genpu.data import load_shard, explode_list_columns
from genpu.preprocessing import pdg_to_class


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--max_particles", type=int, default=400000)
    args = ap.parse_args()
    d0l, z0l, nhl, clsl, ptl, vrl = [], [], [], [], [], []
    for sh in args.shards:
        P = load_shard("particles", sh); T = load_shard("tracker_hits", sh)
        cols = set(P.schema.names)
        assert "perigee_d0" in cols, f"no perigee_d0; have {sorted(cols)}"
        for e in range(min(P.num_rows, T.num_rows)):
            pe = explode_list_columns(P, e); te = explode_list_columns(T, e)
            tpid = te["particle_id"].astype(np.int64)
            cls = pdg_to_class(pe["pdg_id"].astype(np.int64))
            pt = np.hypot(pe["px"], pe["py"]); vr = np.hypot(pe["vx"], pe["vy"])
            for i in range(len(pe["particle_id"])):
                nh = int((tpid == pe["particle_id"][i]).sum())
                if nh == 0:
                    continue
                d0l.append(pe["perigee_d0"][i]); z0l.append(pe["perigee_z0"][i])
                nhl.append(nh); clsl.append(int(cls[i])); ptl.append(pt[i]); vrl.append(vr[i])
            if len(nhl) >= args.max_particles:
                break
        if len(nhl) >= args.max_particles:
            break
    d0 = np.abs(np.array(d0l)); z0 = np.abs(np.array(z0l)); nh = np.array(nhl, float)
    pt = np.array(ptl); vr = np.array(vrl)
    print("=" * 62); print(f"d0 TEST ({len(nh)} particles)"); print("=" * 62)
    print(f"|d0| dist: median={np.median(d0):.2f}  p90={np.percentile(d0,90):.1f}  p99={np.percentile(d0,99):.1f} mm")
    print(f"corr(n_hits, |d0|)={np.corrcoef(nh, d0)[0,1]:+.3f}  "
          f"corr(n_hits, log(|d0|+.1))={np.corrcoef(nh, np.log(d0+0.1))[0,1]:+.3f}  "
          f"corr(n_hits, |z0|)={np.corrcoef(nh, z0)[0,1]:+.3f}  corr(n_hits, vr)={np.corrcoef(nh, vr)[0,1]:+.3f}")
    print("\nn_hits vs |d0| bins:")
    edges = [0, 0.5, 1.5, 5, 20, 100, 1e9]
    for i in range(len(edges) - 1):
        m = (d0 >= edges[i]) & (d0 < edges[i + 1])
        if m.sum() < 50:
            continue
        print(f"  |d0| [{edges[i]:.1f}-{edges[i+1]:.0f}) mm: n={m.sum():7d} ({m.mean()*100:4.1f}%)  "
              f"n_hits mean={nh[m].mean():5.2f} median={np.median(nh[m]):.0f}  frac[==1]={np.mean(nh[m]==1):.2f} "
              f"frac[>=7]={np.mean(nh[m]>=7):.2f}")
    prim = d0 < 1.5; sec = d0 >= 5
    print(f"\nATLAS-style split:")
    print(f"  PRIMARY-like |d0|<1.5mm:  {prim.mean()*100:.1f}%  n_hits mean={nh[prim].mean():.2f} median={np.median(nh[prim]):.0f} frac[>=7]={np.mean(nh[prim]>=7):.2f}")
    print(f"  SECONDARY   |d0|>=5mm:    {sec.mean()*100:.1f}%  n_hits mean={nh[sec].mean():.2f} median={np.median(nh[sec]):.0f} frac[>=7]={np.mean(nh[sec]>=7):.2f}")
    print(f"  of the SINGLE-hit particles, frac with |d0|>=5mm: {np.mean(d0[nh==1]>=5):.2f}  (are single hits mostly secondaries?)")


if __name__ == "__main__":
    main()
