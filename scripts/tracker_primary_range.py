"""For PRIMARY tracks (vr<5, a beamline cut), is the stopping point explained by p_T via
curvature? A soft track curls with radius R = pt/(0.3 B); it cannot exceed transverse radius
2R before curling back, so last_r should be ~min(2R, detector edge). If so, primary length IS
predictable from energy/p_T (unlike the mixed sample, where secondaries wash it out).
Tests: corr(n_hits, log pT) for primaries; last_r vs pT for CENTRAL primaries; the implied 2R slope."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from genpu.detector_geometry import LAYER_MEANS, LAYER_STDS, N_LAYERS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", default="/scratch/gpfs/IOJALVO/lv7805/genpu_data/tracker_slice/v2_pion.npz")
    args = ap.parse_args()
    d = np.load(args.slice)
    cont, hits, off = d["cont"], d["pdg"] if False else d["hits"], d["offsets"]
    n_hits = np.diff(off).astype(int)
    pt = np.exp(cont[:, 0]); aeta = np.abs(cont[:, 1]); vr = cont[:, 5]
    lc = hits[:, 0].astype(int).clip(0, N_LAYERS - 1)
    r_phys = hits[:, 1] * LAYER_STDS[lc, 0] + LAYER_MEANS[lc, 0]
    last_r = r_phys[off[1:] - 1]
    R_EDGE = np.percentile(r_phys, 99.5)

    prim = vr < 5
    cen = prim & (aeta < 0.9)                      # central primaries: curling is transverse
    print("=" * 62); print(f"PRIMARY RANGE ({prim.sum()} primaries, {cen.sum()} central)"); print("=" * 62)
    print(f"detector edge r99.5 = {R_EDGE:.0f} mm")
    print(f"corr(n_hits, log pT):  all={np.corrcoef(n_hits, np.log(pt))[0,1]:+.3f}  "
          f"primaries={np.corrcoef(n_hits[prim], np.log(pt[prim]))[0,1]:+.3f}  "
          f"central-prim={np.corrcoef(n_hits[cen], np.log(pt[cen]))[0,1]:+.3f}")
    print(f"corr(last_r, pT):      primaries={np.corrcoef(last_r[prim], pt[prim])[0,1]:+.3f}  "
          f"central-prim={np.corrcoef(last_r[cen], pt[cen])[0,1]:+.3f}")

    print("\n-- central primaries: last_r and n_hits vs pT --")
    print(f"{'pT bin':>14s} {'n':>7s} {'last_r med':>11s} {'2R pred(2564*pT)':>17s} {'n_hits med':>11s}")
    edges = [0.05, 0.15, 0.25, 0.35, 0.45, 0.6, 1.0, 5.0]
    for i in range(len(edges) - 1):
        m = cen & (pt >= edges[i]) & (pt < edges[i + 1])
        if m.sum() < 20:
            continue
        pmid = np.median(pt[m])
        print(f"  [{edges[i]:.2f}-{edges[i+1]:.2f}] {m.sum():7d} {np.median(last_r[m]):11.0f} "
              f"{min(2564*pmid, R_EDGE):17.0f} {np.median(n_hits[m]):11.0f}")
    # fit last_r ~ slope*pT in the curling regime (pT<0.35, below the edge cap)
    curl = cen & (pt < 0.35)
    slope = np.sum(last_r[curl] * pt[curl]) / np.sum(pt[curl] ** 2)
    print(f"\n  curling regime (central, pT<0.35): last_r ~= {slope:.0f} * pT  "
          f"(2R theory=2564/T; implied B ~ {2564/slope*2.6:.1f} T if slope scales)")
    print(f"  frac central primaries reaching the edge (exited): {np.mean(last_r[cen] > 0.9*R_EDGE):.2f}")


if __name__ == "__main__":
    main()
