"""Why do tracks stop when they do? Decompose n_hits against kinematics and geometry:
  - n_hits distribution (how skewed / how many 2-hit tracks)
  - n_hits vs p_T and |eta|  -> is length kinematically DETERMINED (count head will work) or noisy?
  - WHERE the last hit sits: near the detector outer edge (track EXITED) vs interior (stopped /
    ranged out / interacted). Split short vs long tracks.
Runs on the v2 slice (numpy only)."""
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
    cont, hits, off = d["cont"], d["hits"], d["offsets"]
    n_hits = np.diff(off).astype(int)
    pt = np.exp(cont[:, 0]); aeta = np.abs(cont[:, 1])
    lc = hits[:, 0].astype(int).clip(0, N_LAYERS - 1)
    r_phys = hits[:, 1] * LAYER_STDS[lc, 0] + LAYER_MEANS[lc, 0]
    z_phys = hits[:, 3] * LAYER_STDS[lc, 2] + LAYER_MEANS[lc, 2]
    last_r = r_phys[off[1:] - 1]; first_r = r_phys[off[:-1]]     # hits sorted inner->outer
    last_z = z_phys[off[1:] - 1]
    R_MAX = np.percentile(r_phys, 99.5); Z_MAX = np.percentile(np.abs(z_phys), 99.5)

    print("=" * 64); print(f"TRACK LENGTH DIAGNOSTIC ({len(n_hits)} pion tracks)"); print("=" * 64)
    print(f"n_hits: mean={n_hits.mean():.2f} median={np.median(n_hits):.0f} "
          f"frac[==1]={np.mean(n_hits==1):.2f} [==2]={np.mean(n_hits==2):.2f} "
          f"[<=2]={np.mean(n_hits<=2):.2f} [>=10]={np.mean(n_hits>=10):.2f} max={n_hits.max()}")
    print(f"p_T: median={np.median(pt):.2f} GeV  p5={np.percentile(pt,5):.2f} p95={np.percentile(pt,95):.2f}")
    print(f"detector reach: r99.5={R_MAX:.0f}  |z|99.5={Z_MAX:.0f}")

    print("\n-- n_hits vs kinematics (mean n_hits in bins) --")
    for name, v in [("p_T", pt), ("|eta|", aeta)]:
        q = np.quantile(v, [0, .25, .5, .75, 1.0])
        row = []
        for i in range(4):
            m = (v >= q[i]) & (v <= q[i + 1] if i == 3 else v < q[i + 1])
            row.append(f"[{q[i]:.2f}-{q[i+1]:.2f}]:{n_hits[m].mean():.1f}")
        print(f"  {name:6s} quartiles -> " + "  ".join(row))
    # correlation
    print(f"  corr(n_hits, log p_T)={np.corrcoef(n_hits, np.log(pt))[0,1]:+.3f}   "
          f"corr(n_hits, |eta|)={np.corrcoef(n_hits, aeta)[0,1]:+.3f}")

    print("\n-- WHERE tracks stop (last-hit radius) --")
    def stopinfo(mask, tag):
        lr = last_r[mask]; lz = np.abs(last_z[mask])
        exited = np.mean((lr > 0.9 * R_MAX) | (lz > 0.9 * Z_MAX))     # near barrel OR endcap edge
        print(f"  {tag:16s} n={mask.sum():7d}  last_r mean={lr.mean():6.0f} med={np.median(lr):6.0f}  "
              f"frac exited(edge)={exited:.2f}")
    stopinfo(n_hits == 2, "2-hit tracks")
    stopinfo((n_hits >= 3) & (n_hits <= 5), "3-5 hit")
    stopinfo(n_hits >= 10, "10+ hit")
    print(f"\n  first-hit r: median={np.median(first_r):.0f}  "
          f"(are 2-hit tracks starting deep? -> secondaries)")
    for tag, m in [("2-hit", n_hits == 2), ("10+", n_hits >= 10)]:
        print(f"    {tag}: first_r med={np.median(first_r[m]):.0f}  pt med={np.median(pt[m]):.2f}  |eta| med={np.median(aeta[m]):.2f}")


if __name__ == "__main__":
    main()
