"""Material interaction map on CORRECT data (stage2, truth primary flag): where PRIMARIES stop
(last-hit radius) and where SECONDARIES are born (vertex radius), vs the barrel layers."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from genpu.detector_geometry import LAYER_MEANS, LAYER_IS_BARREL

PF_PDG = 3
AUX_PRIMARY, AUX_VX, AUX_VY, AUX_VZ = 0, 2, 3, 4
TH_R = 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--preproc_dir", default="/scratch/gpfs/IOJALVO/lv7805/genpu_data/preprocessed")
    ap.add_argument("--outdir", default="/home/lv7805/genpu/plots/tracker")
    args = ap.parse_args()
    d = np.load(Path(args.preproc_dir) / f"shard_{args.shard:04d}_stage2.npz")
    pf, aux, th, off = d["particle_features"], d["particle_aux"], d["tracker_hits_flat"], d["tracker_offsets"]
    nh = np.diff(off); keep = nh >= 1
    prim = aux[:, AUX_PRIMARY] > 0.5
    vr = np.hypot(aux[:, AUX_VX], aux[:, AUX_VY])
    last_r = th[off[1:] - 1, TH_R]
    layer_r = np.sort(np.unique(np.round(LAYER_MEANS[LAYER_IS_BARREL, 0]).astype(int)))

    P = keep & prim; S = keep & ~prim
    print("=" * 60); print(f"MATERIAL MAP (correct, stage2 shard {args.shard})"); print("=" * 60)
    print(f"barrel layer radii: {list(layer_r)}")
    print(f"PRIMARY stop last_r:   median={np.median(last_r[P]):.0f} pctiles {np.percentile(last_r[P],[25,50,75,90]).round(0)}")
    print(f"SECONDARY birth vr:    median={np.median(vr[S]):.0f} pctiles {np.percentile(vr[S],[25,50,75,90]).round(0)}")
    dmin = np.min(np.abs(last_r[P][:, None] - layer_r[None, :]), axis=1)
    print(f"frac PRIMARY stops within 20mm of a barrel layer: {np.mean(dmin<20):.2f} "
          f"(uniform ~{2*20*len(layer_r)/(layer_r.max()-layer_r.min()):.2f})")

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    bins = np.linspace(0, 1050, 106)
    ax[0].hist(last_r[P], bins=bins, histtype="step", lw=2, color="C0", label=f"PRIMARY stop (n={P.sum()})")
    ax[0].hist(vr[S], bins=bins, histtype="step", lw=2, color="C1", label=f"SECONDARY birth vr (n={S.sum()})")
    for lr in layer_r:
        ax[0].axvline(lr, color="gray", alpha=0.35, lw=0.8)
    ax[0].set_xlabel("radius (mm)"); ax[0].set_title("where interactions happen (grey=barrel layers) [CORRECT]"); ax[0].legend()
    ax[1].hist(nh[P], bins=np.arange(0, 25), histtype="step", lw=2, color="C0", density=True, label="PRIMARY")
    ax[1].hist(nh[S], bins=np.arange(0, 25), histtype="step", lw=2, color="C1", density=True, label="SECONDARY")
    ax[1].set_xlabel("n_hits"); ax[1].set_title("n_hits: primary vs secondary [CORRECT]"); ax[1].legend()
    plt.tight_layout(); fig.savefig(Path(args.outdir) / "material.png", dpi=100)
    print(f"wrote {Path(args.outdir)/'material.png'}")


if __name__ == "__main__":
    main()
