"""Map the material interactions in the tracker. Two radial distributions, against the
48 detector layers:
  - where PRIMARIES stop (last-hit radius)  -> the interaction point of the parent
  - where SECONDARIES are born (vertex radius vr) -> the daughters of those interactions
If both cluster at the SAME radii and those radii coincide with detector layers/supports,
interactions are localized at material; if smooth/exponential, it's bulk interaction length.
Writes plots/tracker/material.png + a text summary of the layer structure near the peaks."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from genpu.detector_geometry import LAYER_MEANS, LAYER_STDS, N_LAYERS, LAYER_IS_BARREL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", default="/scratch/gpfs/IOJALVO/lv7805/genpu_data/tracker_slice/v2_pion.npz")
    ap.add_argument("--outdir", default="/home/lv7805/genpu/plots/tracker")
    args = ap.parse_args()
    d = np.load(args.slice)
    cont, hits, off = d["cont"], d["hits"], d["offsets"]
    vr = cont[:, 5]
    lc = hits[:, 0].astype(int).clip(0, N_LAYERS - 1)
    r_phys = hits[:, 1] * LAYER_STDS[lc, 0] + LAYER_MEANS[lc, 0]
    last_r = r_phys[off[1:] - 1]; first_r = r_phys[off[:-1]]
    prim = vr < 5; sec = vr >= 5

    layer_r = np.sort(np.unique(np.round(LAYER_MEANS[LAYER_IS_BARREL, 0]).astype(int)))
    print("=" * 62); print("MATERIAL INTERACTION MAP"); print("=" * 62)
    print(f"barrel layer radii (mm): {list(layer_r)}")
    print(f"primary stop (last_r): median={np.median(last_r[prim]):.0f}  "
          f"pctiles 25/50/75/90 = {np.percentile(last_r[prim],[25,50,75,90]).round(0)}")
    print(f"secondary birth (vr):  median={np.median(vr[sec]):.0f}  "
          f"pctiles 25/50/75/90 = {np.percentile(vr[sec],[25,50,75,90]).round(0)}")
    print(f"secondary first hit:   median={np.median(first_r[sec]):.0f}")
    # what layers bracket 460
    near = layer_r[(layer_r > 300) & (layer_r < 650)]
    print(f"barrel layers in 300-650mm (around the ~460 stop): {list(near)}")
    # fraction of primary stops within 20mm of ANY barrel layer (localized at silicon?)
    dmin = np.min(np.abs(last_r[prim][:, None] - layer_r[None, :]), axis=1)
    print(f"frac primary-stops within 20mm of a barrel layer: {np.mean(dmin < 20):.2f} "
          f"(vs uniform expectation ~{2*20*len(layer_r)/(layer_r.max()-layer_r.min()):.2f})")

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    bins = np.linspace(0, 1050, 106)
    ax[0].hist(last_r[prim], bins=bins, histtype="step", lw=2, label=f"primary stop (n={prim.sum()})", color="C0")
    ax[0].hist(vr[sec], bins=bins, histtype="step", lw=2, label=f"secondary birth vr (n={sec.sum()})", color="C1")
    for lr in layer_r:
        ax[0].axvline(lr, color="gray", alpha=0.35, lw=0.8)
    ax[0].set_xlabel("radius (mm)"); ax[0].set_title("where interactions happen (grey = barrel layers)"); ax[0].legend()
    # n_hits vs where the track lives (mean last_r per n_hits)
    nph = np.diff(off)
    ax[1].hist(nph[prim], bins=np.arange(0, 20), histtype="step", lw=2, label="primary", color="C0", density=True)
    ax[1].hist(nph[sec], bins=np.arange(0, 20), histtype="step", lw=2, label="secondary", color="C1", density=True)
    ax[1].set_xlabel("n_hits"); ax[1].set_title("n_hits: primary vs secondary"); ax[1].legend()
    plt.tight_layout(); fig.savefig(Path(args.outdir) / "material.png", dpi=100)
    print(f"\nwrote {Path(args.outdir) / 'material.png'}")


if __name__ == "__main__":
    main()
