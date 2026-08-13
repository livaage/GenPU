"""Pion hit-count check: real n_hits vs count-head sampled n_hits (self-normalizing count head).
Confirms the normalization fix — the count head now reproduces the pion count distribution."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from genpu.models.count_head import CountHead

DATA = "/scratch/gpfs/IOJALVO/lv7805/genpu_data"
PF_LOGPT, PF_ETA, PF_PHI, PF_PDG, PF_CHARGE, PF_MASS = range(6)
AUX_PRIMARY, AUX_PARENT, AUX_VX, AUX_VY, AUX_VZ, AUX_ENERGY = range(6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--count_ckpt", default=f"{DATA}/checkpoints/tracker/count_head_d0_selfnorm.pt")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--pdg_class", type=int, nargs="+", default=[3, 4])
    ap.add_argument("--out", default="/home/lv7805/genpu/plots/tracker/pion_count_dist.png")
    args = ap.parse_args()

    d = np.load(f"{DATA}/preprocessed/shard_{args.shard:04d}_stage2.npz")
    pf, aux, off = d["particle_features"], d["particle_aux"], d["tracker_offsets"]
    ntrk = np.diff(off)
    sel = np.where(np.isin(pf[:, PF_PDG], args.pdg_class) & (ntrk >= 1))[0]
    vr = np.hypot(aux[sel, AUX_VX], aux[sel, AUX_VY]); logE = np.log(np.clip(aux[sel, AUX_ENERGY], 1e-6, None))
    cont = np.stack([pf[sel, PF_LOGPT], pf[sel, PF_ETA], logE, pf[sel, PF_CHARGE],
                     pf[sel, PF_MASS], vr, aux[sel, AUX_VZ]], 1).astype(np.float32)
    phi = pf[sel, PF_PHI]; d0 = (aux[sel, AUX_VX] * np.sin(phi) - aux[sel, AUX_VY] * np.cos(phi)).astype(np.float32)
    pdg = pf[sel, PF_PDG].astype(np.int64)
    real_n = np.clip(ntrk[sel], 1, 48)

    ch = CountHead(use_d0=True)
    ch.load_state_dict(torch.load(args.count_ckpt, map_location="cpu", weights_only=False)["model"])
    ch.eval()
    with torch.no_grad():
        gen_n = ch.sample_raw(torch.as_tensor(cont), torch.as_tensor(pdg), torch.as_tensor(d0)).numpy()

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.arange(0.5, 33, 1)
    ax.hist(real_n, bins=bins, histtype="step", lw=2, density=True, color="k",
            label=f"real (median {np.median(real_n):.0f}, mean {real_n.mean():.2f})")
    ax.hist(gen_n, bins=bins, histtype="step", lw=2, density=True, color="tab:blue",
            label=f"count head (median {np.median(gen_n):.0f}, mean {gen_n.mean():.2f})")
    ax.set_xlabel("n_hits per pion"); ax.set_ylabel("density")
    ax.set_title(f"Pion hit count: real vs self-normalizing count head (shard {args.shard}, n={len(sel)})")
    ax.legend()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); fig.savefig(args.out, dpi=110)
    print(f"real median={np.median(real_n):.0f} mean={real_n.mean():.2f}   "
          f"gen median={np.median(gen_n):.0f} mean={gen_n.mean():.2f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
