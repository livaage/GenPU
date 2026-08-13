"""v4 helix-model track_feats: does the per-track coherence gap (z_r_resid, phi_r_resid spikes)
close vs v3? Real & gen positions = helix_ref[layer] + deviation (particle frame; r,z,phi-linearity
are frame-invariant). Compares to the v3 track_feats coherence smearing."""
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
from genpu.models.tracker_helix_model import TrackerHelixModel

DATA = "/scratch/gpfs/IOJALVO/lv7805/genpu_data"


def track_struct(r, phi, z):
    n = len(r)
    if n < 3:
        return None
    o = np.argsort(r); r, phi, z = r[o], np.unwrap(phi[o]), z[o]
    A = np.vstack([r, np.ones_like(r)]).T
    zres = np.sqrt(np.mean((z - A @ np.linalg.lstsq(A, z, rcond=None)[0]) ** 2))
    pres = np.sqrt(np.mean((phi - A @ np.linalg.lstsq(A, phi, rcond=None)[0]) ** 2))
    return [n, r.mean(), float(np.mean(np.diff(r) > 0)), zres, pres, float(np.diff(r).std())]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--slice", default=f"{DATA}/tracker_slice/helix_pion.npz")
    ap.add_argument("--n_tracks", type=int, default=8000)
    ap.add_argument("--max_hits", type=int, default=32)
    ap.add_argument("--outdir", default="/home/lv7805/genpu/plots/tracker/v4_eval")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"; rng = np.random.default_rng(1)

    d = np.load(args.slice)
    cont, pdg, hits, off = d["cont"], d["pdg"], d["hits"], d["offsets"]
    vxr, vyr = d["vxr"], d["vyr"]
    norm = {"cont_mean": d["cont_mean"], "cont_std": d["cont_std"]}
    model = TrackerHelixModel(norm, max_hits=args.max_hits).to(dev)
    model.load_state_dict(torch.load(args.ckpt, map_location=dev)["model"]); model.eval()

    val = rng.permutation(cont.shape[0])[:args.n_tracks]
    href = TrackerHelixModel.helix_ref(cont[val], vxr[val], vyr[val]).astype(np.float32)
    n_true = np.clip(off[val + 1] - off[val], 1, args.max_hits).astype(np.int64)
    contS = torch.as_tensor((cont[val] - norm["cont_mean"]) / norm["cont_std"], dtype=torch.float32, device=dev)
    pdgT = torch.as_tensor(pdg[val], dtype=torch.long, device=dev)

    Xr, Xg = [], []
    for s in range(0, len(val), 4096):
        e = min(s + 4096, len(val))
        hr = torch.as_tensor(href[s:e], device=dev)
        with torch.no_grad():
            ce = model.cond_embed(contS[s:e], pdgT[s:e])
            gh, gl = model.tracker.generate(ce, torch.as_tensor(n_true[s:e], device=dev), hr)
        gh = gh.cpu().numpy(); gl = gl.cpu().numpy(); hrn = href[s:e]
        for j in range(e - s):
            gi = s + j; k = int(n_true[gi])
            gr = np.hypot(gh[j, :k, 0], gh[j, :k, 1]); gp = np.arctan2(gh[j, :k, 1], gh[j, :k, 0]); gz = gh[j, :k, 2]
            a, b = off[val[gi]], off[val[gi]] + k
            lc = hits[a:b, 0].astype(int); dev3 = hits[a:b, 1:4]
            pos = hrn[j][lc] + dev3                                # real position = this particle's ref[layer] + dev
            rr = np.hypot(pos[:, 0], pos[:, 1]); pp = np.arctan2(pos[:, 1], pos[:, 0]); rz = pos[:, 2]
            fg = track_struct(gr, gp, gz); fr = track_struct(rr, pp, rz)
            if fg: Xg.append(fg)
            if fr: Xr.append(fr)
    Xr, Xg = np.array(Xr), np.array(Xg)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    names = ["n_hits", "r_mean", "r_mono", "z_r_resid", "phi_r_resid", "dr_std"]
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    for j, nm in enumerate(names):
        a = ax[j // 3, j % 3]
        lo = min(Xr[:, j].min(), Xg[:, j].min()); hi = max(Xr[:, j].max(), Xg[:, j].max())
        bins = np.linspace(lo, hi, 50)
        a.hist(Xr[:, j], bins=bins, density=True, histtype="step", lw=2, label="real")
        a.hist(Xg[:, j], bins=bins, density=True, histtype="step", lw=2, label="gen")
        a.set_title(nm); a.legend()
    fig.suptitle("v4 helix track_feats — coherence spikes (z_r_resid, phi_r_resid) vs v3", fontsize=13)
    plt.tight_layout(); fig.savefig(outdir / "track_feats.png", dpi=100)
    for j, nm in enumerate(names):
        print(f"  {nm:12s} real med {np.median(Xr[:,j]):.3f}  gen med {np.median(Xg[:,j]):.3f}")
    print("wrote", outdir / "track_feats.png")


if __name__ == "__main__":
    main()
