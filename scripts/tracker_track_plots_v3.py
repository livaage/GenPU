"""Standard tracker diagnostic plots for the v3 SURFACE-LOCAL model (tracks / track_feats /
hit_hists), the v3 counterpart of tracker_track_plots.py. Real & generated hits are mapped to
physical (r, phi, z, layer) via ModuleGeometry so the plots match the v1 format."""
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
from genpu.models.tracker_module_model import TrackerModuleModel
from genpu.module_geometry import ModuleGeometry

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
    ap.add_argument("--module_geometry", default=f"{DATA}/module_geometry.npz")
    ap.add_argument("--slice", default=f"{DATA}/tracker_slice/surface_multispecies.npz")
    ap.add_argument("--n_tracks", type=int, default=8000)
    ap.add_argument("--max_hits", type=int, default=32)
    ap.add_argument("--outdir", default="/home/lv7805/genpu/plots/tracker/v3_ms_eval")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"; rng = np.random.default_rng(1)

    d = np.load(args.slice)
    cont, pdg, hits, off = d["cont"], d["pdg"], d["hits"], d["offsets"]
    norm = {"cont_mean": d["cont_mean"], "cont_std": d["cont_std"]}
    mg = ModuleGeometry(args.module_geometry)
    model = TrackerModuleModel(norm, module_geometry_path=args.module_geometry, max_hits=args.max_hits).to(dev)
    model.load_state_dict(torch.load(args.ckpt, map_location=dev)["model"]); model.eval()
    lom = torch.as_tensor(mg.layer_class, dtype=torch.long, device=dev)

    val = rng.permutation(cont.shape[0])[:args.n_tracks]
    n_true = np.clip(off[val + 1] - off[val], 1, args.max_hits).astype(np.int64)
    contS = torch.as_tensor((cont[val] - norm["cont_mean"]) / norm["cont_std"], dtype=torch.float32, device=dev)
    pdgT = torch.as_tensor(pdg[val], dtype=torch.long, device=dev)

    real_tracks, gen_tracks, Xr, Xg = [], [], [], []
    rh, zh, ph, lh, rhg, zhg, phg, lhg = ([] for _ in range(8))
    for s in range(0, len(val), 4096):
        e = min(s + 4096, len(val))
        with torch.no_grad():
            ce = model.cond_embed(contS[s:e], pdgT[s:e])
            phys, gmod = model.tracker.generate(ce, torch.as_tensor(n_true[s:e], device=dev))
            glay = lom[gmod]
        phys = phys.cpu().numpy(); glay = glay.cpu().numpy()
        for j in range(e - s):
            gi = s + j; k = int(n_true[gi])
            gx, gy, gz = phys[j, :k, 0], phys[j, :k, 1], phys[j, :k, 2]
            gr = np.hypot(gx, gy); gp = np.arctan2(gy, gx)
            a, b = off[val[gi]], off[val[gi]] + k
            modblk = hits[a:b, 0].astype(np.int64)
            rp = mg.to_physical(modblk, hits[a:b, 1:5])          # (k,4) physical x,y,z,time
            trr = np.hypot(rp[:, 0], rp[:, 1]); trp = np.arctan2(rp[:, 1], rp[:, 0]); trz = rp[:, 2]
            trl = mg.layer_class[modblk].astype(float)
            fg = track_struct(gr, gp, gz); fr = track_struct(trr, trp, trz)
            if fg: Xg.append(fg)
            if fr: Xr.append(fr)
            rh.append(trr); zh.append(trz); ph.append(trp); lh.append(trl)
            rhg.append(gr); zhg.append(gz); phg.append(gp); lhg.append(glay[j, :k].astype(float))
            if k >= 6 and len(real_tracks) < 6: real_tracks.append((trr, trp, trz))
            if k >= 6 and len(gen_tracks) < 6: gen_tracks.append((gr, gp, gz))
    Xr, Xg = np.array(Xr), np.array(Xg)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(2, 6, figsize=(20, 7))
    def draw(row, tracks, tag):
        for c in range(6):
            a = ax[row, c]
            if c < len(tracks):
                r, p, z = tracks[c]; o = np.argsort(r); r, z = r[o], z[o]
                a.plot(z, r, "-", color="0.7", lw=1, zorder=1)
                a.scatter(z, r, c=np.arange(len(r)), cmap="viridis", s=25, zorder=2)
                a.set_title(f"{tag} n={len(r)}", fontsize=9)
            a.set_xlabel("z [mm]"); a.set_ylabel("r [mm]"); a.set_ylim(0, 1100)
    draw(0, real_tracks, "REAL"); draw(1, gen_tracks, "GEN")
    plt.tight_layout(); fig.savefig(outdir / "tracks.png", dpi=100); plt.close(fig)

    names = ["n_hits", "r_mean", "r_mono", "z_r_resid", "phi_r_resid", "dr_std"]
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    for j, nm in enumerate(names):
        a = ax[j // 3, j % 3]
        lo = min(Xr[:, j].min(), Xg[:, j].min()); hi = max(Xr[:, j].max(), Xg[:, j].max())
        bins = np.linspace(lo, hi, 50)
        a.hist(Xr[:, j], bins=bins, density=True, histtype="step", lw=2, label="real")
        a.hist(Xg[:, j], bins=bins, density=True, histtype="step", lw=2, label="gen")
        a.set_title(nm); a.legend()
    plt.tight_layout(); fig.savefig(outdir / "track_feats.png", dpi=100); plt.close(fig)

    rh, zh, ph, lh = map(np.concatenate, (rh, zh, ph, lh))
    rhg, zhg, phg, lhg = map(np.concatenate, (rhg, zhg, phg, lhg))
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    for a, (rv, gv, t, bins) in zip(ax.ravel(), [
        (rh, rhg, "hit r [mm]", np.linspace(0, 1100, 60)),
        (zh, zhg, "hit z [mm]", np.linspace(-3000, 3000, 60)),
        (ph, phg, "hit phi", np.linspace(-3.2, 3.2, 60)),
        (lh, lhg, "hit layer", np.arange(0, 49) - 0.5)]):
        a.hist(rv, bins=bins, density=True, histtype="step", lw=2, label="real")
        a.hist(gv, bins=bins, density=True, histtype="step", lw=2, label="gen")
        a.set_title(t); a.legend()
    plt.tight_layout(); fig.savefig(outdir / "hit_hists.png", dpi=100); plt.close(fig)
    print(f"pooled hit r_mean: real {rh.mean():.1f}  gen {rhg.mean():.1f}")
    print(f"per-track r_mean:  real {Xr[:,1].mean():.1f}  gen {Xg[:,1].mean():.1f}")
    print("wrote", outdir / "tracks.png", outdir / "track_feats.png", outdir / "hit_hists.png")


if __name__ == "__main__":
    main()
