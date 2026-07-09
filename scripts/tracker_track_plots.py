"""Visualise real vs generated tracks + feature histograms (tracker diagnostic).

Produces three PNGs in <ckpt_dir>/eval/:
  tracks.png       example real vs generated tracks in r-z and x-y planes
                   (connected in radial order, coloured by hit index) — shows
                   whether generated hits form ONE coherent helix or jitter/duplicate.
  track_feats.png  per-track feature histograms (real vs gen): r_mono, r_mean,
                   z_r_resid, phi_r_resid, dr_std, n_hits.
  hit_hists.png    pooled per-hit histograms (real vs gen): r, z, phi, layer —
                   sanity check for the r_mean discrepancy.
"""
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
from genpu.models.tracker_model import TrackerModel
from genpu.detector_geometry import LAYER_MEANS, LAYER_STDS, N_LAYERS


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
    ap.add_argument("--slice", default="/scratch/gpfs/IOJALVO/lv7805/genpu_data/tracker_slice/pion.npz")
    ap.add_argument("--n_tracks", type=int, default=8000)
    ap.add_argument("--use_vertex", action="store_true")
    ap.add_argument("--use_helix", action="store_true")
    ap.add_argument("--max_hits", type=int, default=32)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"; rng = np.random.default_rng(1)

    d = np.load(args.slice)
    cont, pdg, hits, off = d["cont"], d["pdg"], d["hits"], d["offsets"]
    norm = {"cont_mean": d["cont_mean"], "cont_std": d["cont_std"]}
    val = rng.permutation(cont.shape[0])[:args.n_tracks]
    model = TrackerModel(norm, use_vertex=args.use_vertex, use_helix=args.use_helix).to(dev)
    model.load_state_dict(torch.load(args.ckpt, map_location=dev)["model"]); model.eval()
    LM, LS = LAYER_MEANS, LAYER_STDS

    n_true = np.clip(off[val + 1] - off[val], 1, args.max_hits).astype(np.int64)
    contS = torch.as_tensor((cont[val] - norm["cont_mean"]) / norm["cont_std"], dtype=torch.float32, device=dev)
    pdgT = torch.as_tensor(pdg[val], dtype=torch.long, device=dev)
    vtxT = torch.as_tensor(cont[val][:, [5, 6]], dtype=torch.float32, device=dev)
    hlxT = TrackerModel.helix_params_from_cont(torch.as_tensor(cont[val], dtype=torch.float32, device=dev))

    real_tracks, gen_tracks = [], []   # each: (r,phi,z) arrays
    Xr, Xg = [], []
    rh, zh, ph, lh = [], [], [], []     # pooled real hit coords
    rhg, zhg, phg, lhg = [], [], [], []
    for s in range(0, len(val), 4096):
        e = min(s + 4096, len(val))
        with torch.no_grad():
            ce = model.cond_embed(contS[s:e], pdgT[s:e])
            nh = torch.as_tensor(n_true[s:e], device=dev)
            vtx = vtxT[s:e] if args.use_vertex else None
            hlx = hlxT[s:e] if args.use_helix else None
            gh, gl = model.tracker.generate(ce, nh, vertex_pos=vtx, helix_params=hlx)
        gh = gh.cpu().numpy(); gl = gl.cpu().numpy()
        for j in range(e - s):
            gi = s + j; k = int(n_true[gi])
            gr, gp, gz = gh[j, :k, 0], gh[j, :k, 1], gh[j, :k, 2]
            a, b = off[val[gi]], off[val[gi]] + k
            lc = hits[a:b, 0].astype(int).clip(0, N_LAYERS - 1)
            trr = hits[a:b, 1] * LS[lc, 0] + LM[lc, 0]
            trp = hits[a:b, 2] * LS[lc, 1] + LM[lc, 1]
            trz = hits[a:b, 3] * LS[lc, 2] + LM[lc, 2]
            fg = track_struct(gr, gp, gz); fr = track_struct(trr, trp, trz)
            if fg: Xg.append(fg)
            if fr: Xr.append(fr)
            rh.append(trr); zh.append(trz); ph.append(trp); lh.append(lc.astype(float))
            rhg.append(gr); zhg.append(gz); phg.append(gp); lhg.append(gl[j, :k].astype(float))
            if k >= 6 and len(real_tracks) < 6:
                real_tracks.append((trr, trp, trz))
            if k >= 6 and len(gen_tracks) < 6:
                gen_tracks.append((gr, gp, gz))
    Xr, Xg = np.array(Xr), np.array(Xg)
    outdir = Path(args.ckpt).parent / "eval"; outdir.mkdir(exist_ok=True)

    # ---- example tracks (r-z and x-y), real vs gen ----
    fig, ax = plt.subplots(2, 6, figsize=(20, 7))
    def draw(row, tracks, tag):
        for c in range(6):
            a = ax[row, c]
            if c < len(tracks):
                r, p, z = tracks[c]; o = np.argsort(r)
                r, p, z = r[o], p[o], z[o]
                a.plot(z, r, "-", color="0.7", lw=1, zorder=1)
                a.scatter(z, r, c=np.arange(len(r)), cmap="viridis", s=25, zorder=2)
                a.set_title(f"{tag} n={len(r)}", fontsize=9)
            a.set_xlabel("z [mm]"); a.set_ylabel("r [mm]"); a.set_ylim(0, 1100)
    draw(0, real_tracks, "REAL"); draw(1, gen_tracks, "GEN")
    plt.tight_layout(); fig.savefig(outdir / "tracks.png", dpi=100); plt.close(fig)

    # ---- per-track feature histograms ----
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

    # ---- pooled per-hit histograms ----
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
