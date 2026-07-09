"""Visualise real vs generated calo showers + feature histograms.

Three PNGs (default -> repo plots/calo/):
  showers.png      example real vs gen showers in the (d_eta,d_phi) plane, marker
                   size/colour ~ cell energy — shows compactness + core-hot shape.
  shower_feats.png per-shower feature histograms (real vs gen): n_points,
                   total_logE, width, lead_frac, pos_energy_corr, logE_std.
  cell_hists.png   pooled per-cell histograms (real vs gen): d_eta, d_phi, log_ecell.
Uses the best calo model (core latent + mixture energy) with the photon_core slice.
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
from genpu.flow.calo_flow import CaloFlow


def feats(deta, dphi, lE):
    E = np.exp(lE); w = E / (E.sum() + 1e-12)
    cx = (w * deta).sum(); cy = (w * dphi).sum()
    rc = np.sqrt((deta - cx) ** 2 + (dphi - cy) ** 2)
    width = np.sqrt((w * rc ** 2).sum()); lead = E.max() / (E.sum() + 1e-12)
    n = len(E)
    corr = float(np.corrcoef(rc, lE)[0, 1]) if (n >= 3 and rc.std() > 1e-6 and lE.std() > 1e-6) else 0.0
    return [n, np.log(E.sum() + 1e-12), width, lead, corr, lE.std()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/scratch/gpfs/IOJALVO/lv7805/genpu_data/checkpoints/calo_flow/photon_coremix_v1/checkpoint_040000.pt")
    ap.add_argument("--slice", default="/scratch/gpfs/IOJALVO/lv7805/genpu_data/calo_slice/photon_core.npz")
    ap.add_argument("--outdir", default="/home/lv7805/genpu/plots/calo")
    ap.add_argument("--n_showers", type=int, default=15000)
    ap.add_argument("--steps", type=int, default=50)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"; rng = np.random.default_rng(1)

    d = np.load(args.slice)
    norm = {k: d[k] for k in ["cont_mean", "cont_std", "glob_mean", "glob_std", "pts_mean", "pts_std"]}
    cont, pdg, glob, pts, off = d["cont"], d["pdg"], d["glob"], d["points_flat"], d["offsets"]
    val = rng.permutation(cont.shape[0])[:args.n_showers]
    model = CaloFlow(norm).to(dev); model.load_state_dict(torch.load(args.ckpt, map_location=dev)["model"]); model.eval()

    # truth: deltas + per-shower core (glob[:,2:4]) -> absolute (d_eta,d_phi) rel. particle
    t_n = np.array([off[i + 1] - off[i] for i in val])
    t_pts = np.concatenate([pts[off[i]:off[i + 1]] for i in val])
    t_core = glob[val, 2:4]
    t_abs = t_pts.copy(); t_abs[:, :2] += np.repeat(t_core, t_n, axis=0)

    # generate
    contS = torch.as_tensor((cont[val] - norm["cont_mean"]) / norm["cont_std"], dtype=torch.float32, device=dev)
    pdgT = torch.as_tensor(pdg[val], dtype=torch.long, device=dev)
    with torch.no_grad():
        ce = model.cond_embed(contS, pdgT); g_std = model.glob.sample(ce)
        g = model.unstd_glob(g_std).cpu().numpy()
    g_n = np.clip(np.round(np.exp(g[:, 1])).astype(int), 1, int(t_n.max()) + 5)
    rep = torch.as_tensor(np.repeat(np.arange(len(val)), g_n), device=dev)
    with torch.no_grad():
        pos = model.unstd_pos(model.points.sample(ce[rep], g_std[rep][:, :2], steps=args.steps)).cpu().numpy()
        lE = model.energy.sample(ce[rep], model.log_floor).cpu().numpy()
    g_core = g[:, 2:4]
    g_abs = np.column_stack([pos + np.repeat(g_core, g_n, axis=0), lE])

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # ---- example showers (d_eta, d_phi), size/colour ~ energy ----
    def pick(nn, ptsflat, k=6, nmin=4):
        segs, o = [], np.concatenate([[0], np.cumsum(nn)])
        for s in range(len(nn)):
            if nn[s] >= nmin and len(segs) < k:
                segs.append(ptsflat[o[s]:o[s + 1]])
        return segs
    reals, gens = pick(t_n, t_abs), pick(g_n, g_abs)
    fig, ax = plt.subplots(2, 6, figsize=(20, 7))
    def draw(row, segs, tag):
        for c in range(6):
            a = ax[row, c]
            if c < len(segs):
                deta, dphi, le = segs[c][:, 0], segs[c][:, 1], segs[c][:, 2]
                sz = 20 + 300 * (np.exp(le) / (np.exp(le).max() + 1e-12))
                sc = a.scatter(deta, dphi, s=sz, c=le, cmap="plasma")
                a.set_title(f"{tag} n={len(le)}", fontsize=9)
            a.set_xlim(-1.5, 1.5); a.set_ylim(-1.5, 1.5)
            a.set_xlabel("d_eta"); a.set_ylabel("d_phi")
    draw(0, reals, "REAL"); draw(1, gens, "GEN")
    plt.tight_layout(); fig.savefig(outdir / "showers.png", dpi=100); plt.close(fig)

    # ---- per-shower feature histograms ----
    Xr = np.array([feats(t_abs[o:o + n, 0], t_abs[o:o + n, 1], t_abs[o:o + n, 2])
                   for o, n in zip(np.concatenate([[0], np.cumsum(t_n)])[:-1], t_n)])
    Xg = np.array([feats(g_abs[o:o + n, 0], g_abs[o:o + n, 1], g_abs[o:o + n, 2])
                   for o, n in zip(np.concatenate([[0], np.cumsum(g_n)])[:-1], g_n)])
    names = ["n_points", "total_logE", "width", "lead_frac", "pos_energy_corr", "logE_std"]
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    for j, nm in enumerate(names):
        a = ax[j // 3, j % 3]
        lo = min(Xr[:, j].min(), Xg[:, j].min()); hi = max(Xr[:, j].max(), Xg[:, j].max())
        bins = np.linspace(lo, hi, 50)
        a.hist(Xr[:, j], bins=bins, density=True, histtype="step", lw=2, label="real")
        a.hist(Xg[:, j], bins=bins, density=True, histtype="step", lw=2, label="gen")
        a.set_title(nm); a.legend()
    plt.tight_layout(); fig.savefig(outdir / "shower_feats.png", dpi=100); plt.close(fig)

    # ---- pooled per-cell histograms ----
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    for a, (rv, gv, t, bins) in zip(ax, [
        (t_abs[:, 0], g_abs[:, 0], "cell d_eta", np.linspace(-2, 2, 60)),
        (t_abs[:, 1], g_abs[:, 1], "cell d_phi", np.linspace(-2, 2, 60)),
        (t_abs[:, 2], g_abs[:, 2], "cell log_E", np.linspace(-11, -4, 60))]):
        a.hist(rv, bins=bins, density=True, histtype="step", lw=2, label="real")
        a.hist(gv, bins=bins, density=True, histtype="step", lw=2, label="gen")
        a.set_title(t); a.legend()
        if "log_E" in t: a.set_yscale("log")
    plt.tight_layout(); fig.savefig(outdir / "cell_hists.png", dpi=100); plt.close(fig)
    print("wrote", outdir / "showers.png", outdir / "shower_feats.png", outdir / "cell_hists.png")


if __name__ == "__main__":
    main()
