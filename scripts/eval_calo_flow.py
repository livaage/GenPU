"""Evaluate the calo flow head: sample showers for val particles and compare
marginals against truth. This is the M2 spike A/B gate."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from genpu.flow.calo_flow import CaloFlow


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--slice", default="/scratch/gpfs/IOJALVO/lv7805/genpu_data/calo_slice/photon.npz")
    ap.add_argument("--n_showers", type=int, default=20000)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--val_frac", type=float, default=0.05)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d = np.load(args.slice)
    norm = {k: d[k] for k in ["cond_mean", "cond_std", "glob_mean", "glob_std", "pts_mean", "pts_std"]}
    cond, glob, pts, off = d["cond"], d["glob"], d["points_flat"], d["offsets"]
    S = cond.shape[0]
    rng = np.random.default_rng(0)
    perm = rng.permutation(S); n_val = int(S * args.val_frac)
    val = perm[:n_val]
    if len(val) > args.n_showers:
        val = val[:args.n_showers]

    model = CaloFlow(norm).to(dev)
    ck = torch.load(args.ckpt, map_location=dev)
    model.load_state_dict(ck["model"]); model.eval()

    # ---- truth marginals for val showers ----
    npt = np.diff(off)
    def truth_points(idx):
        segs = [pts[off[i]:off[i+1]] for i in idx]
        return np.concatenate(segs), np.array([len(s) for s in segs])
    t_pts, t_n = truth_points(val)
    t_totlogE = glob[val, 0]
    energy_mode = str(d["energy_mode"]) if "energy_mode" in d else "frac"
    log_floor = float(d["log_floor"]) if "log_floor" in d else -np.inf

    # ---- generate ----
    condS = torch.as_tensor((cond[val] - norm["cond_mean"]) / norm["cond_std"], dtype=torch.float32, device=dev)
    with torch.no_grad():
        g_std = model.glob.sample(condS)
        g = model.unstd_glob(g_std).cpu().numpy()            # [total_logE, log_n]
    g_totlogE = g[:, 0]
    g_n = np.clip(np.round(np.exp(g[:, 1])).astype(int), 1, int(t_n.max()) + 5)

    # per-point cond/glob repeated by generated N
    rep = np.repeat(np.arange(len(val)), g_n)
    with torch.no_grad():
        c_rep = condS[torch.as_tensor(rep, device=dev)]
        gg_rep = g_std[torch.as_tensor(rep, device=dev)]
        p_std = model.points.sample(c_rep, gg_rep, steps=args.steps)
        p = model.unstd_pts(p_std).cpu().numpy()             # [d_eta, d_phi, log_efrac]

    # rebuild per-point energy (GeV) and per-shower total, per energy_mode
    g_pts = p.copy()
    starts = np.concatenate([[0], np.cumsum(g_n)])
    e_point = np.empty(len(g_pts), np.float32)
    if energy_mode == "abs":
        # 3rd coord is absolute log E; clamp to the physical floor (-> point mass)
        g_pts[:, 2] = np.clip(g_pts[:, 2], log_floor, None)
        e_point = np.exp(g_pts[:, 2]).astype(np.float32)
        t_e = np.exp(t_pts[:, 2]).astype(np.float32)         # truth per-cell energy (GeV)
        # total shower E = sum of generated cells
        g_totlogE = np.array([np.log(e_point[starts[s]:starts[s+1]].sum() + 1e-12)
                              for s in range(len(val))], np.float32)
    else:
        for s in range(len(val)):
            a, b = starts[s], starts[s+1]
            if b <= a:
                continue
            fr = np.exp(g_pts[a:b, 2]); fr = fr / fr.sum()
            e_point[a:b] = fr * np.exp(g_totlogE[s])
        t_e = np.exp(t_pts[:, 2] + np.repeat(t_totlogE, t_n))  # truth per-point energy (GeV)

    # ---- metrics (1-D Wasserstein, lower=better) ----
    def W(a, b):
        return float(wasserstein_distance(a, b))
    metrics = {
        "n_points":   {"W": W(t_n, g_n),                 "true_mean": float(t_n.mean()),        "gen_mean": float(g_n.mean())},
        "total_logE": {"W": W(t_totlogE, g_totlogE),     "true_mean": float(t_totlogE.mean()),  "gen_mean": float(g_totlogE.mean())},
        "d_eta":      {"W": W(t_pts[:,0], g_pts[:,0]),   "true_std": float(t_pts[:,0].std()),   "gen_std": float(g_pts[:,0].std())},
        "d_phi":      {"W": W(t_pts[:,1], g_pts[:,1]),   "true_std": float(t_pts[:,1].std()),   "gen_std": float(g_pts[:,1].std())},
        "log_ecell":  {"W": W(np.log(t_e+1e-12), np.log(e_point+1e-12)), "true_mean": float(np.log(t_e+1e-12).mean()), "gen_mean": float(np.log(e_point+1e-12).mean())},
    }

    # ---- plots ----
    outdir = Path(args.ckpt).parent / "eval"; outdir.mkdir(exist_ok=True)
    fig, ax = plt.subplots(2, 3, figsize=(15, 9))
    def hist(a, td, gd, bins, title, xlabel, logy=False):
        ax[a].hist(td, bins=bins, density=True, histtype="step", lw=2, label="truth")
        ax[a].hist(gd, bins=bins, density=True, histtype="step", lw=2, label="gen")
        ax[a].set_title(title); ax[a].set_xlabel(xlabel); ax[a].legend()
        if logy: ax[a].set_yscale("log")
    nb = np.arange(0, min(int(t_n.max()), 40) + 2) - 0.5
    hist((0,0), t_n, g_n, nb, "points / shower", "N")
    hist((0,1), t_totlogE, g_totlogE, 40, "total shower log E", "log E [GeV]")
    hist((0,2), t_pts[:,0], g_pts[:,0], np.linspace(-2,2,60), "d_eta (localisation)", "cell_eta - part_eta")
    hist((1,0), t_pts[:,1], g_pts[:,1], np.linspace(-2,2,60), "d_phi (localisation)", "cell_phi - part_phi")
    hist((1,1), np.log(t_e+1e-12), np.log(e_point+1e-12), 60, "per-cell log E (incl. floor tail)", "log E [GeV]", logy=True)
    tr = np.sqrt(t_pts[:,0]**2 + t_pts[:,1]**2); gr = np.sqrt(g_pts[:,0]**2 + g_pts[:,1]**2)
    hist((1,2), tr, gr, np.linspace(0,2,60), "radial profile", "sqrt(d_eta^2+d_phi^2)")
    plt.tight_layout(); fig.savefig(outdir / "marginals.png", dpi=110); plt.close(fig)

    print("="*60); print("CALO FLOW eval —", Path(args.ckpt).name); print("="*60)
    print(f"val showers: {len(val)}   truth pts: {len(t_pts)}   gen pts: {len(g_pts)}")
    for k, v in metrics.items():
        extra = " ".join(f"{kk}={vv:.3f}" for kk, vv in v.items() if kk != "W")
        print(f"  {k:12s}  W={v['W']:.4f}   {extra}")
    print(f"\nplots -> {outdir/'marginals.png'}")
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
