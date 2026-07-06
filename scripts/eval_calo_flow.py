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

    # ---- per-PARTICLE diagnostics ----
    # (a) per-shower observables: energy-weighted width + centroid offset.
    #     Tests the i.i.d.-points assumption directly (pooled marginals can hide this).
    def shower_stats(pts, counts, e):
        off = np.concatenate([[0], np.cumsum(counts)])[:-1]
        r2 = pts[:, 0] ** 2 + pts[:, 1] ** 2
        se = np.add.reduceat(e, off)
        se = np.clip(se, 1e-30, None)
        width = np.sqrt(np.clip(np.add.reduceat(e * r2, off) / se, 0, None))
        cx = np.add.reduceat(e * pts[:, 0], off) / se
        cy = np.add.reduceat(e * pts[:, 1], off) / se
        return width, np.sqrt(cx ** 2 + cy ** 2)
    t_w, t_co = shower_stats(t_pts, t_n, t_e)
    g_w, g_co = shower_stats(g_pts, g_n, e_point)
    metrics["shower_width"] = {"W": W(t_w, g_w), "true_mean": float(t_w.mean()), "gen_mean": float(g_w.mean())}
    metrics["centroid_off"] = {"W": W(t_co, g_co), "true_mean": float(t_co.mean()), "gen_mean": float(g_co.mean())}

    # (b) conditional response: bin the SAME particles by their log-E, compare
    #     truth vs gen response WITHIN each bin (does response track conditioning?).
    part_logE = cond[val, 2]
    edges = np.quantile(part_logE, np.linspace(0, 1, 5))
    edges[-1] += 1e-6
    cond_rows = []
    for b in range(4):
        m = (part_logE >= edges[b]) & (part_logE < edges[b + 1])
        if m.sum() == 0:
            continue
        cond_rows.append({
            "bin": f"[{edges[b]:.1f},{edges[b+1]:.1f})", "n": int(m.sum()),
            "N_true": float(t_n[m].mean()), "N_gen": float(g_n[m].mean()),
            "logE_true": float(t_totlogE[m].mean()), "logE_gen": float(g_totlogE[m].mean()),
            "w_true": float(t_w[m].mean()), "w_gen": float(g_w[m].mean()),
        })
    metrics["conditional_by_particle_logE"] = cond_rows

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

    # per-particle figure: per-shower observables + conditional response
    fig2, ax2 = plt.subplots(1, 3, figsize=(15, 4.5))
    ax2[0].hist(t_w, bins=np.linspace(0, 1.2, 60), density=True, histtype="step", lw=2, label="truth")
    ax2[0].hist(g_w, bins=np.linspace(0, 1.2, 60), density=True, histtype="step", lw=2, label="gen")
    ax2[0].set_title("per-shower energy-weighted width"); ax2[0].set_xlabel("width"); ax2[0].legend()
    ax2[1].hist(t_co, bins=np.linspace(0, 1.0, 60), density=True, histtype="step", lw=2, label="truth")
    ax2[1].hist(g_co, bins=np.linspace(0, 1.0, 60), density=True, histtype="step", lw=2, label="gen")
    ax2[1].set_title("per-shower centroid offset"); ax2[1].set_xlabel("|centroid - particle dir|"); ax2[1].legend()
    ctr = [0.5 * (edges[b] + edges[b + 1]) for b in range(len(cond_rows))]
    ax2[2].plot(ctr, [r["logE_true"] for r in cond_rows], "o-", label="truth")
    ax2[2].plot(ctr, [r["logE_gen"] for r in cond_rows], "s--", label="gen")
    ax2[2].set_title("conditional: shower logE vs particle logE"); ax2[2].set_xlabel("particle logE bin"); ax2[2].set_ylabel("shower logE"); ax2[2].legend()
    plt.tight_layout(); fig2.savefig(outdir / "per_particle.png", dpi=110); plt.close(fig2)

    print("="*60); print("CALO FLOW eval —", Path(args.ckpt).name); print("="*60)
    print(f"val showers: {len(val)}   truth pts: {len(t_pts)}   gen pts: {len(g_pts)}")
    print("-- pooled marginals --")
    for k, v in metrics.items():
        if not isinstance(v, dict):
            continue
        extra = " ".join(f"{kk}={vv:.3f}" for kk, vv in v.items() if kk != "W")
        print(f"  {k:14s}  W={v['W']:.4f}   {extra}")
    print("-- conditional response by particle logE (per-particle fidelity) --")
    print(f"  {'bin':16s} {'n':>6s} | {'N t/g':>13s} | {'logE t/g':>15s} | {'width t/g':>13s}")
    for r in cond_rows:
        print(f"  {r['bin']:16s} {r['n']:6d} | {r['N_true']:5.2f}/{r['N_gen']:<5.2f}   "
              f"| {r['logE_true']:6.2f}/{r['logE_gen']:<6.2f}  | {r['w_true']:5.3f}/{r['w_gen']:<5.3f}")
    print(f"\nplots -> {outdir/'marginals.png'} , {outdir/'per_particle.png'}")
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
