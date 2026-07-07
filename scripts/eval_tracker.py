"""Evaluate the AR tracker head: sample hit sequences for val particles and
compare marginals against truth."""
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
from genpu.models.tracker_model import TrackerModel
from genpu.detector_geometry import LAYER_MEANS, LAYER_STDS, N_LAYERS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--slice", default="/scratch/gpfs/IOJALVO/lv7805/genpu_data/tracker_slice/pion.npz")
    ap.add_argument("--n_particles", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--val_frac", type=float, default=0.05)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    d = np.load(args.slice)
    norm = {k: d[k] for k in ["cont_mean", "cont_std"]}
    cont, pdg, hits, off = d["cont"], d["pdg"], d["hits"], d["offsets"]
    S = cont.shape[0]
    rng = np.random.default_rng(0)
    perm = rng.permutation(S); n_val = int(S * args.val_frac)
    val = perm[:n_val]
    if len(val) > args.n_particles:
        val = val[:args.n_particles]

    ck = torch.load(args.ckpt, map_location=dev)
    M = int(ck["args"].get("max_hits", 32)) if "args" in ck else 32
    model = TrackerModel(norm, max_hits=M).to(dev)
    model.load_state_dict(ck["model"]); model.eval()

    # ---- truth: per-particle hit sequences (already residual space) ----
    n_hits_truth = np.minimum(off[val + 1] - off[val], M).astype(np.int64)
    t_layers, t_resid = [], []   # per-hit layer_class and (r,phi,z,time) residuals
    for i in val:
        a, nh = off[i], min(int(off[i + 1] - off[i]), M)
        t_layers.append(hits[a:a + nh, 0].astype(np.int64))
        t_resid.append(hits[a:a + nh, 1:5])
    t_layers = np.concatenate(t_layers) if len(t_layers) else np.zeros(0, np.int64)
    t_resid = np.concatenate(t_resid) if len(t_resid) else np.zeros((0, 4), np.float32)

    # ---- generate (conditioned on truth n_hits: this spike has no count head) ----
    contS = ((cont[val] - norm["cont_mean"]) / norm["cont_std"]).astype(np.float32)
    LM = torch.as_tensor(LAYER_MEANS, dtype=torch.float32)
    LS = torch.as_tensor(LAYER_STDS, dtype=torch.float32)
    g_layers, g_resid, g_n = [], [], []
    with torch.no_grad():
        for s in range(0, len(val), args.batch):
            cS = torch.as_tensor(contS[s:s + args.batch], device=dev)
            pg = torch.as_tensor(pdg[val][s:s + args.batch], dtype=torch.long, device=dev)
            nh = torch.as_tensor(n_hits_truth[s:s + args.batch], device=dev)
            ce = model.cond_embed(cS, pg)
            phys, lay = model.tracker.generate(ce, nh)         # (B,Nmax,4), (B,Nmax)
            phys, lay, nh = phys.cpu(), lay.cpu(), nh.cpu()
            lc = lay.clamp(0, N_LAYERS - 1)
            resid = (phys - LM[lc]) / LS[lc]                    # back to residual space
            for b in range(phys.shape[0]):
                k = int(nh[b])
                g_layers.append(lay[b, :k].numpy())
                g_resid.append(resid[b, :k].numpy())
                g_n.append(k)
    g_layers = np.concatenate(g_layers) if len(g_layers) else np.zeros(0, np.int64)
    g_resid = np.concatenate(g_resid) if len(g_resid) else np.zeros((0, 4), np.float32)
    g_n = np.asarray(g_n, np.int64)

    # ---- metrics (1-D Wasserstein, lower=better) ----
    def W(a, b):
        return float(wasserstein_distance(a, b)) if len(a) and len(b) else float("nan")
    metrics = {
        "n_hits":      {"W": W(n_hits_truth, g_n),          "true_mean": float(n_hits_truth.mean()), "gen_mean": float(g_n.mean())},
        "layer_class": {"W": W(t_layers, g_layers),         "true_mean": float(t_layers.mean()),     "gen_mean": float(g_layers.mean())},
        "r_resid":     {"W": W(t_resid[:, 0], g_resid[:, 0]), "true_std": float(t_resid[:, 0].std()), "gen_std": float(g_resid[:, 0].std())},
        "phi_resid":   {"W": W(t_resid[:, 1], g_resid[:, 1]), "true_std": float(t_resid[:, 1].std()), "gen_std": float(g_resid[:, 1].std())},
        "z_resid":     {"W": W(t_resid[:, 2], g_resid[:, 2]), "true_std": float(t_resid[:, 2].std()), "gen_std": float(g_resid[:, 2].std())},
    }

    # ---- conditional table: mean hit-count binned by particle log_E ----
    part_logE = cont[val, 2]  # log_E is index 2 in CONT_FEATURES
    edges = np.quantile(part_logE, np.linspace(0, 1, 5))
    edges[-1] += 1e-6
    cond_rows = []
    for b in range(4):
        m = (part_logE >= edges[b]) & (part_logE < edges[b + 1])
        if m.sum() == 0:
            continue
        cond_rows.append({
            "bin": f"[{edges[b]:.1f},{edges[b+1]:.1f})", "n": int(m.sum()),
            "N_true": float(n_hits_truth[m].mean()), "N_gen": float(g_n[m].mean()),
        })
    metrics["conditional_by_particle_logE"] = cond_rows

    # ---- plots ----
    outdir = Path(args.ckpt).parent / "eval"; outdir.mkdir(exist_ok=True)
    fig, ax = plt.subplots(2, 3, figsize=(15, 9))
    def hist(a, td, gd, bins, title, xlabel):
        ax[a].hist(td, bins=bins, density=True, histtype="step", lw=2, label="truth")
        ax[a].hist(gd, bins=bins, density=True, histtype="step", lw=2, label="gen")
        ax[a].set_title(title); ax[a].set_xlabel(xlabel); ax[a].legend()
    nb = np.arange(0, min(int(n_hits_truth.max()), M) + 2) - 0.5
    hist((0, 0), n_hits_truth, g_n, nb, "hits / particle", "N")
    hist((0, 1), t_layers, g_layers, np.arange(N_LAYERS + 1) - 0.5, "layer occupancy", "layer_class")
    hist((0, 2), t_resid[:, 0], g_resid[:, 0], np.linspace(-4, 4, 60), "r residual", "(r - mean)/std")
    hist((1, 0), t_resid[:, 1], g_resid[:, 1], np.linspace(-4, 4, 60), "phi residual", "(phi - mean)/std")
    hist((1, 1), t_resid[:, 2], g_resid[:, 2], np.linspace(-4, 4, 60), "z residual", "(z - mean)/std")
    ctr = [0.5 * (edges[b] + edges[b + 1]) for b in range(len(cond_rows))]
    ax[1, 2].plot(ctr, [r["N_true"] for r in cond_rows], "o-", label="truth")
    ax[1, 2].plot(ctr, [r["N_gen"] for r in cond_rows], "s--", label="gen")
    ax[1, 2].set_title("conditional: hits vs particle logE"); ax[1, 2].set_xlabel("particle logE bin"); ax[1, 2].set_ylabel("mean hits"); ax[1, 2].legend()
    plt.tight_layout(); fig.savefig(outdir / "marginals.png", dpi=110); plt.close(fig)

    print("=" * 60); print("TRACKER eval —", Path(args.ckpt).name); print("=" * 60)
    print(f"val particles: {len(val)}   truth hits: {len(t_resid)}   gen hits: {len(g_resid)}")
    print("(gen conditioned on truth hit-count — this spike has no count head)")
    print("-- pooled marginals --")
    for k, v in metrics.items():
        if not isinstance(v, dict):
            continue
        extra = " ".join(f"{kk}={vv:.3f}" for kk, vv in v.items() if kk != "W")
        print(f"  {k:14s}  W={v['W']:.4f}   {extra}")
    print("-- conditional hit-count by particle logE --")
    print(f"  {'bin':16s} {'n':>6s} | {'N t/g':>13s}")
    for r in cond_rows:
        print(f"  {r['bin']:16s} {r['n']:6d} | {r['N_true']:5.2f}/{r['N_gen']:<5.2f}")
    print(f"\nplots -> {outdir/'marginals.png'}")
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
