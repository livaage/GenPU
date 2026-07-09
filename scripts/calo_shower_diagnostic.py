"""Diagnose what drives the calo head's residual: a PER-SHOWER two-sample test
with structure-revealing observables (not just pooled marginals).

Key feature: pos_energy_corr — correlation between a cell's distance-from-shower-
centroid and its log-energy. Real EM showers are core-hot (strong NEGATIVE corr);
our energy head conditions on the particle ONLY (decoupled from position to fix the
marginal), so generated energy is ~independent of position -> corr ~ 0. If this is
the dominant discriminator, the residual is the i.i.d.-points / energy-position
decoupling, and the fix is to re-introduce (careful) position-energy coupling.

Reports per-feature |Δ|/σ and both a linear and MLP two-sample AUC.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from genpu.flow.calo_flow import CaloFlow


def shower_features(pts, logE_col, counts):
    """Per-shower structure features. pts: (P,3)=[d_eta,d_phi,log_ecell]; counts per shower."""
    off = np.concatenate([[0], np.cumsum(counts)])
    feats = []
    for s in range(len(counts)):
        a, b = off[s], off[s + 1]
        deta, dphi, lE = pts[a:b, 0], pts[a:b, 1], pts[a:b, 2]
        E = np.exp(lE); w = E / (E.sum() + 1e-12)
        cx = (w * deta).sum(); cy = (w * dphi).sum()
        rc = np.sqrt((deta - cx) ** 2 + (dphi - cy) ** 2)     # dist from energy centroid
        width = np.sqrt((w * rc ** 2).sum())
        lead = E.max() / (E.sum() + 1e-12)                    # leading-cell energy fraction
        n = b - a
        # position-energy correlation (core-hot => negative). Needs >=3 points.
        if n >= 3 and rc.std() > 1e-6 and lE.std() > 1e-6:
            corr = float(np.corrcoef(rc, lE)[0, 1])
        else:
            corr = 0.0
        feats.append([n, np.log(E.sum() + 1e-12), width, lead, corr, lE.std()])
    return np.array(feats, np.float32)


def rank_auc(scores, labels):
    order = np.argsort(scores); ranks = np.empty_like(order, float); ranks[order] = np.arange(1, len(scores) + 1)
    pos = labels == 1; npos, nneg = pos.sum(), (~pos).sum()
    return 0.5 if npos == 0 or nneg == 0 else (ranks[pos].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--slice", default="/scratch/gpfs/IOJALVO/lv7805/genpu_data/calo_slice/photon_ctr.npz")
    ap.add_argument("--n_showers", type=int, default=20000)
    ap.add_argument("--steps", type=int, default=50)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(0)

    d = np.load(args.slice)
    norm = {k: d[k] for k in ["cont_mean", "cont_std", "glob_mean", "glob_std", "pts_mean", "pts_std"]}
    cont, pdg, glob, pts, off = d["cont"], d["pdg"], d["glob"], d["points_flat"], d["offsets"]
    S = cont.shape[0]
    val = rng.permutation(S)[:int(S * 0.05)][:args.n_showers]

    model = CaloFlow(norm).to(dev); model.load_state_dict(torch.load(args.ckpt, map_location=dev)["model"]); model.eval()

    # truth per-shower points
    t_n = np.array([off[i + 1] - off[i] for i in val])
    t_pts = np.concatenate([pts[off[i]:off[i + 1]] for i in val])

    # generate
    contS = torch.as_tensor((cont[val] - norm["cont_mean"]) / norm["cont_std"], dtype=torch.float32, device=dev)
    pdgT = torch.as_tensor(pdg[val], dtype=torch.long, device=dev)
    with torch.no_grad():
        ce = model.cond_embed(contS, pdgT); g_std = model.glob.sample(ce)
        g = model.unstd_glob(g_std).cpu().numpy()
    g_n = np.clip(np.round(np.exp(g[:, 1])).astype(int), 1, int(t_n.max()) + 5)
    rep = torch.as_tensor(np.repeat(np.arange(len(val)), g_n), device=dev)
    with torch.no_grad():
        pos_std = model.points.sample(ce[rep], g_std[rep], steps=args.steps)
        logE = model.energy.sample(ce[rep], model.log_floor)
        pos = model.unstd_pos(pos_std).cpu().numpy()
    g_pts = np.concatenate([pos, logE[:, None].cpu().numpy()], axis=1)

    Xr = shower_features(t_pts, 2, t_n)
    Xg = shower_features(g_pts, 2, g_n)
    names = ["n_points", "total_logE", "width", "lead_frac", "pos_energy_corr", "logE_std"]

    # per-feature separation
    print("=" * 60); print("CALO PER-SHOWER DIAGNOSTIC —", Path(args.ckpt).name); print("=" * 60)
    print(f"{'feature':18s} {'truth':>10s} {'gen':>10s} {'|Δ|/σ':>8s}")
    X = np.concatenate([Xr, Xg])
    for j, nm in enumerate(names):
        dd = abs(np.nanmean(Xr[:, j]) - np.nanmean(Xg[:, j])) / (np.nanstd(X[:, j]) + 1e-9)
        print(f"  {nm:16s} {np.nanmean(Xr[:,j]):10.3f} {np.nanmean(Xg[:,j]):10.3f} {dd:8.2f}")

    # two-sample classifiers (linear + MLP)
    y = np.concatenate([np.zeros(len(Xr)), np.ones(len(Xg))])
    Xs = np.nan_to_num((X - np.nanmean(X, 0)) / (np.nanstd(X, 0) + 1e-6))
    perm = rng.permutation(len(X)); ntr = len(X) // 2; tr, te = perm[:ntr], perm[ntr:]
    Xt = torch.as_tensor(Xs, dtype=torch.float32, device=dev); yt = torch.as_tensor(y, dtype=torch.float32, device=dev)
    for name, net in [("linear", torch.nn.Linear(X.shape[1], 1)),
                      ("mlp", torch.nn.Sequential(torch.nn.Linear(X.shape[1], 64), torch.nn.SiLU(),
                                                  torch.nn.Linear(64, 64), torch.nn.SiLU(), torch.nn.Linear(64, 1)))]:
        net = net.to(dev); opt = torch.optim.Adam(net.parameters(), lr=1e-3); tri = torch.as_tensor(tr, device=dev)
        for _ in range(600):
            opt.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(net(Xt[tri]).squeeze(-1), yt[tri])
            loss.backward(); opt.step()
        with torch.no_grad():
            s = net(Xt[torch.as_tensor(te, device=dev)]).squeeze(-1).cpu().numpy()
        print(f"  per-shower AUC ({name}): {rank_auc(s, y[te]):.4f}")


if __name__ == "__main__":
    main()
