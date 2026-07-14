"""Evaluate the v2 state-carrying tracker. Key questions:
  1. STOP head: does the emergent n_hits distribution match real? (no truth n_hits used)
  2. STATE: does an explicit direction state fix coherence? half-persist -> real 0.70?
     and does it DEGRADE when the state is ablated? (proves the state does the work)
  3. Coherence spikes (r_mono, z_r_resid) and per-track two-sample AUC vs real.
Generation is fully self-driven (stop-based length, state fed back). No analytic geometry.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from genpu.models.tracker_state_ar import TrackerStateModel
from genpu.detector_geometry import LAYER_MEANS, LAYER_STDS, N_LAYERS


def _resid(x, y):
    A = np.vstack([x, np.ones_like(x)]).T
    return y - A @ np.linalg.lstsq(A, y, rcond=None)[0]


def track_feats(r, phi, z):
    if len(r) < 3:
        return None
    o = np.argsort(r); r, phi, z = r[o], np.unwrap(phi[o]), z[o]
    return [len(r), r.mean(), float(np.mean(np.diff(r) > 0)),
            float(np.sqrt(np.mean(_resid(r, z) ** 2))),
            float(np.sqrt(np.mean(_resid(r, phi) ** 2))), float(np.diff(r).std())]


def half_persist(tracks):
    ff, ss = [], []
    for r, phi, z in tracks:
        if len(r) < 6:
            continue
        o = np.argsort(r); r, z = r[o], z[o]; h = len(r) // 2
        if h < 3 or len(r) - h < 3:
            continue
        ff.append(np.sqrt(np.mean(_resid(r[:h], z[:h]) ** 2)))
        ss.append(np.sqrt(np.mean(_resid(r[h:], z[h:]) ** 2)))
    ff, ss = np.array(ff), np.array(ss)
    return float(np.corrcoef(ff, ss)[0, 1]) if len(ff) > 20 else float("nan"), len(ff)


def rank_auc(s, y):
    o = np.argsort(s); ra = np.empty_like(o, float); ra[o] = np.arange(1, len(s) + 1)
    p = y == 1; npo, nne = p.sum(), (~p).sum()
    return 0.5 if npo == 0 or nne == 0 else (ra[p].sum() - npo * (npo + 1) / 2) / (npo * nne)


def gen_tracks(model, ce, vtx, dev, ablate=False, bs=4096):
    out = []
    for s in range(0, ce.shape[0], bs):
        e = min(s + bs, ce.shape[0])
        with torch.no_grad():
            ph, _, ng = model.tracker.generate(ce[s:e], vtx[s:e], ablate_state=ablate)
        ph = ph.cpu().numpy(); ng = ng.cpu().numpy()
        for j in range(e - s):
            k = int(ng[j])
            if k >= 3:
                out.append((ph[j, :k, 0], ph[j, :k, 1], ph[j, :k, 2]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--slice", default="/scratch/gpfs/IOJALVO/lv7805/genpu_data/tracker_slice/v2_pion.npz")
    ap.add_argument("--n_tracks", type=int, default=15000)
    ap.add_argument("--state_feedback", action="store_true", help="match a feedback-trained ckpt")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"; rng = np.random.default_rng(0)
    d = np.load(args.slice)
    cont, pdg, hits, off = d["cont"], d["pdg"], d["hits"], d["offsets"]
    norm = {"cont_mean": d["cont_mean"], "cont_std": d["cont_std"]}
    val = rng.permutation(cont.shape[0])[:args.n_tracks]
    model = TrackerStateModel(norm, state_feedback=args.state_feedback).to(dev)
    model.load_state_dict(torch.load(args.ckpt, map_location=dev)["model"]); model.eval()
    LM, LS = LAYER_MEANS, LAYER_STDS

    contS = torch.as_tensor((cont[val] - norm["cont_mean"]) / norm["cont_std"], dtype=torch.float32, device=dev)
    pdgT = torch.as_tensor(pdg[val], dtype=torch.long, device=dev)
    vtx = torch.as_tensor(cont[val][:, [5, 6]], dtype=torch.float32, device=dev)
    with torch.no_grad():
        ce = model.cond_embed(contS, pdgT)

    # real tracks + real n_hits
    real, real_n = [], []
    for gi in val:
        a, b = off[gi], off[gi + 1]; real_n.append(b - a)
        lc = hits[a:b, 0].astype(int).clip(0, N_LAYERS - 1)
        real.append((hits[a:b, 1] * LS[lc, 0] + LM[lc, 0], hits[a:b, 2] * LS[lc, 1] + LM[lc, 1],
                     hits[a:b, 3] * LS[lc, 2] + LM[lc, 2]))
    gen = gen_tracks(model, ce, vtx, dev, ablate=False)
    genA = gen_tracks(model, ce, vtx, dev, ablate=True)
    gen_n = [len(t[0]) for t in gen]

    print("=" * 62); print("TRACKER v2 EVAL —", Path(args.ckpt).name); print("=" * 62)
    # 1. STOP head -> emergent n_hits
    rn, gn = np.array(real_n), np.array([len(t[0]) for t in real])  # gn placeholder
    print(f"n_hits (STOP head, emergent):  real mean={np.mean(real_n):.2f} median={np.median(real_n):.0f}"
          f"   gen mean={np.mean(gen_n):.2f} median={np.median(gen_n):.0f}   (n gen>=3: {len(gen)})")
    # 2. STATE ablation -> half-persist
    hp_r, _ = half_persist(real); hp_g, _ = half_persist(gen); hp_a, _ = half_persist(genA)
    print(f"half-persist:  real={hp_r:.3f}   gen={hp_g:.3f}   gen(state ABLATED)={hp_a:.3f}"
          f"   [v1 was 0.58, target {hp_r:.2f}]")
    # 3. coherence + AUC
    Xr = np.array([f for t in real if (f := track_feats(*t))])
    Xg = np.array([f for t in gen if (f := track_feats(*t))])
    names = ["n_hits", "r_mean", "r_mono", "z_r_resid", "phi_r_resid", "dr_std"]
    print(f"{'feature':14s} {'real':>10s} {'gen':>10s}")
    for j, nm in enumerate(names):
        print(f"  {nm:12s} {Xr[:,j].mean():10.4f} {Xg[:,j].mean():10.4f}")
    n = min(len(Xr), len(Xg)); X = np.concatenate([Xr[:n], Xg[:n]]); y = np.concatenate([np.zeros(n), np.ones(n)])
    Xs = np.nan_to_num((X - X.mean(0)) / (X.std(0) + 1e-6))
    perm = rng.permutation(len(X)); tr, te = perm[:len(X) // 2], perm[len(X) // 2:]
    Xt = torch.as_tensor(Xs, dtype=torch.float32, device=dev); yt = torch.as_tensor(y, dtype=torch.float32, device=dev)
    net = torch.nn.Sequential(torch.nn.Linear(6, 64), torch.nn.SiLU(), torch.nn.Linear(64, 1)).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3); tri = torch.as_tensor(tr, device=dev)
    for _ in range(600):
        opt.zero_grad(); l = torch.nn.functional.binary_cross_entropy_with_logits(net(Xt[tri]).squeeze(-1), yt[tri]); l.backward(); opt.step()
    with torch.no_grad():
        sc = net(Xt[torch.as_tensor(te, device=dev)]).squeeze(-1).cpu().numpy()
    print(f"per-track AUC (mlp): {rank_auc(sc, y[te]):.4f}   [v1 512-bin was 0.80]")


if __name__ == "__main__":
    main()
