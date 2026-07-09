"""Diagnose the tracker's remaining ~0.81: a PER-TRACK two-sample test with
helix-COHERENCE features (the event gate's marginals are all matched, so the
residual must be per-track structure the event features can't see).

A real charged track's hits lie on ONE smooth helix: r increases monotonically
inner->outer, z is ~linear in r, phi turns smoothly with r. If generated hits
are on plausible layers but don't form a coherent trajectory, these will differ:
  r_mono      fraction of consecutive steps with r increasing
  z_r_resid   RMS residual of a z-vs-r linear fit (helix -> small)
  phi_r_resid RMS residual of an (unwrapped) phi-vs-r linear fit
  dr_std      spread of consecutive r-steps
Reports per-feature |Δ|/σ and linear + MLP two-sample AUC.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from genpu.models.tracker_model import TrackerModel
from genpu.detector_geometry import LAYER_MEANS, LAYER_STDS, N_LAYERS


def track_feats(r, phi, z):
    """Per-track helix-coherence features from physical (r,phi,z), sorted by r."""
    n = len(r)
    if n < 3:
        return None
    order = np.argsort(r); r, phi, z = r[order], np.unwrap(phi[order]), z[order]
    r_mono = float(np.mean(np.diff(r) > 0))
    dr = np.diff(r); dr_std = float(dr.std())
    # linear fits vs r
    A = np.vstack([r, np.ones_like(r)]).T
    z_res = float(np.sqrt(np.mean((z - A @ np.linalg.lstsq(A, z, rcond=None)[0]) ** 2)))
    phi_res = float(np.sqrt(np.mean((phi - A @ np.linalg.lstsq(A, phi, rcond=None)[0]) ** 2)))
    return [n, r.mean(), r_mono, z_res, phi_res, dr_std]


def rank_auc(s, y):
    o = np.argsort(s); ra = np.empty_like(o, float); ra[o] = np.arange(1, len(s) + 1)
    p = y == 1; npo, nne = p.sum(), (~p).sum()
    return 0.5 if npo == 0 or nne == 0 else (ra[p].sum() - npo * (npo + 1) / 2) / (npo * nne)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--slice", default="/scratch/gpfs/IOJALVO/lv7805/genpu_data/tracker_slice/pion.npz")
    ap.add_argument("--n_tracks", type=int, default=15000)
    ap.add_argument("--use_vertex", action="store_true")
    ap.add_argument("--use_helix", action="store_true")
    ap.add_argument("--cont_temp", type=float, default=1.0, help="sampling temperature for r/phi/z/time")
    ap.add_argument("--max_hits", type=int, default=32)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"; rng = np.random.default_rng(0)

    d = np.load(args.slice)
    cont, pdg, hits, off = d["cont"], d["pdg"], d["hits"], d["offsets"]
    norm = {"cont_mean": d["cont_mean"], "cont_std": d["cont_std"]}
    S = cont.shape[0]
    val = rng.permutation(S)[:args.n_tracks]

    model = TrackerModel(norm, use_vertex=args.use_vertex, use_helix=args.use_helix).to(dev)
    model.load_state_dict(torch.load(args.ckpt, map_location=dev)["model"]); model.eval()
    LM = LAYER_MEANS; LS = LAYER_STDS

    # truth physical (r,phi,z) per track from residuals + layer stats
    Xr, Xg = [], []
    n_true = np.clip(off[val + 1] - off[val], 1, args.max_hits).astype(np.int64)
    contS = torch.as_tensor((cont[val] - norm["cont_mean"]) / norm["cont_std"], dtype=torch.float32, device=dev)
    pdgT = torch.as_tensor(pdg[val], dtype=torch.long, device=dev)
    vtxT = torch.as_tensor(cont[val][:, [5, 6]], dtype=torch.float32, device=dev)
    hlxT = TrackerModel.helix_params_from_cont(torch.as_tensor(cont[val], dtype=torch.float32, device=dev))
    for s in range(0, len(val), 4096):
        e = min(s + 4096, len(val))
        with torch.no_grad():
            ce = model.cond_embed(contS[s:e], pdgT[s:e])
            nh = torch.as_tensor(n_true[s:e], device=dev)
            vtx = vtxT[s:e] if args.use_vertex else None
            hlx = hlxT[s:e] if args.use_helix else None
            gh, gl = model.tracker.generate(ce, nh, vertex_pos=vtx, helix_params=hlx, cont_temp=args.cont_temp)
        gh = gh.cpu().numpy()
        for j in range(e - s):
            gi = s + j; k = int(n_true[gi])
            # gen physical
            f = track_feats(gh[j, :k, 0], gh[j, :k, 1], gh[j, :k, 2])
            if f: Xg.append(f)
            # truth physical from residuals
            a, b = off[val[gi]], off[val[gi]] + k
            lc = hits[a:b, 0].astype(int).clip(0, N_LAYERS - 1)
            rr = hits[a:b, 1] * LS[lc, 0] + LM[lc, 0]
            pp = hits[a:b, 2] * LS[lc, 1] + LM[lc, 1]
            zz = hits[a:b, 3] * LS[lc, 2] + LM[lc, 2]
            f = track_feats(rr, pp, zz)
            if f: Xr.append(f)
    Xr, Xg = np.array(Xr), np.array(Xg)
    names = ["n_hits", "r_mean", "r_mono", "z_r_resid", "phi_r_resid", "dr_std"]

    print("=" * 60); print("TRACKER PER-TRACK DIAGNOSTIC —", Path(args.ckpt).name); print("=" * 60)
    print(f"tracks: truth {len(Xr)}  gen {len(Xg)}")
    print(f"{'feature':16s} {'truth':>12s} {'gen':>12s} {'|Δ|/σ':>8s}")
    X = np.concatenate([Xr, Xg])
    for j, nm in enumerate(names):
        dd = abs(Xr[:, j].mean() - Xg[:, j].mean()) / (X[:, j].std() + 1e-9)
        print(f"  {nm:14s} {Xr[:,j].mean():12.4f} {Xg[:,j].mean():12.4f} {dd:8.2f}")

    y = np.concatenate([np.zeros(len(Xr)), np.ones(len(Xg))])
    Xs = np.nan_to_num((X - X.mean(0)) / (X.std(0) + 1e-6))
    perm = rng.permutation(len(X)); ntr = len(X) // 2; tr, te = perm[:ntr], perm[ntr:]
    Xt = torch.as_tensor(Xs, dtype=torch.float32, device=dev); yt = torch.as_tensor(y, dtype=torch.float32, device=dev)
    for nm, net in [("linear", torch.nn.Linear(X.shape[1], 1)),
                    ("mlp", torch.nn.Sequential(torch.nn.Linear(X.shape[1], 64), torch.nn.SiLU(),
                                                torch.nn.Linear(64, 64), torch.nn.SiLU(), torch.nn.Linear(64, 1)))]:
        net = net.to(dev); opt = torch.optim.Adam(net.parameters(), lr=1e-3); tri = torch.as_tensor(tr, device=dev)
        for _ in range(600):
            opt.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(net(Xt[tri]).squeeze(-1), yt[tri]); loss.backward(); opt.step()
        with torch.no_grad():
            sc = net(Xt[torch.as_tensor(te, device=dev)]).squeeze(-1).cpu().numpy()
        print(f"  per-track AUC ({nm}): {rank_auc(sc, y[te]):.4f}")


if __name__ == "__main__":
    main()
