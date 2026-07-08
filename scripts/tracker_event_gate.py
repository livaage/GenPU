"""Thin end-to-end TRACKER gate (M4-style, charged-pion first pass).

Mirror of calo_event_gate.py for the tracker head: truth pions -> generate each
one's tracker-hit sequence -> superpose per event -> hit-level two-sample test
(real vs generated event). Tests whether the tracker's known free-running issues
(layer-occupancy drift, residual narrowing) show up at event level, and WHERE a
classifier discriminates.

Scope/caveats (first pass, same spirit as the calo gate):
  - CHARGED PIONS only (the per-species tracker model).
  - hit COUNT per particle taken from truth (generate is conditioned on n_hits;
    no count/EOS head yet) -> isolates hit PLACEMENT quality from count.
  - not held-out (model far too small to memorise); n_hits capped at max_hits.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from genpu.models.tracker_model import TrackerModel

PF_LOGPT, PF_ETA, PF_PHI, PF_PDG, PF_CHARGE, PF_MASS = range(6)
AUX_PRIMARY, AUX_PARENT, AUX_VX, AUX_VY, AUX_VZ, AUX_ENERGY = range(6)
# tracker_hits_flat: (M,5) = [layer_class, r, phi, z, time]  (physical r,phi,z,time)
TH_LAYER, TH_R, TH_PHI, TH_Z, TH_TIME = range(5)


def event_features(layer, r, z, n_src):
    """Position-invariant per-event tracker observables (real & gen, same fn)."""
    return np.array([
        len(r),                              # total hits
        layer.mean(), layer.std(),           # layer occupancy centre + spread (drift shows here)
        r.mean(), r.std(),                   # radial profile
        z.std(),                             # longitudinal spread
        float((layer < 12).mean()),          # fraction in inner layers
        len(r) / max(n_src, 1),              # hits per source pion
    ], dtype=np.float32)


def rank_auc(scores, labels):
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64); ranks[order] = np.arange(1, len(scores) + 1)
    pos = labels == 1; n_pos, n_neg = pos.sum(), (~pos).sum()
    if n_pos == 0 or n_neg == 0:
        return 0.5
    return (ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--preproc_dir", default="/scratch/gpfs/IOJALVO/lv7805/genpu_data/preprocessed")
    ap.add_argument("--slice", default="/scratch/gpfs/IOJALVO/lv7805/genpu_data/tracker_slice/pion.npz",
                    help="only for its cont norm buffers")
    ap.add_argument("--pdg_class", type=int, nargs="+", default=[3, 4])
    ap.add_argument("--max_hits", type=int, default=32)
    ap.add_argument("--layer_temp", type=float, default=1.0)
    ap.add_argument("--cont_temp", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--max_particles", type=int, default=0, help="0=all; smoke subsample")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0); rng = np.random.default_rng(0)

    dsl = np.load(args.slice)
    norm = {"cont_mean": dsl["cont_mean"], "cont_std": dsl["cont_std"]}
    model = TrackerModel(norm).to(dev)
    model.load_state_dict(torch.load(args.ckpt, map_location=dev)["model"]); model.eval()

    d = np.load(Path(args.preproc_dir) / f"shard_{args.shard:04d}_stage2.npz")
    pf, aux, th, off, eid = (d["particle_features"], d["particle_aux"],
                             d["tracker_hits_flat"], d["tracker_offsets"], d["event_ids"])
    ntrk = np.diff(off)
    sel = np.where(np.isin(pf[:, PF_PDG], args.pdg_class) & (ntrk >= 1))[0]
    if args.max_particles and len(sel) > args.max_particles:
        sel = sel[:args.max_particles]

    vr = np.hypot(aux[sel, AUX_VX], aux[sel, AUX_VY])
    logE = np.log(np.clip(aux[sel, AUX_ENERGY], 1e-6, None))
    cont = np.stack([pf[sel, PF_LOGPT], pf[sel, PF_ETA], logE, pf[sel, PF_CHARGE],
                     pf[sel, PF_MASS], vr, aux[sel, AUX_VZ]], axis=1).astype(np.float32)
    pdg = pf[sel, PF_PDG].astype(np.int64)
    n_true = np.clip(ntrk[sel], 1, args.max_hits).astype(np.int64)

    # ---- generate tracker hits per pion (batched AR), conditioned on truth n_hits ----
    contS = torch.as_tensor((cont - norm["cont_mean"]) / norm["cont_std"], device=dev)
    pdgT = torch.as_tensor(pdg, device=dev)
    gen_layer, gen_r, gen_z, gen_src = [], [], [], []
    for s in range(0, len(sel), args.batch):
        e = min(s + args.batch, len(sel))
        with torch.no_grad():
            ce = model.cond_embed(contS[s:e], pdgT[s:e])
            nh = torch.as_tensor(n_true[s:e], device=dev)
            hits, layers = model.tracker.generate(ce, nh, layer_temp=args.layer_temp,
                                                  cont_temp=args.cont_temp)  # (b,Nmax,4), (b,Nmax)
        hits = hits.cpu().numpy(); layers = layers.cpu().numpy()
        for j in range(e - s):
            k = int(n_true[s + j])
            gen_layer.append(layers[j, :k]); gen_r.append(hits[j, :k, TH_R - 1])
            gen_z.append(hits[j, :k, TH_Z - 1]); gen_src.append(np.full(k, s + j))
    gen_layer = np.concatenate(gen_layer); gen_r = np.concatenate(gen_r)
    gen_z = np.concatenate(gen_z); gen_src = np.concatenate(gen_src)

    # ---- group by event, build real & gen feature rows ----
    ev_of = eid[sel]; uev = np.unique(ev_of)
    Xr, Xg = [], []
    for ev in uev:
        pm = np.where(ev_of == ev)[0]; n_src = len(pm)
        rl, rr, rz = [], [], []
        for i in pm:
            a, b = off[sel[i]], off[sel[i] + 1]
            rl.append(th[a:b, TH_LAYER]); rr.append(th[a:b, TH_R]); rz.append(th[a:b, TH_Z])
        Xr.append(event_features(np.concatenate(rl), np.concatenate(rr), np.concatenate(rz), n_src))
        gmask = np.isin(gen_src, pm)
        Xg.append(event_features(gen_layer[gmask], gen_r[gmask], gen_z[gmask], n_src))
    Xr, Xg = np.array(Xr), np.array(Xg)

    # ---- two-sample classifier (torch MLP, rank-AUC on held-out half) ----
    X = np.concatenate([Xr, Xg]); y = np.concatenate([np.zeros(len(Xr)), np.ones(len(Xg))])
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-6)
    perm = rng.permutation(len(X)); ntr = len(X) // 2
    tr, te = perm[:ntr], perm[ntr:]
    Xt = torch.as_tensor(Xs, dtype=torch.float32, device=dev)
    yt = torch.as_tensor(y, dtype=torch.float32, device=dev)
    clf = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 64), torch.nn.SiLU(),
                              torch.nn.Linear(64, 64), torch.nn.SiLU(), torch.nn.Linear(64, 1)).to(dev)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3); tri = torch.as_tensor(tr, device=dev)
    for _ in range(800):
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(clf(Xt[tri]).squeeze(-1), yt[tri])
        loss.backward(); opt.step()
    with torch.no_grad():
        s_te = clf(Xt[torch.as_tensor(te, device=dev)]).squeeze(-1).cpu().numpy()
    auc = rank_auc(s_te, y[te])

    names = ["n_hits", "layer_mean", "layer_std", "r_mean", "r_std", "z_std", "frac_inner", "hits_per_pion"]
    print("=" * 60); print(f"TRACKER EVENT GATE (pion) — {Path(args.ckpt).name}"); print("=" * 60)
    print(f"events: {len(uev)}  (real {len(Xr)} vs gen {len(Xg)})   test AUC = {auc:.4f}   (0.5 = indistinguishable)")
    print(f"{'feature':16s} {'real_mean':>12s} {'gen_mean':>12s} {'|Δ|/σ':>8s}")
    for j, nm in enumerate(names):
        dstd = abs(Xr[:, j].mean() - Xg[:, j].mean()) / (X[:, j].std() + 1e-9)
        print(f"  {nm:14s} {Xr[:,j].mean():12.3f} {Xg[:,j].mean():12.3f} {dstd:8.2f}")
    out = Path(args.ckpt).parent / "eval" / "event_gate.json"; out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"auc": float(auc), "n_events": int(len(uev)), "features": names,
                               "real_mean": Xr.mean(0).tolist(), "gen_mean": Xg.mean(0).tolist()}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
