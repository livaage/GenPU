"""Cheap inference-only test of the exposure-bias drift: sweep cont_temp (per-step sampling
temperature for r/phi/z/time heads) and measure the rich-15 gate AUC + outer-tail stats.
Reducing per-step noise (temp<1) should damp the compounding drift that inflates r_p90/absz_p90.
Bounds the free (no-retrain) recovery before committing to scheduled-sampling training."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from genpu.models.tracker_model import TrackerModel
from genpu.models.count_head import CountHead

PF_LOGPT, PF_ETA, PF_PHI, PF_PDG, PF_CHARGE, PF_MASS = range(6)
AUX_PRIMARY, AUX_PARENT, AUX_VX, AUX_VY, AUX_VZ, AUX_ENERGY = range(6)
TH_LAYER, TH_R, TH_PHI, TH_Z, TH_TIME = range(5)
RICH = ["n_hits", "hits_per", "layer_mean", "layer_std", "layer_p95", "r_mean", "r_std",
        "r_p90", "absz_mean", "z_std", "absz_p90", "frac_inner", "frac_r500", "frac_len7", "len_p95"]


def rich_features(layer, r, z, lengths, n_src):
    if len(r) == 0:
        return np.zeros(len(RICH), np.float32)
    az = np.abs(z); L = np.asarray(lengths)
    return np.array([len(r), len(r)/max(n_src,1), layer.mean(), layer.std(), np.percentile(layer,95),
                     r.mean(), r.std(), np.percentile(r,90), az.mean(), z.std(), np.percentile(az,90),
                     float((layer<12).mean()), float((r>500).mean()),
                     float((L>=7).mean()) if len(L) else 0., np.percentile(L,95) if len(L) else 0.], np.float32)


def rank_auc(s, y):
    o = np.argsort(s); ra = np.empty_like(o, float); ra[o] = np.arange(1, len(s)+1)
    p = y == 1; npo, nne = p.sum(), (~p).sum()
    return 0.5 if npo == 0 or nne == 0 else (ra[p].sum() - npo*(npo+1)/2)/(npo*nne)


def mlp_auc(X, y, tr, te, dev):
    Xs = (X - X[tr].mean(0)) / (X[tr].std(0) + 1e-6)
    Xt = torch.as_tensor(Xs.astype(np.float32), device=dev); yt = torch.as_tensor(y, dtype=torch.float32, device=dev)
    clf = torch.nn.Sequential(torch.nn.Linear(X.shape[1],64), torch.nn.SiLU(),
                              torch.nn.Linear(64,64), torch.nn.SiLU(), torch.nn.Linear(64,1)).to(dev)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3); tri = torch.as_tensor(tr, device=dev)
    for _ in range(800):
        opt.zero_grad()
        torch.nn.functional.binary_cross_entropy_with_logits(clf(Xt[tri]).squeeze(-1), yt[tri]).backward(); opt.step()
    with torch.no_grad():
        return rank_auc(clf(Xt[torch.as_tensor(te, device=dev)]).squeeze(-1).cpu().numpy(), y[te])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--count_ckpt", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--preproc_dir", default="/scratch/gpfs/IOJALVO/lv7805/genpu_data/preprocessed")
    ap.add_argument("--slice", default="/scratch/gpfs/IOJALVO/lv7805/genpu_data/tracker_slice/multispecies.npz")
    ap.add_argument("--pdg_class", type=int, nargs="+", default=list(range(15)))
    ap.add_argument("--temps", type=float, nargs="+", default=[1.0, 0.9, 0.8, 0.7, 0.6])
    ap.add_argument("--max_hits", type=int, default=32); ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--use_vertex", action="store_true"); ap.add_argument("--count_no_d0", action="store_true")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"; torch.manual_seed(0); rng = np.random.default_rng(0)

    dsl = np.load(args.slice); norm = {"cont_mean": dsl["cont_mean"], "cont_std": dsl["cont_std"]}
    model = TrackerModel(norm, use_vertex=args.use_vertex).to(dev)
    model.load_state_dict(torch.load(args.ckpt, map_location=dev)["model"]); model.eval()
    ch = CountHead(use_d0=not args.count_no_d0).to(dev)
    ch.load_state_dict(torch.load(args.count_ckpt, map_location=dev)["model"]); ch.eval()

    d = np.load(Path(args.preproc_dir) / f"shard_{args.shard:04d}_stage2.npz")
    pf, aux, th, off, eid = (d["particle_features"], d["particle_aux"], d["tracker_hits_flat"],
                             d["tracker_offsets"], d["event_ids"])
    ntrk = np.diff(off)
    sel = np.where(np.isin(pf[:, PF_PDG], args.pdg_class) & (ntrk >= 1))[0]
    vr = np.hypot(aux[sel, AUX_VX], aux[sel, AUX_VY]); logE = np.log(np.clip(aux[sel, AUX_ENERGY], 1e-6, None))
    cont = np.stack([pf[sel, PF_LOGPT], pf[sel, PF_ETA], logE, pf[sel, PF_CHARGE],
                     pf[sel, PF_MASS], vr, aux[sel, AUX_VZ]], axis=1).astype(np.float32)
    phi = pf[sel, PF_PHI]; d0 = (aux[sel, AUX_VX]*np.sin(phi) - aux[sel, AUX_VY]*np.cos(phi)).astype(np.float32)
    pdg = pf[sel, PF_PDG].astype(np.int64)
    contS = torch.as_tensor((cont - norm["cont_mean"]) / norm["cont_std"], device=dev)
    pdgT = torch.as_tensor(pdg, device=dev); d0T = torch.as_tensor(d0, device=dev)
    vtxT = torch.as_tensor(cont[:, [5, 6]], dtype=torch.float32, device=dev)

    ev_of = eid[sel]; uev = np.unique(ev_of); real_idx = {ev: np.where(ev_of == ev)[0] for ev in uev}
    Xr = []
    for ev in uev:
        pm = real_idx[ev]; rl, rr, rz, rlen = [], [], [], []
        for i in pm:
            a, b = off[sel[i]], off[sel[i]+1]
            rl.append(th[a:b, TH_LAYER]); rr.append(th[a:b, TH_R]); rz.append(th[a:b, TH_Z]); rlen.append(b-a)
        Xr.append(rich_features(np.concatenate(rl), np.concatenate(rr), np.concatenate(rz), rlen, len(pm)))
    Xr = np.array(Xr)
    perm = rng.permutation(2*len(uev)); tr, te = perm[:len(uev)], perm[len(uev):]

    # count head fixed across temps (temp affects only the AR position heads)
    with torch.no_grad():
        n_gen = ch.sample(contS, pdgT, d0T).clamp(1, args.max_hits)
    n_gen_np = n_gen.cpu().numpy()

    print("=" * 78)
    print(f"CONT_TEMP SWEEP  pdg=all  events={len(uev)}   real: r_p90={np.percentile(np.concatenate([th[off[sel[i]]:off[sel[i]+1],TH_R] for i in range(len(sel))]),90):.1f}")
    print(f"  real r_mean={Xr[:,RICH.index('r_mean')].mean():.1f}  r_p90={Xr[:,RICH.index('r_p90')].mean():.1f}  absz_p90={Xr[:,RICH.index('absz_p90')].mean():.1f}")
    print(f"  {'temp':>5s} {'richAUC':>8s} {'gen_rmean':>9s} {'gen_r_p90':>9s} {'gen_absz_p90':>12s} {'gen_nhit':>8s}")
    for temp in args.temps:
        gl, gr, gz, gs = [], [], [], []
        for s in range(0, len(sel), args.batch):
            e = min(s + args.batch, len(sel))
            with torch.no_grad():
                ce = model.cond_embed(contS[s:e], pdgT[s:e])
                vtx = vtxT[s:e] if args.use_vertex else None
                hits, layers = model.tracker.generate(ce, n_gen[s:e], vertex_pos=vtx, cont_temp=temp)
            hits = hits.cpu().numpy(); layers = layers.cpu().numpy()
            for j in range(e - s):
                k = int(n_gen_np[s+j])
                gl.append(layers[j,:k]); gr.append(hits[j,:k,TH_R-1]); gz.append(hits[j,:k,TH_Z-1]); gs.append(np.full(k, s+j))
        gl = np.concatenate(gl); gr = np.concatenate(gr); gz = np.concatenate(gz); gs = np.concatenate(gs)
        Xg = []
        for ev in uev:
            pm = real_idx[ev]; gmask = np.isin(gs, pm)
            glen = np.bincount(gs[gmask], minlength=len(sel))[pm]
            Xg.append(rich_features(gl[gmask], gr[gmask], gz[gmask], glen[glen>0], len(pm)))
        Xg = np.array(Xg)
        X = np.concatenate([Xr, Xg]); y = np.concatenate([np.zeros(len(Xr)), np.ones(len(Xg))])
        auc = mlp_auc(X, y, tr, te, dev)
        print(f"  {temp:>5.2f} {auc:>8.4f} {Xg[:,RICH.index('r_mean')].mean():>9.1f} "
              f"{Xg[:,RICH.index('r_p90')].mean():>9.1f} {Xg[:,RICH.index('absz_p90')].mean():>12.1f} "
              f"{Xg[:,RICH.index('n_hits')].mean():>8.1f}", flush=True)


if __name__ == "__main__":
    main()
