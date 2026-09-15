"""HONEST tracker event gate: the full self-driven pipeline with NO truth n_hits.
For each real particle -> count head samples n_hits (from conditioning + geometric d0) ->
AR generates that many hits -> superpose per event -> two-sample gate vs real. This is the
real 'how well do we match pileup' number; the old gate used truth counts as a crutch."""
from __future__ import annotations
import argparse, json
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


def event_features(layer, r, z, n_src):
    return np.array([len(r), layer.mean() if len(r) else 0, layer.std() if len(r) else 0,
                     r.mean() if len(r) else 0, r.std() if len(r) else 0, z.std() if len(r) else 0,
                     float((layer < 12).mean()) if len(r) else 0, len(r) / max(n_src, 1)], dtype=np.float32)


def rank_auc(s, y):
    o = np.argsort(s); ra = np.empty_like(o, float); ra[o] = np.arange(1, len(s) + 1)
    p = y == 1; npo, nne = p.sum(), (~p).sum()
    return 0.5 if npo == 0 or nne == 0 else (ra[p].sum() - npo * (npo + 1) / 2) / (npo * nne)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="AR tracker checkpoint")
    ap.add_argument("--count_ckpt", required=True, help="count head checkpoint")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--preproc_dir", default="/scratch/gpfs/IOJALVO/lv7805/genpu_data/preprocessed")
    ap.add_argument("--slice", default="/scratch/gpfs/IOJALVO/lv7805/genpu_data/tracker_slice/multispecies.npz")
    ap.add_argument("--pdg_class", type=int, nargs="+", default=[3, 4])
    ap.add_argument("--max_hits", type=int, default=32)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--use_vertex", action="store_true")
    ap.add_argument("--use_mom_feat", action="store_true", help="model uses helix-z momentum-estimate feature")
    ap.add_argument("--truth_count", action="store_true", help="ablation: use truth n_hits (old gate)")
    ap.add_argument("--count_no_d0", action="store_true", help="count head trained without d0")
    ap.add_argument("--plots", action="store_true", help="write the gate ROC curve")
    ap.add_argument("--outdir", default="/home/lv7805/genpu/plots/tracker/metrics")
    ap.add_argument("--tag", default="tracker")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"; torch.manual_seed(0); rng = np.random.default_rng(0)

    dsl = np.load(args.slice); norm = {"cont_mean": dsl["cont_mean"], "cont_std": dsl["cont_std"]}
    model = TrackerModel(norm, use_vertex=args.use_vertex, use_mom_feat=args.use_mom_feat).to(dev)
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
    phi = pf[sel, PF_PHI]; d0 = (aux[sel, AUX_VX] * np.sin(phi) - aux[sel, AUX_VY] * np.cos(phi)).astype(np.float32)
    pdg = pf[sel, PF_PDG].astype(np.int64); n_true = np.clip(ntrk[sel], 1, args.max_hits).astype(np.int64)

    contS = torch.as_tensor((cont - norm["cont_mean"]) / norm["cont_std"], device=dev)
    pdgT = torch.as_tensor(pdg, device=dev); d0T = torch.as_tensor(d0, device=dev)
    vtxT = torch.as_tensor(cont[:, [5, 6]], dtype=torch.float32, device=dev)
    helixT = (TrackerModel.helix_params_from_cont(torch.as_tensor(cont, dtype=torch.float32, device=dev))
              if args.use_mom_feat else None)
    with torch.no_grad():
        if args.truth_count:
            n_gen = torch.as_tensor(n_true, device=dev)
        else:
            n_gen = ch.sample(contS, pdgT, d0T).clamp(1, args.max_hits)     # emergent, honest count
    n_gen_np = n_gen.cpu().numpy()

    gl, gr, gz, gs = [], [], [], []
    for s in range(0, len(sel), args.batch):
        e = min(s + args.batch, len(sel))
        with torch.no_grad():
            ce = model.cond_embed(contS[s:e], pdgT[s:e])
            vtx = vtxT[s:e] if args.use_vertex else None
            hlx = helixT[s:e] if args.use_mom_feat else None
            hits, layers = model.tracker.generate(ce, n_gen[s:e], vertex_pos=vtx, helix_params=hlx)
        hits = hits.cpu().numpy(); layers = layers.cpu().numpy()
        for j in range(e - s):
            k = int(n_gen_np[s + j])
            gl.append(layers[j, :k]); gr.append(hits[j, :k, TH_R - 1]); gz.append(hits[j, :k, TH_Z - 1]); gs.append(np.full(k, s + j))
    gl = np.concatenate(gl); gr = np.concatenate(gr); gz = np.concatenate(gz); gs = np.concatenate(gs)

    ev_of = eid[sel]; uev = np.unique(ev_of); Xr, Xg = [], []
    for ev in uev:
        pm = np.where(ev_of == ev)[0]; n_src = len(pm)
        rl, rr, rz = [], [], []
        for i in pm:
            a, b = off[sel[i]], off[sel[i] + 1]
            rl.append(th[a:b, TH_LAYER]); rr.append(th[a:b, TH_R]); rz.append(th[a:b, TH_Z])
        Xr.append(event_features(np.concatenate(rl), np.concatenate(rr), np.concatenate(rz), n_src))
        gmask = np.isin(gs, pm)
        Xg.append(event_features(gl[gmask], gr[gmask], gz[gmask], n_src))
    Xr, Xg = np.array(Xr), np.array(Xg)

    X = np.concatenate([Xr, Xg]); y = np.concatenate([np.zeros(len(Xr)), np.ones(len(Xg))])
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-6)
    perm = rng.permutation(len(X)); ntr = len(X) // 2; tr, te = perm[:ntr], perm[ntr:]
    Xt = torch.as_tensor(Xs, dtype=torch.float32, device=dev); yt = torch.as_tensor(y, dtype=torch.float32, device=dev)
    clf = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 64), torch.nn.SiLU(), torch.nn.Linear(64, 64), torch.nn.SiLU(), torch.nn.Linear(64, 1)).to(dev)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3); tri = torch.as_tensor(tr, device=dev)
    for _ in range(800):
        opt.zero_grad(); l = torch.nn.functional.binary_cross_entropy_with_logits(clf(Xt[tri]).squeeze(-1), yt[tri]); l.backward(); opt.step()
    with torch.no_grad():
        te_scores = clf(Xt[torch.as_tensor(te, device=dev)]).squeeze(-1).cpu().numpy()
    auc = rank_auc(te_scores, y[te])
    if args.plots:
        # ROC of the two-sample event gate. The diagonal is the target: a generator the classifier
        # cannot separate from Geant sits on it. Same held-out scores the AUC is computed from.
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from pathlib import Path as _P
        o = np.argsort(-te_scores); yo = y[te][o]
        tp = np.cumsum(yo); fp = np.cumsum(1.0 - yo)
        tpr = np.concatenate([[0], tp / max(tp[-1], 1)]); fpr = np.concatenate([[0], fp / max(fp[-1], 1)])
        fig, ax = plt.subplots(figsize=(6.0, 5.6))
        ax.plot(fpr, tpr, lw=2.2, color="#1f6feb", label=f"8 event features   AUC {auc:.3f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1.4, label="indistinguishable (AUC 0.5)")
        ax.set_xlabel("false positive rate (real called generated)")
        ax.set_ylabel("true positive rate (generated called generated)")
        ax.set_title(f"tracker two-sample event gate — {args.tag}\ncloser to the diagonal is better")
        ax.legend(fontsize=9, loc="lower right"); ax.grid(alpha=0.3)
        fig.tight_layout()
        _P(args.outdir).mkdir(parents=True, exist_ok=True)
        rp = _P(args.outdir) / f"roc_{args.tag}.png"
        fig.savefig(rp, dpi=130); plt.close(fig)
        print("wrote", rp, flush=True)
    names = ["n_hits", "layer_mean", "layer_std", "r_mean", "r_std", "z_std", "frac_inner", "hits_per_pion"]
    mode = "TRUTH-count (crutch)" if args.truth_count else "HONEST (count head, no truth n_hits)"
    print("=" * 62); print(f"TRACKER {mode}  pdg={args.pdg_class}"); print("=" * 62)
    print(f"count: real n_hits mean={n_true.mean():.2f} median={np.median(n_true):.0f}   "
          f"gen mean={n_gen_np.mean():.2f} median={np.median(n_gen_np):.0f}")
    print(f"events: {len(uev)}   test AUC = {auc:.4f}   (0.5 = indistinguishable)")
    for j, nm in enumerate(names):
        dstd = abs(Xr[:, j].mean() - Xg[:, j].mean()) / (X[:, j].std() + 1e-9)
        print(f"  {nm:14s} real {Xr[:,j].mean():10.3f}  gen {Xg[:,j].mean():10.3f}  |d|/s {dstd:.2f}")


if __name__ == "__main__":
    main()
