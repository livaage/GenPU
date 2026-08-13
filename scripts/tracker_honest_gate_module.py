"""HONEST tracker event gate for the v3 SURFACE-LOCAL tracker (tracker_module_ar).

Identical protocol to tracker_honest_gate.py (same count head, same real-event features, same
MLP two-sample classifier) — the ONLY change is the generator: v3 emits global (x,y,z)+module,
so we adapt to the gate's (layer, r, z) features via  layer=module->layer_class,  r=hypot(x,y).
This keeps the comparison against the v1 0.77 baseline apples-to-apples (only the tracker differs).
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from genpu.models.tracker_module_model import TrackerModuleModel
from genpu.models.count_head import CountHead
from genpu.module_geometry import ModuleGeometry

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
    ap.add_argument("--ckpt", required=True, help="v3 surface tracker checkpoint")
    ap.add_argument("--count_ckpt", required=True, help="use the *_selfnorm count head (normalizes cont itself)")
    ap.add_argument("--module_geometry", default="/scratch/gpfs/IOJALVO/lv7805/genpu_data/module_geometry.npz")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--preproc_dir", default="/scratch/gpfs/IOJALVO/lv7805/genpu_data/preprocessed")
    ap.add_argument("--slice", default="/scratch/gpfs/IOJALVO/lv7805/genpu_data/tracker_slice/surface_pion.npz")
    ap.add_argument("--pdg_class", type=int, nargs="+", default=[3, 4])
    ap.add_argument("--max_hits", type=int, default=32)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--truth_count", action="store_true")
    ap.add_argument("--count_no_d0", action="store_true")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"; torch.manual_seed(0); rng = np.random.default_rng(0)

    dsl = np.load(args.slice); norm = {"cont_mean": dsl["cont_mean"], "cont_std": dsl["cont_std"]}
    model = TrackerModuleModel(norm, module_geometry_path=args.module_geometry, max_hits=args.max_hits).to(dev)
    model.load_state_dict(torch.load(args.ckpt, map_location=dev)["model"]); model.eval()
    ch = CountHead(use_d0=not args.count_no_d0).to(dev)
    ch.load_state_dict(torch.load(args.count_ckpt, map_location=dev)["model"]); ch.eval()
    mg = ModuleGeometry(args.module_geometry)
    layer_of_module = torch.as_tensor(mg.layer_class, dtype=torch.long, device=dev)

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

    contS = torch.as_tensor((cont - norm["cont_mean"]) / norm["cont_std"], device=dev)   # tracker norm
    contP = torch.as_tensor(cont, dtype=torch.float32, device=dev)                       # RAW for count head
    pdgT = torch.as_tensor(pdg, device=dev); d0T = torch.as_tensor(d0, device=dev)
    with torch.no_grad():
        n_gen = (torch.as_tensor(n_true, device=dev) if args.truth_count
                 else ch.sample_raw(contP, pdgT, d0T).clamp(1, args.max_hits))   # self-normalizing count head
    n_gen_np = n_gen.cpu().numpy()

    gl, gr, gz, gs = [], [], [], []
    for s in range(0, len(sel), args.batch):
        e = min(s + args.batch, len(sel))
        with torch.no_grad():
            ce = model.cond_embed(contS[s:e], pdgT[s:e])
            phys, gmod = model.tracker.generate(ce, n_gen[s:e])       # phys (B,N,4) [x,y,z,time]
            layers = layer_of_module[gmod]                            # module -> layer_class
        phys = phys.cpu().numpy(); layers = layers.cpu().numpy()
        r = np.hypot(phys[:, :, 0], phys[:, :, 1]); z = phys[:, :, 2]
        for j in range(e - s):
            k = int(n_gen_np[s + j])
            gl.append(layers[j, :k]); gr.append(r[j, :k]); gz.append(z[j, :k]); gs.append(np.full(k, s + j))
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
        auc = rank_auc(clf(Xt[torch.as_tensor(te, device=dev)]).squeeze(-1).cpu().numpy(), y[te])
    names = ["n_hits", "layer_mean", "layer_std", "r_mean", "r_std", "z_std", "frac_inner", "hits_per_pion"]
    mode = "TRUTH-count" if args.truth_count else "HONEST (count head)"
    print("=" * 62); print(f"v3 SURFACE TRACKER {mode}  pdg={args.pdg_class}"); print("=" * 62)
    print(f"count: real n_hits mean={n_true.mean():.2f} median={np.median(n_true):.0f}   "
          f"gen mean={n_gen_np.mean():.2f} median={np.median(n_gen_np):.0f}")
    print(f"events: {len(uev)}   test AUC = {auc:.4f}   (0.5 = indistinguishable)")
    for j, nm in enumerate(names):
        dstd = abs(Xr[:, j].mean() - Xg[:, j].mean()) / (X[:, j].std() + 1e-9)
        print(f"  {nm:14s} real {Xr[:,j].mean():10.3f}  gen {Xg[:,j].mean():10.3f}  |d|/s {dstd:.2f}")


if __name__ == "__main__":
    main()
