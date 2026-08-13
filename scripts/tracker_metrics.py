"""Publication metric suite for the v3 surface tracker: on a HELD-OUT shard, compute
(1) per-observable Wasserstein distances (physics + gate features), (2) the two-sample event gate,
(3) generation speed (particles/s, showers-per-second analogue). Writes metrics.json + marginals.png.
Reuses the honest pipeline (self-normalizing count head, no truth n_hits)."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from scipy.stats import wasserstein_distance
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--count_ckpt", required=True)
    ap.add_argument("--module_geometry", default="/scratch/gpfs/IOJALVO/lv7805/genpu_data/module_geometry.npz")
    ap.add_argument("--slice", required=True)
    ap.add_argument("--shard", type=int, default=5, help="HELD-OUT shard (train used 0-2)")
    ap.add_argument("--preproc_dir", default="/scratch/gpfs/IOJALVO/lv7805/genpu_data/preprocessed")
    ap.add_argument("--pdg_class", type=int, nargs="+", default=[0, 1, 3, 4, 7, 8, 11, 12])
    ap.add_argument("--max_hits", type=int, default=32)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--outdir", default="/home/lv7805/genpu/plots/tracker/metrics")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"; torch.manual_seed(0); rng = np.random.default_rng(0)

    dsl = np.load(args.slice); norm = {"cont_mean": dsl["cont_mean"], "cont_std": dsl["cont_std"]}
    model = TrackerModuleModel(norm, module_geometry_path=args.module_geometry, max_hits=args.max_hits).to(dev)
    model.load_state_dict(torch.load(args.ckpt, map_location=dev)["model"]); model.eval()
    ch = CountHead(use_d0=True).to(dev); ch.load_state_dict(torch.load(args.count_ckpt, map_location=dev)["model"]); ch.eval()
    mg = ModuleGeometry(args.module_geometry); lom = torch.as_tensor(mg.layer_class, dtype=torch.long, device=dev)

    d = np.load(Path(args.preproc_dir) / f"shard_{args.shard:04d}_stage2.npz")
    pf, aux, th, off, eid = (d["particle_features"], d["particle_aux"], d["tracker_hits_flat"],
                             d["tracker_offsets"], d["event_ids"])
    ntrk = np.diff(off)
    sel = np.where(np.isin(pf[:, PF_PDG], args.pdg_class) & (ntrk >= 1))[0]
    vr = np.hypot(aux[sel, AUX_VX], aux[sel, AUX_VY]); logE = np.log(np.clip(aux[sel, AUX_ENERGY], 1e-6, None))
    cont = np.stack([pf[sel, PF_LOGPT], pf[sel, PF_ETA], logE, pf[sel, PF_CHARGE],
                     pf[sel, PF_MASS], vr, aux[sel, AUX_VZ]], 1).astype(np.float32)
    phi = pf[sel, PF_PHI]; d0 = (aux[sel, AUX_VX] * np.sin(phi) - aux[sel, AUX_VY] * np.cos(phi)).astype(np.float32)
    pdg = pf[sel, PF_PDG].astype(np.int64); n_true = np.clip(ntrk[sel], 1, args.max_hits).astype(np.int64)
    contS = torch.as_tensor((cont - norm["cont_mean"]) / norm["cont_std"], device=dev)
    contP = torch.as_tensor(cont, dtype=torch.float32, device=dev)
    pdgT = torch.as_tensor(pdg, device=dev); d0T = torch.as_tensor(d0, device=dev)
    with torch.no_grad():
        n_gen = ch.sample_raw(contP, pdgT, d0T).clamp(1, args.max_hits)
    n_gen_np = n_gen.cpu().numpy()

    # generate + time it
    gl, gr, gz, gs = [], [], [], []
    torch.cuda.synchronize() if dev == "cuda" else None
    t0 = time.time()
    for s in range(0, len(sel), args.batch):
        e = min(s + args.batch, len(sel))
        with torch.no_grad():
            ce = model.cond_embed(contS[s:e], pdgT[s:e])
            phys, gmod = model.tracker.generate(ce, n_gen[s:e]); layers = lom[gmod]
        phys = phys.cpu().numpy(); layers = layers.cpu().numpy()
        r = np.hypot(phys[:, :, 0], phys[:, :, 1]); z = phys[:, :, 2]
        for j in range(e - s):
            k = int(n_gen_np[s + j]); gl.append(layers[j, :k]); gr.append(r[j, :k]); gz.append(z[j, :k]); gs.append(np.full(k, s + j))
    torch.cuda.synchronize() if dev == "cuda" else None
    gen_time = time.time() - t0
    gl = np.concatenate(gl); gr = np.concatenate(gr); gz = np.concatenate(gz); gs = np.concatenate(gs)

    # real pooled hits
    RL, RR, RZ = [], [], []
    for i in sel:
        a, b = off[i], off[i + 1]; RL.append(th[a:b, TH_LAYER]); RR.append(th[a:b, TH_R]); RZ.append(th[a:b, TH_Z])
    RL, RR, RZ = map(np.concatenate, (RL, RR, RZ))

    # (1) per-observable Wasserstein (physics + hit-level)
    obs = {"hit_r": (RR, gr), "hit_z": (RZ, gz), "hit_layer": (RL.astype(float), gl.astype(float)),
           "hits_per_particle": (n_true.astype(float), n_gen_np.astype(float))}
    wass = {k: float(wasserstein_distance(a, b)) for k, (a, b) in obs.items()}
    # normalized (scale-free) distance = W / real std, for cross-observable comparison
    wnorm = {k: wass[k] / (obs[k][0].std() + 1e-9) for k in obs}

    # (2) event gate (held-out shard)
    ev_of = eid[sel]; uev = np.unique(ev_of); Xr, Xg = [], []
    for ev in uev:
        pm = np.where(ev_of == ev)[0]; n_src = len(pm)
        rl, rr, rz = [], [], []
        for i in pm:
            a, b = off[sel[i]], off[sel[i] + 1]; rl.append(th[a:b, TH_LAYER]); rr.append(th[a:b, TH_R]); rz.append(th[a:b, TH_Z])
        Xr.append(event_features(np.concatenate(rl), np.concatenate(rr), np.concatenate(rz), n_src))
        gm = np.isin(gs, pm); Xg.append(event_features(gl[gm], gr[gm], gz[gm], n_src))
    Xr, Xg = np.array(Xr), np.array(Xg)
    X = np.concatenate([Xr, Xg]); y = np.concatenate([np.zeros(len(Xr)), np.ones(len(Xg))])
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-6); perm = rng.permutation(len(X)); ntr = len(X) // 2
    Xt = torch.as_tensor(Xs, dtype=torch.float32, device=dev); yt = torch.as_tensor(y, dtype=torch.float32, device=dev)
    clf = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 64), torch.nn.SiLU(), torch.nn.Linear(64, 64), torch.nn.SiLU(), torch.nn.Linear(64, 1)).to(dev)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3); tri = torch.as_tensor(perm[:ntr], device=dev)
    for _ in range(800):
        opt.zero_grad(); l = torch.nn.functional.binary_cross_entropy_with_logits(clf(Xt[tri]).squeeze(-1), yt[tri]); l.backward(); opt.step()
    with torch.no_grad():
        auc = rank_auc(clf(Xt[torch.as_tensor(perm[ntr:], device=dev)]).squeeze(-1).cpu().numpy(), y[perm[ntr:]])

    # (3) speed
    n_part = len(sel); n_hits_gen = len(gr)
    speed = {"particles": int(n_part), "gen_seconds": round(gen_time, 2),
             "particles_per_sec": round(n_part / gen_time, 1), "hits_per_sec": round(n_hits_gen / gen_time, 1),
             "us_per_particle": round(1e6 * gen_time / n_part, 1)}

    out = {"shard_heldout": args.shard, "pdg_class": args.pdg_class, "n_particles": int(n_part),
           "event_gate_auc": round(float(auc), 4), "wasserstein": {k: round(v, 4) for k, v in wass.items()},
           "wasserstein_over_std": {k: round(v, 4) for k, v in wnorm.items()}, "speed": speed}
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "metrics.json").write_text(json.dumps(out, indent=2))

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    for a, (nm, rv, gv, bins) in zip(ax, [
        ("hit_r [mm]", RR, gr, np.linspace(0, 1100, 60)),
        ("hit_z [mm]", RZ, gz, np.linspace(-3000, 3000, 60)),
        ("hit_layer", RL.astype(float), gl.astype(float), np.arange(0, 49) - .5)]):
        a.hist(rv, bins=bins, density=True, histtype="step", lw=2, label="real")
        a.hist(gv, bins=bins, density=True, histtype="step", lw=2, label="gen"); a.set_title(nm); a.legend()
    fig.suptitle(f"tracker held-out (shard {args.shard}) — gate AUC {auc:.3f}", fontsize=13)
    plt.tight_layout(); fig.savefig(outdir / "marginals.png", dpi=110)
    print(json.dumps(out, indent=2)); print("wrote", outdir / "metrics.json")


if __name__ == "__main__":
    main()
