"""Publication metric suite for the calo flow model, on a HELD-OUT shard:
(1) two-sample event gate, (2) per-observable Wasserstein (shower shape), (3) CONDITIONAL ENERGY
RESPONSE — <E_reco/E_true> linearity + resolution vs incident energy, (4) generation speed (times the
ODE). Model + norm are read self-contained from the checkpoint. Works for photon (pdg 2) or pion [3,4].
"""
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
from genpu.flow.calo_flow import CaloFlow

PF_LOGPT, PF_ETA, PF_PHI, PF_PDG, PF_CHARGE, PF_MASS = range(6)
AUX_PRIMARY, AUX_PARENT, AUX_VX, AUX_VY, AUX_VZ, AUX_ENERGY = range(6)
CH_ETA, CH_PHI, CH_LOGE, CH_FRAC, CH_DET = range(5)


def event_features(eta, phi, E, n_src):
    logE = np.log(E + 1e-12); tot = E.sum()
    p90 = np.percentile(logE, 90) if len(logE) else -30.0
    return np.array([len(E), np.log(tot + 1e-12), logE.mean(), logE.std(), logE.max(), p90,
                     float((logE < np.log(5e-5) + 0.5).mean()), len(E) / max(n_src, 1)], dtype=np.float32)


def rank_auc(s, y):
    o = np.argsort(s); ra = np.empty_like(o, float); ra[o] = np.arange(1, len(s) + 1)
    p = y == 1; npo, nne = p.sum(), (~p).sum()
    return 0.5 if npo == 0 or nne == 0 else (ra[p].sum() - npo * (npo + 1) / 2) / (npo * nne)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--pdg_class", type=int, nargs="+", default=[2])
    ap.add_argument("--shard", type=int, default=5, help="HELD-OUT shard (train used 0-2)")
    ap.add_argument("--preproc_dir", default="/scratch/gpfs/IOJALVO/lv7805/genpu_data/preprocessed")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--batch", type=int, default=200000)
    ap.add_argument("--tag", default="photon")
    ap.add_argument("--outdir", default="/home/lv7805/genpu/plots/calo/metrics")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"; torch.manual_seed(0); rng = np.random.default_rng(0)

    ckpt = torch.load(args.ckpt, map_location=dev, weights_only=False)
    sd = ckpt["model"]
    src = ckpt.get("norm", sd)   # some (older) checkpoints store norm separately, and rename cont->cond
    def _get(base):
        for k in (f"cont_{base}", f"cond_{base}"):
            if k in src:
                v = src[k]; return v.cpu().numpy() if hasattr(v, "cpu") else np.asarray(v)
        raise KeyError(base)
    norm = {"cont_mean": _get("mean"), "cont_std": _get("std")}
    for k in ["glob_mean", "glob_std", "pts_mean", "pts_std"]:
        v = src[k]; norm[k] = v.cpu().numpy() if hasattr(v, "cpu") else np.asarray(v)
    model = CaloFlow(norm).to(dev)
    miss = model.load_state_dict(sd, strict=False)
    unexp = [k for k in miss.unexpected_keys if not k.startswith(("cond_mean", "cond_std"))]
    missk = [k for k in miss.missing_keys if not k.startswith(("cont_mean", "cont_std"))]
    if missk or unexp:
        print(f"WARN load: missing={missk[:6]} unexpected={unexp[:6]}")
    model.eval()

    d = np.load(Path(args.preproc_dir) / f"shard_{args.shard:04d}_stage2.npz")
    pf, aux, ch, off, eid = (d["particle_features"], d["particle_aux"], d["calo_hits_flat"],
                             d["calo_offsets"], d["event_ids"])
    ncal = np.diff(off)
    sel = np.where(np.isin(pf[:, PF_PDG], args.pdg_class) & (ncal >= 1))[0]
    vr = np.hypot(aux[sel, AUX_VX], aux[sel, AUX_VY]); logE = np.log(np.clip(aux[sel, AUX_ENERGY], 1e-6, None))
    cont = np.stack([pf[sel, PF_LOGPT], pf[sel, PF_ETA], logE, pf[sel, PF_CHARGE],
                     pf[sel, PF_MASS], vr, aux[sel, AUX_VZ]], 1).astype(np.float32)
    pdg = pf[sel, PF_PDG].astype(np.int64); p_eta, p_phi = pf[sel, PF_ETA], pf[sel, PF_PHI]
    E_true = aux[sel, AUX_ENERGY].astype(np.float64)

    contS = torch.as_tensor((cont - norm["cont_mean"]) / norm["cont_std"], device=dev)
    pdgT = torch.as_tensor(pdg, device=dev)
    # generate (batched) + time it
    gen_eta, gen_phi, gen_E, gen_src = [], [], [], []
    torch.cuda.synchronize() if dev == "cuda" else None
    t0 = time.time()
    for s in range(0, len(sel), args.batch):
        e = min(s + args.batch, len(sel))
        with torch.no_grad():
            ce = model.cond_embed(contS[s:e], pdgT[s:e])
            g_std = model.glob.sample(ce); g = model.unstd_glob(g_std).cpu().numpy()
            g_n = np.clip(np.round(np.exp(g[:, 1])).astype(int), 1, 128)
            repn = np.repeat(np.arange(e - s), g_n); rep = torch.as_tensor(repn, device=dev)
            pos_std = model.points.sample(ce[rep], g_std[rep][:, :2], steps=args.steps)
            le = model.energy.sample(ce[rep], model.log_floor).cpu().numpy()
            pos = model.unstd_pos(pos_std).cpu().numpy()
        # per-shower core (glob dims 2,3) when present: points are DELTAS from the core, so add it.
        core = g[:, 2:4] if g.shape[1] >= 4 else np.zeros((len(g), 2), np.float32)
        gen_E.append(np.exp(le).astype(np.float32))
        gen_eta.append(p_eta[s:e][repn] + core[repn, 0] + pos[:, 0])
        gen_phi.append(p_phi[s:e][repn] + core[repn, 1] + pos[:, 1])
        gen_src.append(repn + s)
    torch.cuda.synchronize() if dev == "cuda" else None
    gen_time = time.time() - t0
    gen_E = np.concatenate(gen_E); gen_eta = np.concatenate(gen_eta); gen_phi = np.concatenate(gen_phi); gen_src = np.concatenate(gen_src)

    # ---- real pooled + per-shower ----
    RE, RP, RG = [], [], []; r_n = np.zeros(len(sel)); r_Ereco = np.zeros(len(sel)); r_width = np.zeros(len(sel))
    for k, i in enumerate(sel):
        a, b = off[i], off[i + 1]
        de = ch[a:b, CH_ETA] - p_eta[k]; dp = ch[a:b, CH_PHI] - p_phi[k]; ee = np.exp(ch[a:b, CH_LOGE])
        RE.append(ch[a:b, CH_LOGE]); RP.append(de); RG.append(dp)
        r_n[k] = b - a; r_Ereco[k] = ee.sum(); r_width[k] = np.sqrt(np.mean(de**2 + dp**2))
    RE, RP, RG = map(np.concatenate, (RE, RP, RG))
    # gen per-shower
    g_n_arr = np.bincount(gen_src, minlength=len(sel)).astype(float)
    g_Ereco = np.bincount(gen_src, weights=gen_E, minlength=len(sel))
    g_de = gen_eta - p_eta[gen_src]; g_dp = gen_phi - p_phi[gen_src]
    g_width = np.sqrt(np.bincount(gen_src, weights=g_de**2 + g_dp**2, minlength=len(sel)) / np.maximum(g_n_arr, 1))

    # (1) per-observable Wasserstein
    obs = {"cells_per_shower": (r_n, g_n_arr), "cell_logE": (RE, np.log(gen_E + 1e-12)),
           "d_eta": (RP, g_de), "d_phi": (RG, g_dp), "shower_width": (r_width, g_width),
           "logEreco": (np.log(r_Ereco + 1e-12), np.log(g_Ereco + 1e-12))}
    wass = {k: float(wasserstein_distance(a, b)) for k, (a, b) in obs.items()}
    wnorm = {k: wass[k] / (obs[k][0].std() + 1e-9) for k in obs}

    # (3) conditional energy response: <E_reco/E_true> + resolution vs E_true
    resp_r = r_Ereco / E_true; resp_g = g_Ereco / E_true
    bins = np.quantile(np.log10(E_true), np.linspace(0, 1, 7))
    bi = np.clip(np.digitize(np.log10(E_true), bins[1:-1]), 0, 5)
    response = []
    for bb in range(6):
        m = bi == bb
        if m.sum() < 50:
            continue
        response.append({"logE_true_med": round(float(np.median(np.log10(E_true[m]))), 2),
                         "n": int(m.sum()),
                         "resp_real": round(float(np.mean(resp_r[m])), 3), "resp_gen": round(float(np.mean(resp_g[m])), 3),
                         "reso_real": round(float(np.std(resp_r[m]) / (np.mean(resp_r[m]) + 1e-9)), 3),
                         "reso_gen": round(float(np.std(resp_g[m]) / (np.mean(resp_g[m]) + 1e-9)), 3)})

    # (2) event gate
    ev_of = eid[sel]; uev = np.unique(ev_of); Xr, Xg = [], []
    for ev in uev:
        pm = np.where(ev_of == ev)[0]; n_src = len(pm)
        re, rp, rE = [], [], []
        for i in pm:
            a, b = off[sel[i]], off[sel[i] + 1]; re.append(ch[a:b, CH_ETA]); rp.append(ch[a:b, CH_PHI]); rE.append(np.exp(ch[a:b, CH_LOGE]))
        Xr.append(event_features(np.concatenate(re), np.concatenate(rp), np.concatenate(rE), n_src))
        gm = np.isin(gen_src, pm); Xg.append(event_features(gen_eta[gm], gen_phi[gm], gen_E[gm], n_src))
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

    speed = {"particles": int(len(sel)), "gen_seconds": round(gen_time, 2), "ode_steps": args.steps,
             "particles_per_sec": round(len(sel) / gen_time, 1), "us_per_particle": round(1e6 * gen_time / len(sel), 1)}
    out = {"tag": args.tag, "pdg_class": args.pdg_class, "shard_heldout": args.shard, "n_particles": int(len(sel)),
           "event_gate_auc": round(float(auc), 4), "wasserstein_over_std": {k: round(v, 4) for k, v in wnorm.items()},
           "energy_response": response, "speed": speed}
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"metrics_{args.tag}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2)); print("wrote", outdir / f"metrics_{args.tag}.json")


if __name__ == "__main__":
    main()
