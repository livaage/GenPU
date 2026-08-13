"""Count-routed HYBRID generation + gate.
Routing by the count head's sampled n_hits:
  n_hits==1 & neutral (photon/neutron): straight-line flight from vertex along momentum,
             interaction length sampled from empirical per-species displacement (held-out shard).
  n_hits==1 & charged/fragment:         place the hit at the production vertex (r=vr, z=vz).
  n_hits>1:                             the AR (unchanged).
Single-hit hits are snapped to the nearest detector layer (barrel: by r, endcap: by |z|).
Clean A/B: generates the AR for ALL particles (baseline), then for HYBRID overrides only the
single-hit particles with placement. Same train/test split -> the AUC delta is purely the routing.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from genpu.models.tracker_model import TrackerModel
from genpu.models.count_head import CountHead
from genpu.detector_geometry import LAYER_MEANS, LAYER_IS_BARREL, N_LAYERS

PF_LOGPT, PF_ETA, PF_PHI, PF_PDG, PF_CHARGE, PF_MASS = range(6)
AUX_PRIMARY, AUX_PARENT, AUX_VX, AUX_VY, AUX_VZ, AUX_ENERGY = range(6)
TH_LAYER, TH_R, TH_PHI, TH_Z, TH_TIME = range(5)
NEUTRAL = {2, 9, 10}  # photon, neutron, antineutron
R_L = LAYER_MEANS[:, 0].astype(np.float32); Z_L = np.abs(LAYER_MEANS[:, 2]).astype(np.float32)
IS_BAR = np.asarray(LAYER_IS_BARREL, bool)
RICH = ["n_hits", "hits_per", "layer_mean", "layer_std", "layer_p95", "r_mean", "r_std",
        "r_p90", "absz_mean", "z_std", "absz_p90", "frac_inner", "frac_r500", "frac_len7", "len_p95"]


def snap_layer(r, z, chunk=200000):
    """nearest layer: barrel by |r-r_L|, endcap by ||z|-z_L|. r,z: (N,) -> (N,) layer idx."""
    out = np.empty(len(r), np.int64); az = np.abs(z)
    for s in range(0, len(r), chunk):
        e = min(s + chunk, len(r))
        db = np.abs(r[s:e, None] - R_L[None, :])           # barrel distance
        de = np.abs(az[s:e, None] - Z_L[None, :])          # endcap distance
        d = np.where(IS_BAR[None, :], db, de)
        out[s:e] = d.argmin(1)
    return out


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


def flight_reference(preproc_dir, shard):
    """empirical per-species displacement VECTOR (dr=hit_r-vtx_r, dz=hit_z-vtx_z) for single-hit
    photons/neutrons (held-out shard). Paired so (dr,dz) and its eta-correlation survive; applied to
    the vertex and clipped to the detector. Reproduces the real neutral (r,z) marginal because the
    routed particles are the same real population (same eta distribution) as this reference -- unlike
    the old 'fly a 3D magnitude along momentum' model, which decoupled flight-length from eta and had
    no detector bound, so central neutrals flew past the tracker."""
    d = np.load(Path(preproc_dir) / f"shard_{shard:04d}_stage2.npz")
    pf, aux, th, off = d["particle_features"], d["particle_aux"], d["tracker_hits_flat"], d["tracker_offsets"]
    nh = np.diff(off); one = np.where(nh == 1)[0]; a = off[one]
    vr = np.hypot(aux[one, AUX_VX], aux[one, AUX_VY])
    dr = th[a, TH_R] - vr; dz = th[a, TH_Z] - aux[one, AUX_VZ]
    cls = pf[one, PF_PDG].astype(int)
    ref = {"photon": {"dr": dr[cls == 2], "dz": dz[cls == 2]},
           "neutron": {"dr": dr[np.isin(cls, [9, 10])], "dz": dz[np.isin(cls, [9, 10])]}}
    for k in ref:
        print(f"  neutral ref (shard {shard}) {k}: n={len(ref[k]['dr'])} dr med={np.median(ref[k]['dr']):.0f} "
              f"dz med={np.median(ref[k]['dz']):.0f}", flush=True)
    return ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--count_ckpt", required=True)
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--ref_shard", type=int, default=1)
    ap.add_argument("--preproc_dir", default="/scratch/gpfs/IOJALVO/lv7805/genpu_data/preprocessed")
    ap.add_argument("--slice", default="/scratch/gpfs/IOJALVO/lv7805/genpu_data/tracker_slice/multispecies.npz")
    ap.add_argument("--pdg_class", type=int, nargs="+", default=list(range(15)))
    ap.add_argument("--max_hits", type=int, default=32); ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--use_vertex", action="store_true"); ap.add_argument("--count_no_d0", action="store_true")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"; torch.manual_seed(0); rng = np.random.default_rng(0)

    dsl = np.load(args.slice); norm = {"cont_mean": dsl["cont_mean"], "cont_std": dsl["cont_std"]}
    model = TrackerModel(norm, use_vertex=args.use_vertex).to(dev)
    model.load_state_dict(torch.load(args.ckpt, map_location=dev)["model"]); model.eval()
    ch = CountHead(use_d0=not args.count_no_d0).to(dev)
    ch.load_state_dict(torch.load(args.count_ckpt, map_location=dev)["model"]); ch.eval()
    ref = flight_reference(args.preproc_dir, args.ref_shard)

    d = np.load(Path(args.preproc_dir) / f"shard_{args.shard:04d}_stage2.npz")
    pf, aux, th, off, eid = (d["particle_features"], d["particle_aux"], d["tracker_hits_flat"],
                             d["tracker_offsets"], d["event_ids"])
    ntrk = np.diff(off)
    sel = np.where(np.isin(pf[:, PF_PDG], args.pdg_class) & (ntrk >= 1))[0]
    vx, vy, vz = aux[sel, AUX_VX], aux[sel, AUX_VY], aux[sel, AUX_VZ]
    vr = np.hypot(vx, vy); logE = np.log(np.clip(aux[sel, AUX_ENERGY], 1e-6, None))
    eta, ph = pf[sel, PF_ETA], pf[sel, PF_PHI]
    cont = np.stack([pf[sel, PF_LOGPT], eta, logE, pf[sel, PF_CHARGE], pf[sel, PF_MASS], vr, vz], axis=1).astype(np.float32)
    d0 = (vx*np.sin(ph) - vy*np.cos(ph)).astype(np.float32); pdg = pf[sel, PF_PDG].astype(np.int64)
    contS = torch.as_tensor((cont - norm["cont_mean"]) / norm["cont_std"], device=dev)
    pdgT = torch.as_tensor(pdg, device=dev); d0T = torch.as_tensor(d0, device=dev)
    vtxT = torch.as_tensor(cont[:, [5, 6]], dtype=torch.float32, device=dev)

    with torch.no_grad():
        n_gen = ch.sample(contS, pdgT, d0T).clamp(1, args.max_hits)
    n_gen_np = n_gen.cpu().numpy()

    # ---- AR generate for ALL particles (baseline) ----
    gl, gr, gz, gs = [], [], [], []
    for s in range(0, len(sel), args.batch):
        e = min(s + args.batch, len(sel))
        with torch.no_grad():
            ce = model.cond_embed(contS[s:e], pdgT[s:e])
            vtx = vtxT[s:e] if args.use_vertex else None
            hits, layers = model.tracker.generate(ce, n_gen[s:e], vertex_pos=vtx)
        hits = hits.cpu().numpy(); layers = layers.cpu().numpy()
        for j in range(e - s):
            k = int(n_gen_np[s+j])
            gl.append(layers[j,:k]); gr.append(hits[j,:k,TH_R-1]); gz.append(hits[j,:k,TH_Z-1]); gs.append(np.full(k, s+j))
    gl = np.concatenate(gl); gr = np.concatenate(gr); gz = np.concatenate(gz); gs = np.concatenate(gs)

    # ---- placement for single-hit particles ----
    one = np.where(n_gen_np == 1)[0]
    neu = np.isin(pdg[one], list(NEUTRAL))
    p_r = np.empty(len(one), np.float32); p_z = np.empty(len(one), np.float32)
    # charged/fragment: at the production vertex
    ch_i = one[~neu]; p_r[~neu] = vr[ch_i]; p_z[~neu] = vz[ch_i]
    # neutral: straight-line flight vertex + L*dir, L ~ empirical per-species displacement
    nu_i = one[neu]
    if len(nu_i):
        is_photon = pdg[nu_i] == 2
        dr = np.zeros(len(nu_i), np.float32); dz = np.zeros(len(nu_i), np.float32)
        for spec, msk in [("photon", is_photon), ("neutron", ~is_photon)]:
            R = ref[spec]; k = int(msk.sum())
            if k and len(R["dr"]):
                idx = rng.integers(0, len(R["dr"]), size=k)
                dr[msk] = R["dr"][idx]; dz[msk] = R["dz"][idx]
        p_r[neu] = np.clip(vr[nu_i] + dr, 0.0, R_L.max()); p_z[neu] = vz[nu_i] + dz
    p_layer = snap_layer(p_r, p_z).astype(np.float32)

    # ---- assemble baseline vs hybrid flat arrays ----
    keep_multi = n_gen_np[gs] > 1
    hyb_l = np.concatenate([gl[keep_multi], p_layer]); hyb_r = np.concatenate([gr[keep_multi], p_r])
    hyb_z = np.concatenate([gz[keep_multi], p_z]); hyb_s = np.concatenate([gs[keep_multi], one])

    truth1 = ntrk[sel] == 1
    print(f"\nrouting: predicted single-hit={len(one)} ({len(one)/len(sel)*100:.1f}%)  "
          f"TRUTH single-hit={truth1.sum()} ({truth1.mean()*100:.1f}%)  <- count-head over/under-route?"
          f"   [charged/frag {(~neu).sum()}, neutral-flight {neu.sum()}]")
    def shape(n): return [float((n == 1).mean()), float((n == 2).mean()),
                          float(((n >= 3) & (n <= 6)).mean()), float((n >= 7).mean())]
    print(f"  n_hits shape [==1, ==2, 3-6, 7+]:  TRUTH {np.round(shape(np.clip(ntrk[sel],1,99)),3)}   "
          f"count-head {np.round(shape(n_gen_np),3)}")
    a1 = off[sel[np.where(truth1)[0]]]; real1r = th[a1, TH_R]
    ar1r = gr[np.isin(gs, one)]
    def mp(x): return f"med={np.median(x):.0f} mean={x.mean():.0f} p90={np.percentile(x,90):.0f}"
    print(f"  single-hit r_phys:  REAL   {mp(real1r)}")
    print(f"                      AR     {mp(ar1r)}")
    print(f"                      PLACED {mp(p_r)}")

    # ---- gate both ----
    ev_of = eid[sel]; uev = np.unique(ev_of); real_idx = {ev: np.where(ev_of == ev)[0] for ev in uev}
    def build_X(fl, fr, fz, fsrc):
        X = []
        for ev in uev:
            pm = real_idx[ev]; gmask = np.isin(fsrc, pm)
            glen = np.bincount(fsrc[gmask], minlength=len(sel))[pm]
            X.append(rich_features(fl[gmask], fr[gmask], fz[gmask], glen[glen > 0], len(pm)))
        return np.array(X)
    Xr = []
    for ev in uev:
        pm = real_idx[ev]; rl, rr, rz, rlen = [], [], [], []
        for i in pm:
            a, b = off[sel[i]], off[sel[i]+1]
            rl.append(th[a:b, TH_LAYER]); rr.append(th[a:b, TH_R]); rz.append(th[a:b, TH_Z]); rlen.append(b-a)
        Xr.append(rich_features(np.concatenate(rl), np.concatenate(rr), np.concatenate(rz), rlen, len(pm)))
    Xr = np.array(Xr)
    Xg_base = build_X(gl, gr, gz, gs); Xg_hyb = build_X(hyb_l, hyb_r, hyb_z, hyb_s)
    perm = rng.permutation(2*len(uev)); tr, te = perm[:len(uev)], perm[len(uev):]

    print("\n" + "=" * 78)
    print(f"{'variant':10s} {'richAUC':>8s} " + " ".join(f"{n:>10s}" for n in ["r_mean","r_p90","absz_p90","frac_inner","layer_mean"]))
    print(f"{'REAL':10s} {'--':>8s} " + " ".join(f"{Xr[:,RICH.index(n)].mean():>10.2f}" for n in ["r_mean","r_p90","absz_mean","frac_inner","layer_mean"]))
    for tag, Xg in [("BASELINE", Xg_base), ("HYBRID", Xg_hyb)]:
        X = np.concatenate([Xr, Xg]); y = np.concatenate([np.zeros(len(Xr)), np.ones(len(Xg))])
        auc = mlp_auc(X, y, tr, te, dev)
        print(f"{tag:10s} {auc:>8.4f} " + " ".join(f"{Xg[:,RICH.index(n)].mean():>10.2f}" for n in ["r_mean","r_p90","absz_p90","frac_inner","layer_mean"]))


if __name__ == "__main__":
    main()
