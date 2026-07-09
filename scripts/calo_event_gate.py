"""Thin end-to-end calo gate (M4-style, photon-only first pass).

Truth particles -> generate each photon's calo shower -> superpose per event ->
hit-level two-sample test (real event vs generated event). This is the first
gate: does the calo head's quality survive at event level, and if not, WHERE
does a classifier discriminate?

Scope/caveats (first pass):
  - PHOTONS only (uses the per-species photon calo model; multi-species needs the
    unified model, i.e. the M3 unify step).
  - Existence/count of WHICH particles fire is taken from truth (we generate a
    shower for exactly the photons that truly have >=1 calo hit) — this isolates
    SHOWER quality from the separate "does it fire" question.
  - Not held-out (model capacity is far too small to memorise ~4M points, so a
    two-sample test on training-shard events is still meaningful; a held-out
    shard is a later refinement).

Classifier: small torch MLP, real=0 gen=1, trained on half the events, rank-AUC
on the other half. AUC -> 0.5 == indistinguishable. Also reports per-feature
real-vs-gen means so we see WHERE it discriminates.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from genpu.flow.calo_flow import CaloFlow

PF_LOGPT, PF_ETA, PF_PHI, PF_PDG, PF_CHARGE, PF_MASS = range(6)
AUX_PRIMARY, AUX_PARENT, AUX_VX, AUX_VY, AUX_VZ, AUX_ENERGY = range(6)
CH_ETA, CH_PHI, CH_LOGE, CH_FRAC, CH_DET = range(5)


def event_features(eta, phi, E, n_src):
    """Position-invariant per-event observables (real & gen use the same fn)."""
    logE = np.log(E + 1e-12)
    tot = E.sum()
    p90 = np.percentile(logE, 90) if len(logE) else -30.0
    return np.array([
        len(E),                        # total hits (gen N can differ from truth)
        np.log(tot + 1e-12),           # log total energy
        logE.mean(), logE.std(),       # cell-energy spectrum shape
        logE.max(), p90,               # leading / high cells
        float((logE < np.log(5e-5) + 0.5).mean()),  # fraction near the floor
        len(E) / max(n_src, 1),        # mean cells per source photon
    ], dtype=np.float32)


def rank_auc(scores, labels):
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    pos = labels == 1
    n_pos, n_neg = pos.sum(), (~pos).sum()
    if n_pos == 0 or n_neg == 0:
        return 0.5
    return (ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--preproc_dir", default="/scratch/gpfs/IOJALVO/lv7805/genpu_data/preprocessed")
    ap.add_argument("--slice", default="/scratch/gpfs/IOJALVO/lv7805/genpu_data/calo_slice/photon_ctr.npz",
                    help="only used for its norm buffers (checkpoint also carries them)")
    ap.add_argument("--pdg_class", type=int, default=2)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--max_events", type=int, default=0, help="0 = all")
    ap.add_argument("--max_particles", type=int, default=0, help="0 = all; subsample photons (smoke only)")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0); rng = np.random.default_rng(0)

    # ---- model ----
    dsl = np.load(args.slice)
    norm = {k: dsl[k] for k in ["cont_mean", "cont_std", "glob_mean", "glob_std", "pts_mean", "pts_std"]}
    log_floor = float(np.log(5e-5))
    model = CaloFlow(norm).to(dev)
    model.load_state_dict(torch.load(args.ckpt, map_location=dev)["model"]); model.eval()

    # ---- photons with calo, from one shard ----
    d = np.load(Path(args.preproc_dir) / f"shard_{args.shard:04d}_stage2.npz")
    pf, aux, ch, off, eid = (d["particle_features"], d["particle_aux"],
                             d["calo_hits_flat"], d["calo_offsets"], d["event_ids"])
    ncal = np.diff(off)
    sel = np.where((pf[:, PF_PDG] == args.pdg_class) & (ncal >= 1))[0]
    if args.max_particles and len(sel) > args.max_particles:
        sel = sel[:args.max_particles]  # smoke only (breaks event completeness)

    # conditioning (7 CONT_FEATURES) + pdg for each selected photon
    vr = np.hypot(aux[sel, AUX_VX], aux[sel, AUX_VY])
    logE = np.log(np.clip(aux[sel, AUX_ENERGY], 1e-6, None))
    cont = np.stack([pf[sel, PF_LOGPT], pf[sel, PF_ETA], logE, pf[sel, PF_CHARGE],
                     pf[sel, PF_MASS], vr, aux[sel, AUX_VZ]], axis=1).astype(np.float32)
    pdg = pf[sel, PF_PDG].astype(np.int64)
    p_eta, p_phi = pf[sel, PF_ETA], pf[sel, PF_PHI]

    # ---- generate showers for all selected photons (batched) ----
    contS = torch.as_tensor((cont - norm["cont_mean"]) / norm["cont_std"], device=dev)
    pdgT = torch.as_tensor(pdg, device=dev)
    with torch.no_grad():
        ce = model.cond_embed(contS, pdgT)
        g_std = model.glob.sample(ce)                       # standardised globals
        g = model.unstd_glob(g_std).cpu().numpy()
    g_n = np.clip(np.round(np.exp(g[:, 1])).astype(int), 1, 128)
    repn = np.repeat(np.arange(len(sel)), g_n)
    rep = torch.as_tensor(repn, device=dev)
    with torch.no_grad():
        pos_std = model.points.sample(ce[rep], g_std[rep][:, :2], steps=args.steps)     # (P,2)
        logE = model.energy.sample(ce[rep], model.log_floor).cpu().numpy()
        pos = model.unstd_pos(pos_std).cpu().numpy()
    gen_E = np.exp(logE).astype(np.float32)
    gen_eta = p_eta[repn] + pos[:, 0]
    gen_phi = p_phi[repn] + pos[:, 1]
    gen_src = repn

    # ---- group by event, build real & gen event feature rows ----
    ev_of_photon = eid[sel]
    uev = np.unique(ev_of_photon)
    if args.max_events and len(uev) > args.max_events:
        uev = rng.choice(uev, args.max_events, replace=False)
    Xr, Xg = [], []
    for ev in uev:
        pm = np.where(ev_of_photon == ev)[0]           # photons in this event
        n_src = len(pm)
        # real hits of these photons
        re, rp, rE = [], [], []
        for i in pm:
            a, b = off[sel[i]], off[sel[i] + 1]
            re.append(ch[a:b, CH_ETA]); rp.append(ch[a:b, CH_PHI]); rE.append(np.exp(ch[a:b, CH_LOGE]))
        Xr.append(event_features(np.concatenate(re), np.concatenate(rp), np.concatenate(rE), n_src))
        # gen hits of these photons
        gmask = np.isin(gen_src, pm)
        Xg.append(event_features(gen_eta[gmask], gen_phi[gmask], gen_E[gmask], n_src))
    Xr, Xg = np.array(Xr), np.array(Xg)

    # ---- two-sample classifier (torch MLP, rank-AUC on held-out half) ----
    X = np.concatenate([Xr, Xg]); y = np.concatenate([np.zeros(len(Xr)), np.ones(len(Xg))])
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xs = (X - mu) / sd
    perm = rng.permutation(len(X)); ntr = len(X) // 2
    tr, te = perm[:ntr], perm[ntr:]
    Xt = torch.as_tensor(Xs, dtype=torch.float32, device=dev)
    yt = torch.as_tensor(y, dtype=torch.float32, device=dev)
    clf = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 64), torch.nn.SiLU(),
                              torch.nn.Linear(64, 64), torch.nn.SiLU(), torch.nn.Linear(64, 1)).to(dev)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3)
    tri = torch.as_tensor(tr, device=dev)
    for _ in range(800):
        opt.zero_grad()
        logit = clf(Xt[tri]).squeeze(-1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, yt[tri])
        loss.backward(); opt.step()
    with torch.no_grad():
        s_te = clf(Xt[torch.as_tensor(te, device=dev)]).squeeze(-1).cpu().numpy()
    auc = rank_auc(s_te, y[te])

    feat_names = ["n_hits", "log_totE", "logE_mean", "logE_std", "logE_max", "logE_p90",
                  "frac_near_floor", "cells_per_photon"]
    print("=" * 60)
    print(f"CALO EVENT GATE (photon-only) — {Path(args.ckpt).name}")
    print("=" * 60)
    print(f"events: {len(uev)}  (real {len(Xr)} vs gen {len(Xg)})   test AUC = {auc:.4f}   (0.5 = indistinguishable)")
    print(f"{'feature':18s} {'real_mean':>12s} {'gen_mean':>12s} {'|Δ|/σ':>8s}")
    for j, nm in enumerate(feat_names):
        dstd = abs(Xr[:, j].mean() - Xg[:, j].mean()) / (X[:, j].std() + 1e-9)
        print(f"  {nm:16s} {Xr[:,j].mean():12.3f} {Xg[:,j].mean():12.3f} {dstd:8.2f}")
    out = Path(args.ckpt).parent / "eval" / "event_gate.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"auc": float(auc), "n_events": int(len(uev)),
                               "features": feat_names,
                               "real_mean": Xr.mean(0).tolist(), "gen_mean": Xg.mean(0).tolist()}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
