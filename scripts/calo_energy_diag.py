"""Why is the pion calo energy wrong? Three questions, answered on the slice + checkpoint.

(Q1) TAIL: can EnergyHead emit unphysical cells? Its log-E is a Gaussian mixture with
     log_sigma clamped at +3 and NO upper bound (only clamp(min=log_floor-12)), so a tail
     draw is exp(mu + 20 sigma). Compare sampled logE quantiles/max against real.
(Q2) SHOWER-LEVEL LATENT: decompose per-cell logE variance into between-shower and
     within-shower. EnergyHead conditions on the PARTICLE only, so it can reproduce only the
     part of the between-shower variance that particle features predict; the rest is emitted
     as independent per-cell noise -> per-shower totals too broad.
(Q3) CONSISTENCY: the GlobalHead samples total_logE but generation never uses it (the total
     is the sum of i.i.d. cells). Measure what response/resolution you would get by instead
     rescaling cells to the sampled total (the "energy-partition" fix), using REAL totals as
     the oracle for what a perfect GlobalHead buys.

CPU-only (no ODE sampling needed), so it runs on the login node.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from genpu.flow.calo_flow import CaloFlow


def load_model(ckpt_path, dev="cpu"):
    ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)
    sd = ckpt["model"]; src = ckpt.get("norm", sd)

    def _get(base):
        for k in (f"cont_{base}", f"cond_{base}"):
            if k in src:
                v = src[k]; return v.cpu().numpy() if hasattr(v, "cpu") else np.asarray(v)
        raise KeyError(base)
    norm = {"cont_mean": _get("mean"), "cont_std": _get("std")}
    for k in ["glob_mean", "glob_std", "pts_mean", "pts_std"]:
        v = src[k]; norm[k] = v.cpu().numpy() if hasattr(v, "cpu") else np.asarray(v)
    model = CaloFlow(norm).to(dev)
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model, norm


def qsum(x, name):
    q = np.percentile(x, [0.1, 1, 50, 99, 99.9, 99.99])
    return (f"  {name:>12}: min {x.min():8.2f}  q0.1 {q[0]:7.2f}  q1 {q[1]:7.2f}  med {q[2]:7.2f}  "
            f"q99 {q[3]:7.2f}  q99.9 {q[4]:7.2f}  q99.99 {q[5]:7.2f}  MAX {x.max():8.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", default="pion")
    ap.add_argument("--max_showers", type=int, default=300000)
    ap.add_argument("--out", default="/home/lv7805/genpu/plots/calo/metrics")
    args = ap.parse_args()
    rng = np.random.default_rng(0); torch.manual_seed(0)

    d = np.load(args.slice)
    cont, pdg, glob, pts, off = d["cont"], d["pdg"], d["glob"], d["points_flat"], d["offsets"]
    S = cont.shape[0]
    if S > args.max_showers:      # contiguous prefix keeps offsets valid
        S = args.max_showers; cont, pdg, glob = cont[:S], pdg[:S], glob[:S]
        off = off[:S + 1]; pts = pts[:off[-1]]
    n = np.diff(off).astype(np.int64)
    logE = pts[:, 2].astype(np.float64)                 # PHYSICAL log-E per cell
    sh = np.repeat(np.arange(S), n)
    print(f"[{args.tag}] {S} showers, {len(logE)} cells, cells/shower mean {n.mean():.1f} med {np.median(n):.0f}")

    # ---------- Q2: variance decomposition of per-cell logE ----------
    sh_mean = np.bincount(sh, weights=logE, minlength=S) / n
    within = logE - sh_mean[sh]
    v_tot = logE.var(); v_btw = np.var(np.repeat(sh_mean, n)); v_wtn = within.var()
    # how much of the BETWEEN-shower variance do the particle features explain? (linear probe)
    Xc = np.concatenate([cont, np.ones((S, 1), np.float32)], 1).astype(np.float64)
    w = np.linalg.lstsq(Xc, sh_mean, rcond=None)[0]
    r2_cond = 1.0 - np.var(sh_mean - Xc @ w) / np.var(sh_mean)
    # ...and if it ALSO saw the shower total+count (i.e. the GlobalHead output)?
    Xg = np.concatenate([Xc, glob[:, :2].astype(np.float64)], 1)
    wg = np.linalg.lstsq(Xg, sh_mean, rcond=None)[0]
    r2_glob = 1.0 - np.var(sh_mean - Xg @ wg) / np.var(sh_mean)
    print("\nQ2 per-cell logE variance decomposition")
    print(f"  total var {v_tot:.3f} = between-shower {v_btw:.3f} ({100*v_btw/v_tot:.1f}%) "
          f"+ within-shower {v_wtn:.3f} ({100*v_wtn/v_tot:.1f}%)")
    print(f"  between-shower mean logE: R^2 from particle cond alone = {r2_cond:.3f}   "
          f"with (total_logE, log_n) added = {r2_glob:.3f}")
    print(f"  -> cond-only EnergyHead can address {100*r2_cond*v_btw/v_tot:.1f}% of cell-logE variance; "
          f"the remaining {100*(1-r2_cond)*v_btw/v_tot:.1f}% shower-level part is emitted as i.i.d. noise")

    # ---------- Q1: what does the trained EnergyHead actually emit? ----------
    model, norm = load_model(args.ckpt)
    contS = torch.as_tensor((cont - norm["cont_mean"]) / norm["cont_std"], dtype=torch.float32)
    pdgT = torch.as_tensor(pdg, dtype=torch.long)
    with torch.no_grad():
        ce = model.cond_embed(contS, pdgT)
        rep = torch.as_tensor(sh)                       # one gen cell per REAL cell (real n: isolates energy)
        gen_logE = model.energy.sample(ce[rep], model.log_floor).double().numpy()
    print("\nQ1 per-cell logE: real vs generated (real cell multiplicity, so this is ENERGY only)")
    print(qsum(logE, "real")); print(qsum(gen_logE, "gen"))
    n_over = int((gen_logE > logE.max()).sum())
    print(f"  cells above the real MAX: {n_over} ({100*n_over/len(gen_logE):.4f}%)   "
          f"energy in them: {np.exp(gen_logE[gen_logE > logE.max()]).sum():.1f} GeV "
          f"vs total real {np.exp(logE).sum():.1f} GeV")

    # ---------- Q3: totals — sum-of-cells vs rescale-to-total ----------
    E_real = np.bincount(sh, weights=np.exp(logE), minlength=S)
    E_gen = np.bincount(sh, weights=np.exp(gen_logE), minlength=S)
    E_part = np.exp(cont[:, 2].astype(np.float64))       # cond col 2 = log_E_particle
    r_r, r_g = E_real / E_part, E_gen / E_part
    def stat(r):
        return (f"mean {r.mean():10.3f}  median {np.median(r):8.4f}  "
                f"IQR/med {(np.percentile(r,75)-np.percentile(r,25))/np.median(r):6.3f}  max {r.max():.3e}")
    print("\nQ3 shower total E_reco/E_true")
    print(f"  real           : {stat(r_r)}")
    print(f"  gen sum-of-cells: {stat(r_g)}")
    # oracle: cap the tail at the real max cell energy
    capped = np.minimum(gen_logE, logE.max())
    r_cap = np.bincount(sh, weights=np.exp(capped), minlength=S) / E_part
    print(f"  gen, tail capped at real max cell: {stat(r_cap)}")
    # oracle: rescale each shower to its (here: REAL) total -> what a perfect GlobalHead buys
    scale = np.where(E_gen > 0, E_real / np.maximum(E_gen, 1e-30), 1.0)
    r_rs = np.bincount(sh, weights=np.exp(gen_logE) * scale[sh], minlength=S) / E_part
    print(f"  gen, rescaled to real total (oracle): {stat(r_rs)}")
    # resolution in energy bins, real vs the three gen variants
    print("\n  resolution (std/mean of E_reco/E_true) in E_true sextiles")
    bins = np.quantile(np.log10(E_part), np.linspace(0, 1, 7))
    bi = np.clip(np.digitize(np.log10(E_part), bins[1:-1]), 0, 5)
    print(f"    {'logE_med':>9} {'n':>8} {'real':>8} {'gen':>10} {'gen_cap':>9} {'gen_rescale':>11}")
    for b in range(6):
        m = bi == b
        if m.sum() < 50:
            continue
        def reso(r): return r[m].std() / (r[m].mean() + 1e-12)
        print(f"    {np.median(np.log10(E_part[m])):>9.2f} {m.sum():>8} {reso(r_r):>8.3f} "
              f"{reso(r_g):>10.3f} {reso(r_cap):>9.3f} {reso(r_rs):>11.3f}")

    # ---------- Q4: is the GlobalHead's sampled total good enough to normalise TO? ----------
    # (Q3's rescale used the REAL total = oracle. The honest fix uses the SAMPLED total, so the
    #  partition architecture is only as good as GlobalHead's conditional total_logE.)
    with torch.no_grad():
        g_std = model.glob.sample(ce)
        g = model.unstd_glob(g_std).double().numpy()
    E_glob = np.exp(g[:, 0])
    r_gl = E_glob / E_part
    n_gen = np.clip(np.round(np.exp(g[:, 1])), 1, 128)
    print("\nQ4 GlobalHead total_logE (the normalisation target)")
    print(f"  real total  : {stat(r_r)}")
    print(f"  sampled total: {stat(r_gl)}")
    print(f"  cells/shower real mean {n.mean():.2f} med {np.median(n):.0f} | "
          f"sampled mean {n_gen.mean():.2f} med {np.median(n_gen):.0f}")
    print(f"  {'logE_med':>9} {'n':>8} {'reso_real':>10} {'reso_globalhead':>16}")
    for b in range(6):
        m = bi == b
        if m.sum() < 50:
            continue
        print(f"  {np.median(np.log10(E_part[m])):>9.2f} {m.sum():>8} "
              f"{r_r[m].std()/r_r[m].mean():>10.3f} {r_gl[m].std()/r_gl[m].mean():>16.3f}")
    from scipy.stats import wasserstein_distance
    w_tot = wasserstein_distance(np.log(E_real + 1e-30), np.log(E_glob + 1e-30)) / np.log(E_real + 1e-30).std()
    print(f"  W/sigma on log total: sampled-vs-real {w_tot:.4f}  "
          f"(sum-of-cells-vs-real {wasserstein_distance(np.log(E_real+1e-30), np.log(E_gen+1e-30))/np.log(E_real+1e-30).std():.4f})")

    out = {"tag": args.tag, "showers": int(S), "cells": int(len(logE)),
           "resp_globalhead_mean": float(r_gl.mean()), "w_logtotal_globalhead": float(w_tot),
           "var_between_frac": float(v_btw / v_tot), "r2_cond": float(r2_cond), "r2_with_glob": float(r2_glob),
           "real_logE_max": float(logE.max()), "gen_logE_max": float(gen_logE.max()),
           "cells_above_real_max": n_over,
           "resp_real_mean": float(r_r.mean()), "resp_gen_mean": float(r_g.mean()),
           "resp_gen_capped_mean": float(r_cap.mean()), "resp_gen_rescaled_mean": float(r_rs.mean())}
    o = Path(args.out); o.mkdir(parents=True, exist_ok=True)
    (o / f"energy_diag_{args.tag}.json").write_text(json.dumps(out, indent=2))
    print("\nwrote", o / f"energy_diag_{args.tag}.json")


if __name__ == "__main__":
    main()
