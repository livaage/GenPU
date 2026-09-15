"""Why is the calo event gate at 0.81 when no single feature exceeds 0.58?

`calo_metrics.py` reports a composite AUC and a per-feature AUC table. On the v2 e+- model those
disagree badly: composite 0.81, best single feature `logE_max` 0.576. A per-feature table cannot
explain that, and the usual reading -- "the classifier is using multivariate structure" -- is not
actionable until you know WHICH structure.

Two things can produce a composite far above every marginal:

  (a) AGGREGATION. Many independent small marginal offsets add in quadrature. For k features with
      per-feature AUCs a_i, an equal-variance Gaussian model gives d_i = sqrt(2)*Phi^-1(a_i) and
      a combined AUC of Phi(sqrt(sum_i d_i^2)/sqrt(2)). No joint error is needed. This script
      prints that prediction so the composite can be compared against it.

  (b) DEPENDENCE. The marginals are right but the JOINT is wrong -- e.g. generated `n_cells` is
      correct on its own and correct on its own, but wrong as a function of how many showers the
      event holds. Every single-feature AUC stays at 0.50 and the composite still climbs.

The decomposition here separates them by destroying one at a time, on the SAME classifier:

  full          raw features
  marginals     each column permuted WITHIN its own class -> dependence destroyed, marginals exact
  copula        each column rank-transformed WITHIN its own class -> marginals made identical
                (uniform), dependence untouched
  copula-quad   a quadratic logistic model in copula space -> the part of the dependence gap that
                lives in the CORRELATION MATRIX alone, vs what needs higher order

and then names the responsible structure three ways: pairwise interaction excess
AUC(i,j) - max(AUC(i),AUC(j)), the Spearman rho_gen - rho_real matrix, and -- using the fact that
real and generated events are PAIRED (same event, same truth particle list, same n_src) -- the
per-event residual gen-real and what it is conditional on.

Input is the npz written by `calo_metrics.py --dump_features`. Cheap: a few thousand events by ~12
features, CPU, seconds to a couple of minutes. Nothing here regenerates showers.
"""
from __future__ import annotations
import argparse, itertools, json
from pathlib import Path
import numpy as np
import torch
from scipy.special import ndtri


def rank_auc(s, y):
    o = np.argsort(s); ra = np.empty_like(o, float); ra[o] = np.arange(1, len(s) + 1)
    p = y == 1; npo, nne = p.sum(), (~p).sum()
    return 0.5 if npo == 0 or nne == 0 else (ra[p].sum() - npo * (npo + 1) / 2) / (npo * nne)


def gate(Xr, Xg, seed=0, steps=800, hidden=64, dev="cpu", return_scores=False):
    """The calo_metrics two-sample gate, verbatim in structure: 50/50 split, 2x64 SiLU MLP,
    800 Adam steps, held-out rank AUC. Kept identical so numbers here are comparable to the
    ones in metrics_*.json rather than to a different classifier."""
    X = np.concatenate([Xr, Xg]).astype(np.float64)
    y = np.concatenate([np.zeros(len(Xr)), np.ones(len(Xg))])
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-6)
    g = np.random.default_rng(seed)
    perm = g.permutation(len(X)); ntr = len(X) // 2
    torch.manual_seed(seed)
    Xt = torch.as_tensor(Xs, dtype=torch.float32, device=dev)
    yt = torch.as_tensor(y, dtype=torch.float32, device=dev)
    clf = torch.nn.Sequential(torch.nn.Linear(X.shape[1], hidden), torch.nn.SiLU(),
                              torch.nn.Linear(hidden, hidden), torch.nn.SiLU(),
                              torch.nn.Linear(hidden, 1)).to(dev)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3)
    tri = torch.as_tensor(perm[:ntr], device=dev)
    for _ in range(steps):
        opt.zero_grad()
        torch.nn.functional.binary_cross_entropy_with_logits(
            clf(Xt[tri]).squeeze(-1), yt[tri]).backward()
        opt.step()
    te = perm[ntr:]
    with torch.no_grad():
        s = clf(Xt[torch.as_tensor(te, device=dev)]).squeeze(-1).cpu().numpy()
    auc = float(rank_auc(s, y[te]))
    return (auc, s, te, y) if return_scores else auc


def gate_multi(Xr, Xg, seeds, **kw):
    a = [gate(Xr, Xg, seed=s, **kw) for s in seeds]
    return float(np.mean(a)), float(np.std(a)), a


def to_copula(X):
    """Rank -> uniform -> standard normal, WITHIN this array. Applied to each class separately the
    marginals become identical by construction, so any surviving separation is dependence."""
    n = len(X)
    Z = np.empty_like(X, dtype=np.float64)
    for j in range(X.shape[1]):
        o = np.argsort(X[:, j], kind="stable")
        r = np.empty(n); r[o] = np.arange(1, n + 1)
        Z[:, j] = ndtri(r / (n + 1.0))
    return Z


def quad_features(Z):
    """[z_j] + [z_i z_j, i<=j]. A logistic model on this is exactly a second-order (Gaussian)
    discriminant, so it sees the covariance difference and nothing beyond it."""
    k = Z.shape[1]
    prods = [Z[:, i] * Z[:, j] for i, j in itertools.combinations_with_replacement(range(k), 2)]
    return np.concatenate([Z, np.stack(prods, 1)], 1)


def logistic_auc(Xr, Xg, seed=0, steps=2000, l2=1e-3, dev="cpu"):
    X = np.concatenate([Xr, Xg]).astype(np.float64)
    y = np.concatenate([np.zeros(len(Xr)), np.ones(len(Xg))])
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-6)
    g = np.random.default_rng(seed); perm = g.permutation(len(X)); ntr = len(X) // 2
    torch.manual_seed(seed)
    Xt = torch.as_tensor(Xs, dtype=torch.float32, device=dev)
    yt = torch.as_tensor(y, dtype=torch.float32, device=dev)
    lin = torch.nn.Linear(X.shape[1], 1).to(dev)
    opt = torch.optim.Adam(lin.parameters(), lr=1e-2, weight_decay=l2)
    tri = torch.as_tensor(perm[:ntr], device=dev)
    for _ in range(steps):
        opt.zero_grad()
        torch.nn.functional.binary_cross_entropy_with_logits(
            lin(Xt[tri]).squeeze(-1), yt[tri]).backward()
        opt.step()
    te = perm[ntr:]
    with torch.no_grad():
        s = lin(Xt[torch.as_tensor(te, device=dev)]).squeeze(-1).cpu().numpy()
    return float(rank_auc(s, y[te]))


def spearman_mat(X):
    n, k = X.shape
    R = np.empty_like(X, dtype=np.float64)
    for j in range(k):
        o = np.argsort(X[:, j], kind="stable"); r = np.empty(n); r[o] = np.arange(n)
        R[:, j] = r
    R = (R - R.mean(0)) / (R.std(0) + 1e-12)
    return (R.T @ R) / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, help="npz from calo_metrics.py --dump_features")
    ap.add_argument("--seeds", type=int, default=3, help="classifier seeds per gate (the gate is "
                    "itself noisy; a single seed cannot separate a 0.01 effect)")
    ap.add_argument("--pairs", action="store_true", help="all pairwise 2-feature gates (k*(k-1)/2 "
                    "classifier fits; ~1-3 min for k=12)")
    ap.add_argument("--gate_steps", type=int, default=800,
                    help="Adam steps per classifier fit. 800 matches calo_metrics.py exactly; this "
                         "script fits ~100 of them, so lower it only for a smoke run.")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--outdir", default="/home/lv7805/genpu/plots/calo/metrics")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = list(range(args.seeds))
    GK = dict(dev=dev, steps=args.gate_steps)

    d = np.load(args.features, allow_pickle=True)
    Xr, Xg = d["Xr"].astype(np.float64), d["Xg"].astype(np.float64)
    names = [str(x) for x in d["feature_names"]]
    n_src = d["n_src"].astype(np.float64)
    tag = args.tag or str(d["tag"])
    k = Xr.shape[1]
    print(f"{tag}: {len(Xr)} events x {k} features (paired), device={dev}, seeds={seeds}\n")

    # non-finite guard: width/depth features are 0.0 for empty events, but a NaN would silently
    # poison a standardisation and every AUC downstream
    bad = ~np.isfinite(Xr).all(1) | ~np.isfinite(Xg).all(1)
    if bad.any():
        print(f"dropping {bad.sum()} events with non-finite features")
        Xr, Xg, n_src = Xr[~bad], Xg[~bad], n_src[~bad]

    out = {"tag": tag, "n_events": int(len(Xr)), "features": names, "gate_seeds": seeds}

    # ---------- 1. per-feature AUC + what pure aggregation of them would give ----------
    per_feat = {}
    for j, nm in enumerate(names):
        a = rank_auc(np.concatenate([Xr[:, j], Xg[:, j]]), np.r_[np.zeros(len(Xr)), np.ones(len(Xg))])
        per_feat[nm] = float(max(a, 1 - a))
    dsq = sum((np.sqrt(2) * ndtri(min(max(a, 0.5), 0.999999))) ** 2 for a in per_feat.values())
    from scipy.stats import norm as _norm
    auc_indep = float(_norm.cdf(np.sqrt(dsq) / np.sqrt(2)))
    out["per_feature_auc"] = {n_: round(v, 4) for n_, v in per_feat.items()}
    out["auc_if_marginals_independent"] = round(auc_indep, 4)

    # ---------- 2. the decomposition ----------
    full_m, full_s, _ = gate_multi(Xr, Xg, seeds, **GK)

    rng = np.random.default_rng(0)
    Pr, Pg = Xr.copy(), Xg.copy()
    for j in range(k):                       # permute WITHIN class: marginals exact, joint gone
        Pr[:, j] = Pr[rng.permutation(len(Pr)), j]
        Pg[:, j] = Pg[rng.permutation(len(Pg)), j]
    marg_m, marg_s, _ = gate_multi(Pr, Pg, seeds, **GK)

    Zr, Zg = to_copula(Xr), to_copula(Xg)    # marginals identical, joint untouched
    cop_m, cop_s, _ = gate_multi(Zr, Zg, seeds, **GK)
    quad = float(np.mean([logistic_auc(quad_features(Zr), quad_features(Zg), seed=s, dev=dev)
                          for s in seeds]))
    lin_cop = float(np.mean([logistic_auc(Zr, Zg, seed=s, dev=dev) for s in seeds]))

    out["decomposition"] = {
        "full": [round(full_m, 4), round(full_s, 4)],
        "marginals_only_permuted": [round(marg_m, 4), round(marg_s, 4)],
        "copula_only_rank_matched": [round(cop_m, 4), round(cop_s, 4)],
        "copula_quadratic_logistic": round(quad, 4),
        "copula_linear_logistic_control": round(lin_cop, 4),
        "best_single_feature": round(max(per_feat.values()), 4),
    }
    print("=== decomposition (held-out AUC, 0.5 = indistinguishable) ===")
    print(f"  full (raw features)                {full_m:.4f} +- {full_s:.4f}")
    print(f"  marginals only (joint destroyed)   {marg_m:.4f} +- {marg_s:.4f}")
    print(f"  copula only (marginals matched)    {cop_m:.4f} +- {cop_s:.4f}")
    print(f"  copula, quadratic logistic         {quad:.4f}   (= correlation-matrix part)")
    print(f"  copula, linear logistic [control]  {lin_cop:.4f}   (should be ~0.5 by construction)")
    print(f"  best single feature                {max(per_feat.values()):.4f}")
    print(f"  aggregation prediction (indep.)    {auc_indep:.4f}   (what the marginals alone buy "
          f"if independent)\n")

    # ---------- 3. leave-one-out and greedy forward selection on the full gate ----------
    loo = {}
    for j, nm in enumerate(names):
        cols = [c for c in range(k) if c != j]
        m, _, _ = gate_multi(Xr[:, cols], Xg[:, cols], seeds, **GK)
        loo[nm] = round(full_m - m, 4)
    out["leave_one_out_delta_auc"] = loo
    print("=== leave-one-out (AUC lost when this feature is REMOVED from the full gate) ===")
    for nm, v in sorted(loo.items(), key=lambda kv: -kv[1]):
        print(f"  {nm:>18} {v:+.4f}")

    chosen, traj, remaining = [], [], list(range(k))
    while remaining:
        best = None
        for j in remaining:
            m, _, _ = gate_multi(Xr[:, chosen + [j]], Xg[:, chosen + [j]], seeds, **GK)
            if best is None or m > best[0]:
                best = (m, j)
        chosen.append(best[1]); remaining.remove(best[1])
        traj.append({"added": names[best[1]], "auc": round(best[0], 4)})
        if len(chosen) >= min(k, 8):
            break
    out["greedy_forward"] = traj
    print("\n=== greedy forward selection ===")
    prev = 0.5
    for t in traj:
        print(f"  +{t['added']:>18}  -> {t['auc']:.4f}  ({t['auc']-prev:+.4f})"); prev = t["auc"]

    # ---------- 4. pairwise interaction excess ----------
    if args.pairs:
        single = {}
        for j in range(k):
            single[j], _, _ = gate_multi(Xr[:, [j]], Xg[:, [j]], seeds, **GK)
        pairs = []
        for i, j in itertools.combinations(range(k), 2):
            m, _, _ = gate_multi(Xr[:, [i, j]], Xg[:, [i, j]], seeds, **GK)
            pairs.append({"pair": [names[i], names[j]], "auc": round(m, 4),
                          "excess": round(m - max(single[i], single[j]), 4)})
        pairs.sort(key=lambda p: -p["excess"])
        out["single_feature_gate_auc"] = {names[j]: round(v, 4) for j, v in single.items()}
        out["pairwise"] = pairs
        print("\n=== pairwise interaction excess: AUC(i,j) - max(AUC(i), AUC(j)) ===")
        for p in pairs[:12]:
            print(f"  {p['pair'][0]:>18} x {p['pair'][1]:<18} {p['auc']:.4f}  excess {p['excess']:+.4f}")

    # ---------- 5. Spearman structure: which correlation is generated wrong ----------
    Sr, Sg = spearman_mat(Xr), spearman_mat(Xg)
    dS = Sg - Sr
    ent = [{"pair": [names[i], names[j]], "real": round(float(Sr[i, j]), 3),
            "gen": round(float(Sg[i, j]), 3), "delta": round(float(dS[i, j]), 3)}
           for i, j in itertools.combinations(range(k), 2)]
    ent.sort(key=lambda e: -abs(e["delta"]))
    out["spearman_delta"] = ent
    print("\n=== rank correlation, generated minus real (top |delta|) ===")
    print(f"  {'pair':>40} {'real':>7} {'gen':>7} {'delta':>7}")
    for e in ent[:12]:
        print(f"  {e['pair'][0]:>19} x {e['pair'][1]:<18} {e['real']:>7.3f} {e['gen']:>7.3f} "
              f"{e['delta']:>+7.3f}")

    # ---------- 6. PAIRED residuals: the bias the unpaired gate cannot see ----------
    # Row i of Xr and Xg is the same event with the same truth particle list, so gen-real is a
    # matched-pair residual. A feature can be marginally unbiased (single-feature AUC 0.50) and
    # still be systematically wrong given the event's occupancy -- exactly the failure a
    # multivariate classifier converts into AUC.
    resid = {}
    ls = np.log(n_src + 1.0)
    for j, nm in enumerate(names):
        dj = Xg[:, j] - Xr[:, j]
        sd = Xr[:, j].std() + 1e-12
        cc = float(np.corrcoef(dj, ls)[0, 1]) if dj.std() > 0 else 0.0
        cs = float(np.corrcoef(dj, Xr[:, j])[0, 1]) if dj.std() > 0 else 0.0
        resid[nm] = {"bias_over_sigma": round(float(dj.mean() / sd), 4),
                     "paired_scatter_over_sigma": round(float(dj.std() / sd), 4),
                     "corr_with_log_n_src": round(cc, 3),
                     "corr_with_own_real_value": round(cs, 3)}
    out["paired_residuals"] = resid
    print("\n=== paired per-event residual (gen - real), in units of the REAL spread ===")
    print(f"  {'feature':>18} {'bias/sig':>9} {'scatter/sig':>12} {'r(d,log n_src)':>15} {'r(d,real)':>10}")
    for nm, v in sorted(resid.items(), key=lambda kv: -abs(kv[1]["bias_over_sigma"])):
        print(f"  {nm:>18} {v['bias_over_sigma']:>+9.4f} {v['paired_scatter_over_sigma']:>12.4f} "
              f"{v['corr_with_log_n_src']:>+15.3f} {v['corr_with_own_real_value']:>+10.3f}")

    # ---------- 7. what the full classifier actually flags ----------
    auc1, s, te, y = gate(Xr, Xg, seed=0, return_scores=True, **GK)
    X = np.concatenate([Xr, Xg]); mu, sg = X.mean(0), X.std(0) + 1e-12
    gm = y[te] == 1
    top = np.argsort(-s)                      # most confidently GENERATED
    tn = max(int(0.1 * gm.sum()), 20)
    sel = te[top[:tn]]
    prof = {names[j]: round(float(((X[sel, j] - mu[j]) / sg[j]).mean()
                                  - ((X[te[gm], j] - mu[j]) / sg[j]).mean()), 3)
            for j in range(k)}
    out["most_flagged_decile_shift"] = prof
    print(f"\n=== the {tn} events the gate is most sure are GENERATED (seed 0, AUC {auc1:.4f}) ===")
    print("    mean z-shift of each feature vs all generated events:")
    for nm, v in sorted(prof.items(), key=lambda kv: -abs(kv[1]))[:8]:
        print(f"  {nm:>18} {v:>+7.3f}")

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    fp = outdir / f"gate_diag_{tag}.json"
    fp.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {fp}")


if __name__ == "__main__":
    main()
