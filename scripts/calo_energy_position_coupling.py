"""Does a calo cell's ENERGY depend on WHERE that cell sits in its shower?

The 2026-08-25 decomposition put the v2 calo defect majority in the JOINT (copula 0.70-0.72 vs
marginals 0.60-0.62) and showed the copula half did not move when pooling moved the marginal half
by 0.13. Nothing tried so far could have moved it. This measures the mechanism proposed for it.

ARCHITECTURAL CLAIM UNDER TEST. In `CaloFlow.sample_showers` the cells of one shower are drawn
conditionally i.i.d. given (particle embedding, sampled global): positions from `PointCFM.sample`
(calo_flow.py:537), energies from `EnergyHead.sample` (calo_flow.py:556) -- and the energy head's
`_out` takes only (cond_embed, glob_std[:,idx], n_cells) (calo_flow.py:192). It never sees the
cell's own position. So BY CONSTRUCTION, within a shower:

    E  _||_  (d_eta, d_phi, depth)      given (particle, total_logE, log_n)

and, because the realized point cloud's shape is pure i.i.d. sampling noise while the realized
energy multiset is pure Bernoulli+mixture noise, ALSO at shower level:

    shape(width, depth profile)  _||_  energy multiset(frac_floor, logE_std, ...)   given the global

Real showers have a hard core and a faint fringe, so neither should hold. That gradient is exactly
the kind of structure a copula gap measures and a marginal AUC cannot see.

NOT THE HYPOTHESIS FALSIFIED ON 2026-08-16. That one was "cells need SHARED RANDOMNESS so the
at-floor COUNT is over-dispersed", killed by D_floor,real = 1.035 for e±. This one predicts
heterogeneous per-cell p(floor) instead -- and a Poisson-binomial with cell-varying p is
*under*-dispersed relative to Binomial(n, p_bar) by Jensen, which that entry noted as its own
conservativeness caveat. So D_real ~= 1 is what a position-dependent floor rate PREDICTS. The two
tests are measuring different things and the earlier result does not constrain this one.

WHAT IS MEASURED

  A. within-shower gradient   p(floor) and <logE> vs the cell's within-shower RANK of r (distance
                              from the realized centroid) and of depth. Rank, not value, so no
                              shower-level scale enters and showers of every size contribute alike.
  B. within-shower Spearman   rho(logE, r) and rho(logE, depth), per shower, then averaged.
  C. shower-level coupling    PARTIAL Spearman of a SHAPE quantity against an ENERGY quantity,
                              controlling for (log_n, log_totE) -- i.e. removing everything the
                              sampled global already carries. This is the bridge from A to the
                              event-level correlations the gate sees. The model's value is
                              analytically ZERO; --ckpt measures it rather than assuming it.

The generated side is the control with a known answer, in the style the 2026-08-25 diagnose script
used synthetic arms: if it does not come back ~0 on C, this script's reading of the architecture is
wrong and the real-side numbers should not be trusted either.

Both sides are re-centred on their OWN realized cell centroid before r is computed. Real slice
points are mean-subtracted by construction (`build_calo_slice_v2.py:243`) but generated ones are
offsets from a SEPARATELY sampled core, so without this the generated r is inflated and the
comparison is biased toward the conclusion.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

LOG_FLOOR = float(np.log(5e-5))
# the gate's definition (`calo_metrics.py:41`), one-sided and wide; kept primary so every number
# here is comparable to a frac_near_floor AUC. The EnergyHead trains on a NARROW two-sided band
# (`calo_flow.py:212`) -- reported alongside, since the head can only learn what it is trained on.
FLOOR_GATE = lambda le: le < LOG_FLOOR + 0.5
FLOOR_HEAD = lambda le: np.abs(le - LOG_FLOOR) < 0.05


# ---------------------------------------------------------------- CSR helpers

def within_rank(vals, off, n_per, jitter=None, rng=None):
    """Normalised rank in [0,1] of `vals` within each CSR group. n=1 groups get 0.5.

    `jitter` breaks ties RANDOMLY (log-E has a large exact atom at the floor). Random tie-breaking
    attenuates a rank correlation toward zero relative to proper mid-ranks, so it can only make
    this test harder to pass, never easier.
    """
    v = vals
    if jitter:
        v = v + rng.normal(0.0, jitter, size=len(v))
    src = np.repeat(np.arange(len(n_per)), n_per)
    order = np.lexsort((v, src))
    pos = np.arange(len(v), dtype=np.float64) - off[src[order]]
    r = np.empty(len(v), np.float64)
    r[order] = pos
    denom = np.maximum(n_per - 1, 1).astype(np.float64)
    out = r / denom[src]
    out[np.repeat(n_per == 1, n_per)] = 0.5
    return out


def group_mean(vals, src, S):
    s = np.zeros(S, np.float64).__iadd__(0)
    s = np.bincount(src, weights=vals, minlength=S)
    c = np.bincount(src, minlength=S).astype(np.float64)
    return s / np.maximum(c, 1)


# ---------------------------------------------------------------- statistics

def rankit(x):
    """Ranks scaled to [0,1]; ties get mid-ranks."""
    from scipy.stats import rankdata
    return (rankdata(x) - 1.0) / max(len(x) - 1, 1)


def partial_spearman(a, b, ctrl):
    """Spearman(a, b) with the columns of `ctrl` regressed out, all rank-transformed first."""
    ra, rb = rankit(a), rankit(b)
    X = np.column_stack([np.ones(len(a))] + [rankit(c) for c in ctrl])
    qa = ra - X @ np.linalg.lstsq(X, ra, rcond=None)[0]
    qb = rb - X @ np.linalg.lstsq(X, rb, rcond=None)[0]
    d = np.sqrt((qa @ qa) * (qb @ qb))
    return float(qa @ qb / d) if d > 0 else 0.0


def within_spearman(x, y, off, n_per, rng, jitter_y=1e-4):
    """Mean over showers of the within-shower Spearman rho(x, y). Showers with n < 3 skipped."""
    rx = within_rank(x, off, n_per, rng=rng)
    ry = within_rank(y, off, n_per, jitter=jitter_y, rng=rng)
    src = np.repeat(np.arange(len(n_per)), n_per)
    keep = n_per >= 3
    mx, my = group_mean(rx, src, len(n_per)), group_mean(ry, src, len(n_per))
    dx, dy = rx - mx[src], ry - my[src]
    num = np.bincount(src, weights=dx * dy, minlength=len(n_per))
    den = np.sqrt(np.bincount(src, weights=dx * dx, minlength=len(n_per))
                  * np.bincount(src, weights=dy * dy, minlength=len(n_per)))
    rho = np.where(den > 0, num / np.maximum(den, 1e-30), 0.0)[keep]
    return float(rho.mean()), float(rho.std() / np.sqrt(max(len(rho), 1))), int(len(rho))


def shuffle_energies_matched_n(logE, n_per, rng):
    """Pair each shower's POSITIONS with another shower's ENERGY multiset, matched on cell count.

    THE NULL FOR PART C, and the one this script's own synthetic check showed is needed: a
    permutation of energies *within* a shower leaves every per-shower energy statistic identical,
    so it cannot test a shower-level correlation at all. Permuting whole multisets ACROSS showers
    of equal n leaves both marginals exactly intact (the multiset distribution and the width
    distribution are unchanged) and destroys only the shape<->energy pairing -- which is precisely
    the independence `CaloFlow.sample_showers` imposes. So this null IS the architecture, measured
    on real cells, and any real-vs-null gap is the coupling the model cannot represent.
    """
    off = np.concatenate([[0], np.cumsum(n_per)]).astype(np.int64)
    out = np.empty_like(logE)
    for nn in np.unique(n_per):
        idx = np.where(n_per == nn)[0]
        ar = np.arange(nn)
        donor = rng.permutation(idx) if len(idx) > 1 else idx
        # SCRAMBLE THE DONATED BLOCK'S INTERNAL ORDER. Pairing cell k of the recipient with cell k
        # of the donor made the null inherit the real gradient (measured 2026-08-27: null
        # rho(logE, r) = -0.062 against a real -0.098, where a null must be 0), because stored cell
        # order tracks both r and log-E. Shuffling within the block leaves the MULTISET untouched,
        # so every shower-level statistic -- i.e. all of part C -- is unchanged, while parts A and B
        # get the zero baseline they require.
        perm_in = np.argsort(rng.random((len(idx), nn)), axis=1)
        srcpos = (off[donor][:, None] + perm_in).ravel()
        dstpos = (off[idx][:, None] + ar).ravel()
        out[dstpos] = logE[srcpos]
    return out


def profile(rank_x, mask_floor, logE, nbin=10):
    """p(floor) and <logE> in bins of a within-shower rank."""
    b = np.clip((rank_x * nbin).astype(np.int64), 0, nbin - 1)
    cnt = np.bincount(b, minlength=nbin).astype(np.float64)
    pf = np.bincount(b, weights=mask_floor.astype(np.float64), minlength=nbin) / np.maximum(cnt, 1)
    me = np.bincount(b, weights=logE, minlength=nbin) / np.maximum(cnt, 1)
    return pf.tolist(), me.tolist(), cnt.astype(np.int64).tolist()


# ---------------------------------------------------------------- one side

def analyse(tag, q_eta, q_phi, depth, logE, off, n_per, n_min, rng, nbin):
    S = len(n_per)
    src = np.repeat(np.arange(S), n_per)
    # RE-CENTRE on the realized centroid (see module docstring)
    ce = group_mean(q_eta.astype(np.float64), src, S)
    cp = group_mean(q_phi.astype(np.float64), src, S)
    de, dp = q_eta - ce[src], q_phi - cp[src]
    r = np.hypot(de, dp)
    le = logE.astype(np.float64)
    fl_g, fl_h = FLOOR_GATE(le), FLOOR_HEAD(le)

    out = {"tag": tag, "showers": int(S), "cells": int(len(le)),
           "cells_per_shower": float(n_per.mean()),
           "frac_floor_gate": float(fl_g.mean()), "frac_floor_head": float(fl_h.mean())}

    # ---- A: profiles vs within-shower rank of r and of depth (showers with n >= 3 only,
    #         since a rank is meaningless for n < 3)
    big = n_per >= 3
    cell_big = np.repeat(big, n_per)
    if big.sum():
        offb = np.concatenate([[0], np.cumsum(n_per[big])]).astype(np.int64)
        nb = n_per[big]
        rr = within_rank(r[cell_big], offb, nb, rng=rng)
        rd = within_rank(depth[cell_big].astype(np.float64), offb, nb, rng=rng)
        for nm, rk in (("r", rr), ("depth", rd)):
            pf, me, cn = profile(rk, fl_g[cell_big], le[cell_big], nbin)
            out[f"profile_vs_{nm}"] = {"p_floor_gate": pf, "mean_logE": me, "n": cn}
            pfh, _, _ = profile(rk, fl_h[cell_big], le[cell_big], nbin)
            out[f"profile_vs_{nm}"]["p_floor_head"] = pfh

    # ---- B: within-shower Spearman
    for nm, x in (("r", r), ("depth", depth.astype(np.float64))):
        m, se, nsh = within_spearman(x, le, off, n_per, rng)
        out[f"within_rho_logE_{nm}"] = {"mean": m, "sem": se, "showers": nsh}

    # ---- C: shower-level SHAPE x ENERGY partial correlations, controlling for the global
    keep = n_per >= n_min
    if keep.sum() >= 100:
        ks = np.where(keep)[0]
        cm = np.repeat(keep, n_per)
        s2 = np.repeat(np.arange(len(ks)), n_per[ks])
        e = np.exp(le[cm])
        n_k = n_per[ks].astype(np.float64)
        tot = np.bincount(s2, weights=e, minlength=len(ks))
        width = np.sqrt(np.bincount(s2, weights=(de[cm] ** 2 + dp[cm] ** 2), minlength=len(ks)) / n_k)
        lem = np.bincount(s2, weights=le[cm], minlength=len(ks)) / n_k
        les = np.sqrt(np.maximum(np.bincount(s2, weights=le[cm] ** 2, minlength=len(ks)) / n_k - lem ** 2, 0))
        ffl = np.bincount(s2, weights=fl_g[cm].astype(np.float64), minlength=len(ks)) / n_k
        lemax = np.full(len(ks), -np.inf)
        np.maximum.at(lemax, s2, le[cm])
        w = e / np.maximum(tot[s2], 1e-30)
        dmean = np.bincount(s2, weights=w * depth[cm], minlength=len(ks))
        dstd = np.sqrt(np.maximum(
            np.bincount(s2, weights=w * (depth[cm] - dmean[s2]) ** 2, minlength=len(ks)), 0))
        # UNWEIGHTED depth spread: the energy-weighted one is a shape-x-energy MIXTURE, so a
        # nonzero partial correlation with an energy quantity could come from the weighting alone.
        dmean_u = np.bincount(s2, weights=depth[cm], minlength=len(ks)) / n_k
        dstd_u = np.sqrt(np.maximum(
            np.bincount(s2, weights=(depth[cm] - dmean_u[s2]) ** 2, minlength=len(ks)) / n_k, 0))
        ctrl = [np.log(n_k), np.log(np.maximum(tot, 1e-30))]
        shape = {"width": width, "depth_mean_unw": dmean_u, "depth_std_unw": dstd_u}
        energy = {"frac_floor": ffl, "logE_std": les, "logE_mean": lem, "logE_max": lemax}
        out["partial_spearman_shape_x_energy"] = {
            f"{a}__{b}": partial_spearman(shape[a], energy[b], ctrl)
            for a in shape for b in energy}
        out["partial_n_showers"] = int(len(ks))
        # energy-weighted depth reported separately, flagged as the mixed quantity
        out["partial_spearman_eweighted_depth"] = {
            f"{a}__{b}": partial_spearman(v, energy[b], ctrl)
            for a, v in (("depth_mean_ew", dmean), ("depth_std_ew", dstd)) for b in energy}
    return out


# ---------------------------------------------------------------- generation

def generate(ckpt_path, slice_path, pdg_class, sel, dev, steps, batch, no_partition,
             no_econs, geometry):
    """Sample showers from a checkpoint on the SAME conditioning rows the real side used."""
    import torch
    from genpu.flow.calo_flow import CaloFlow

    ck = torch.load(ckpt_path, map_location=dev, weights_only=False)
    sd = ck["model"]; nsrc = ck.get("norm", sd)

    def _get(base):
        for k in (f"cont_{base}", f"cond_{base}"):
            if k in nsrc:
                v = nsrc[k]; return v.cpu().numpy() if hasattr(v, "cpu") else np.asarray(v)
        raise KeyError(base)

    norm = {"cont_mean": _get("mean"), "cont_std": _get("std")}
    for k in ["glob_mean", "glob_std", "pts_mean", "pts_std"]:
        v = nsrc[k]; norm[k] = v.cpu().numpy() if hasattr(v, "cpu") else np.asarray(v)
    model, _, _ = CaloFlow.from_checkpoint(sd, norm)
    model = model.to(dev).eval()

    d = np.load(slice_path)
    cont = d["cont"].astype(np.float32)[sel]
    pdg = d["pdg"].astype(np.int64)[sel]
    E_true = d["E_true"].astype(np.float32)[sel]
    anc = d["anchor"].astype(np.float32)[sel]
    amode = d["anchor_mode"].astype(np.int64)[sel]

    contS = torch.as_tensor((cont - norm["cont_mean"]) / norm["cont_std"], device=dev)
    pdgT = torch.as_tensor(pdg, device=dev)
    E_trueT = torch.as_tensor(E_true, device=dev)
    anchorT = torch.as_tensor(anc, device=dev) if bool(model.core_anchored > 0) else None
    modeT = torch.as_tensor(amode, device=dev) if model.anchor_cond else None

    PE, PP, PD, PL, NN = [], [], [], [], []
    for s in range(0, len(sel), batch):
        e = min(s + batch, len(sel))
        with torch.no_grad():
            sh = model.sample_showers(contS[s:e], pdgT[s:e], steps=steps,
                                      partition=not no_partition,
                                      e_true=None if no_econs else E_trueT[s:e],
                                      core_anchor=None if anchorT is None else anchorT[s:e],
                                      anchor_mode=None if modeT is None else modeT[s:e])
        pos = sh["pos"].cpu().numpy()
        PE.append(pos[:, 0]); PP.append(pos[:, 1])
        if pos.shape[1] < 3:
            raise SystemExit("checkpoint is 2-D (pos_dim<3): no depth to correlate against. "
                             "This test needs a v2 3-D model.")
        PD.append(pos[:, 2])
        PL.append(sh["logE"].cpu().numpy())
        NN.append(sh["n"].cpu().numpy())
    n_per = np.concatenate(NN).astype(np.int64)
    return (np.concatenate(PE).astype(np.float64), np.concatenate(PP).astype(np.float64),
            np.concatenate(PD).astype(np.float64), np.concatenate(PL).astype(np.float64),
            np.concatenate([[0], np.cumsum(n_per)]).astype(np.int64), n_per)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_slice", required=True, help="a v2 calo slice npz (points_flat is (P,4))")
    ap.add_argument("--pdg_class", type=int, nargs="+", default=None,
                    help="restrict to these classes; default = every class present, plus a "
                         "per-class breakdown")
    ap.add_argument("--ckpt", default=None, help="optional: generated CONTROL, expected ~0 on part C")
    ap.add_argument("--max_showers", type=int, default=400000,
                    help="subsample cap PER ANALYSED GROUP (the slices run to 5M showers / 125M cells)")
    ap.add_argument("--n_min", type=int, default=5, help="min cells for the shower-level part C")
    ap.add_argument("--nbin", type=int, default=10)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--batch", type=int, default=200000)
    ap.add_argument("--no_partition", action="store_true")
    ap.add_argument("--no_econs", action="store_true")
    ap.add_argument("--geometry", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--outdir", default="/home/lv7805/genpu/plots/calo/metrics")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    t0 = time.time()

    d = np.load(a.real_slice)
    pts, off = d["points_flat"], d["offsets"].astype(np.int64)
    pdg = d["pdg"].astype(np.int64)
    if pts.shape[1] < 4:
        raise SystemExit(f"{a.real_slice} is a v1 slice (points_flat {pts.shape}); no depth column.")
    n_per_all = np.diff(off)
    print(f"slice {Path(a.real_slice).name}: {len(pdg):,} showers, {len(pts):,} cells, "
          f"classes {sorted(set(pdg.tolist()))}", flush=True)

    classes = sorted(set(pdg.tolist())) if a.pdg_class is None else list(a.pdg_class)
    groups = [("all", np.arange(len(pdg)))] if a.pdg_class is None else []
    if a.pdg_class is not None:
        groups.append((f"pdg{'_'.join(map(str, a.pdg_class))}",
                       np.where(np.isin(pdg, a.pdg_class))[0]))
    for c in classes:
        idx = np.where(pdg == c)[0]
        if len(idx) >= 1000:
            groups.append((f"pdg{c}", idx))

    res = {"slice": str(a.real_slice), "ckpt": a.ckpt, "n_min": a.n_min, "nbin": a.nbin,
           "max_showers": a.max_showers, "seed": a.seed, "groups": {}}

    for name, idx in groups:
        if len(idx) > a.max_showers:
            idx = np.sort(rng.choice(idx, a.max_showers, replace=False))
        n_per = n_per_all[idx]
        ci = (np.repeat(off[idx], n_per)
              + (np.arange(n_per.sum()) - np.repeat(np.cumsum(n_per) - n_per, n_per)))
        p = pts[ci]
        offg = np.concatenate([[0], np.cumsum(n_per)]).astype(np.int64)
        print(f"\n=== {name}: {len(idx):,} showers / {len(p):,} cells "
              f"({time.time()-t0:.0f}s)", flush=True)
        pe, pp, pd, pl = (p[:, 0].astype(np.float64), p[:, 1].astype(np.float64),
                          p[:, 2].astype(np.float64), p[:, 3].astype(np.float64))
        g = {"real": analyse("real", pe, pp, pd, pl, offg, n_per, a.n_min, rng, a.nbin)}
        # NULL: real cells, but each shower's shape paired with another shower's energies.
        # This is exactly the conditional independence the architecture imposes, so it calibrates
        # every number above against "what the current model could produce at best".
        g["null"] = analyse("null", pe, pp, pd,
                            shuffle_energies_matched_n(pl, n_per, rng),
                            offg, n_per, a.n_min, rng, a.nbin)
        if a.ckpt:
            import torch
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            ge, gp, gd, gl, goff, gn = generate(a.ckpt, a.real_slice, a.pdg_class, idx, dev,
                                                a.steps, a.batch, a.no_partition, a.no_econs,
                                                a.geometry)
            g["gen"] = analyse("gen", ge, gp, gd, gl, goff, gn, a.n_min, rng, a.nbin)
        res["groups"][name] = g

        # --- console summary, the numbers the decision turns on
        for side in ("real", "null", "gen"):
            if side not in g:
                continue
            s = g[side]
            print(f"  [{side}] within-shower rho(logE, r)     = "
                  f"{s['within_rho_logE_r']['mean']:+.4f} ± {s['within_rho_logE_r']['sem']:.4f}")
            print(f"  [{side}] within-shower rho(logE, depth) = "
                  f"{s['within_rho_logE_depth']['mean']:+.4f} ± {s['within_rho_logE_depth']['sem']:.4f}")
            if "profile_vs_r" in s:
                pf = s["profile_vs_r"]["p_floor_gate"]
                print(f"  [{side}] p(floor) core->fringe: "
                      f"{pf[0]:.3f} -> {pf[-1]:.3f}  (x{pf[-1]/max(pf[0],1e-9):.2f})")
                pfd = s["profile_vs_depth"]["p_floor_gate"]
                print(f"  [{side}] p(floor) shallow->deep: "
                      f"{pfd[0]:.3f} -> {pfd[-1]:.3f}  (x{pfd[-1]/max(pfd[0],1e-9):.2f})")
            if "partial_spearman_shape_x_energy" in s:
                ps = s["partial_spearman_shape_x_energy"]
                top = sorted(ps.items(), key=lambda kv: -abs(kv[1]))[:5]
                print(f"  [{side}] partial rho(shape, energy | log_n, log_totE), top 5:")
                for k, v in top:
                    print(f"            {k:34s} {v:+.4f}")

    outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)
    jp = outdir / f"epos_coupling_{a.tag}.json"
    jp.write_text(json.dumps(res, indent=1))
    print(f"\nwrote {jp}  ({time.time()-t0:.0f}s)")

    # --- plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        gn = [k for k in res["groups"] if "profile_vs_r" in res["groups"][k]["real"]]
        gn = gn[:6]
        fig, ax = plt.subplots(2, 2, figsize=(11, 8))
        x = (np.arange(a.nbin) + 0.5) / a.nbin
        for k in gn:
            for j, (xn, yn, lab) in enumerate([
                    ("profile_vs_r", "p_floor_gate", "p(floor) vs within-shower rank of r"),
                    ("profile_vs_r", "mean_logE", "<logE> vs within-shower rank of r"),
                    ("profile_vs_depth", "p_floor_gate", "p(floor) vs within-shower rank of depth"),
                    ("profile_vs_depth", "mean_logE", "<logE> vs within-shower rank of depth")]):
                A = ax[j // 2, j % 2]
                A.plot(x, res["groups"][k]["real"][xn][yn], "-o", ms=3, label=f"{k} real")
                if "gen" in res["groups"][k] and xn in res["groups"][k]["gen"]:
                    A.plot(x, res["groups"][k]["gen"][xn][yn], "--s", ms=3, label=f"{k} gen")
                A.set_title(lab, fontsize=9); A.set_xlabel("core → fringe / shallow → deep")
                A.grid(alpha=.3)
        ax[0, 0].legend(fontsize=7)
        fig.suptitle(f"calo cell energy vs position within shower — {a.tag}")
        fig.tight_layout()
        pp = outdir / f"epos_coupling_{a.tag}.png"
        fig.savefig(pp, dpi=130)
        print(f"wrote {pp}")
    except Exception as ex:  # plotting must never lose the numbers
        print(f"plot skipped: {ex}")


if __name__ == "__main__":
    main()
