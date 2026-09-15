"""Are a shower's cells spatially COHERENT, or just i.i.d. draws from a profile?

2026-08-27 measured the ONE-POINT function -- energy as a function of a cell's own position --
and found real showers carry a gradient (rho(logE, depth) up to +0.54 for protons) that the model
produces none of. Conditioning `EnergyHead` on the cell's own position would fix that. It would NOT
fix the TWO-POINT function: with per-cell conditioning, two cells at the same radius still get
i.i.d. draws, so the model reproduces the average falloff with a salt-and-pepper realisation where
a real shower has a contiguous hot core.

That distinction decides how big the fix has to be:
  one-point wrong  -> concat position into the existing head (cheap)
  two-point wrong  -> the cells must see EACH OTHER (attention / set model), and ultimately the
                      cell-grid projection of PIPELINE.md gap #1, since "neighbouring cell" is not
                      even well defined for a continuous point cloud.

WHAT IS MEASURED (all within-shower, so no shower-level scale enters)

  A. xi(separation)  Two-point energy correlation. logE is standardised WITHIN each shower to z,
                     then random cell pairs are binned by separation. E[z_i z_j] vs separation.
                     Real: should DECAY with separation. The model: flat, because given the
                     conditioning its cells are i.i.d.
  B. hot-core        Mean z versus the cell's within-shower rank of DISTANCE TO THE HOTTEST CELL.
                     This is the plain-language question -- "does the max-energy cell have high
                     energy cells around it" -- asked directly.
  C. centroid pull   |energy-weighted centroid - unweighted centroid|, in units of shower width.
                     If energy is spatially coherent the weighted centroid is dragged toward the
                     hot cluster; if energy is scattered at random it is not. This is the bridge to
                     the DISPERSION statistics: core and width are energy-weighted, so coherence
                     biases them, and unmodelled coherence is a candidate for the pion's residual
                     core under-dispersion (0.93, 2026-08-14) and for generated width being MORE
                     dispersed than real and growing with n (2026-08-16).

THE NULL is a permutation of logE WITHIN each shower. It keeps every per-shower energy statistic
exactly intact (the multiset is untouched) and destroys only the energy<->position pairing, which
is precisely what is under test. Note this is the OPPOSITE choice from the shower-level part C of
`calo_energy_position_coupling.py`, where a within-shower permutation was vacuous; here it is the
correct null and the cross-shower one would be wrong. It also fixes the E[z_i z_j] = -1/(n-1)
baseline that within-shower standardisation imposes, so xi should be read against the null, never
against zero.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def within_rank(vals, off, n_per, rng, jitter=0.0):
    v = vals + (rng.normal(0, jitter, len(vals)) if jitter else 0.0)
    src = np.repeat(np.arange(len(n_per)), n_per)
    order = np.lexsort((v, src))
    r = np.empty(len(v), np.float64)
    r[order] = np.arange(len(v), dtype=np.float64) - off[src[order]]
    out = r / np.maximum(n_per - 1, 1)[src]
    out[np.repeat(n_per == 1, n_per)] = 0.5
    return out


def standardise_within(le, src, S, n_per):
    m = np.bincount(src, weights=le, minlength=S) / n_per
    v = np.bincount(src, weights=(le - m[src]) ** 2, minlength=S) / n_per
    s = np.sqrt(np.maximum(v, 1e-12))
    return (le - m[src]) / s[src]


def residualise(z, rk_r, rk_d, nb=12):
    """Subtract the ONE-POINT profile E[z | own radius rank, own depth rank].

    REQUIRED, and the synthetic check is why: an arm whose energy depends only on each cell's OWN
    radius -- i.e. exactly what per-cell position conditioning would produce -- reproduces
    xi(separation) almost perfectly (+0.60 -> -0.47 vs the coherent arm's +0.57 -> -0.39). Two cells
    that are close together have similar radii, so a radial gradient ALONE manufactures a
    separation-dependent correlation. Raw xi therefore cannot tell "hot cells cluster" from "energy
    falls with radius", which is the whole question.

    After this subtraction xi' answers the decision directly: xi' ~ null means the cheap per-cell
    position conditioning is sufficient; xi' > null means the cells must see each other.
    """
    br = np.clip((rk_r * nb).astype(np.int64), 0, nb - 1)
    bd = np.clip((rk_d * nb).astype(np.int64), 0, nb - 1)
    b = br * nb + bd
    cnt = np.bincount(b, minlength=nb * nb).astype(np.float64)
    m = np.bincount(b, weights=z, minlength=nb * nb) / np.maximum(cnt, 1)
    return z - m[b]


def shuffle_within_profile(v, rk_r, rk_d, rng, nb=12):
    """Permute values among ALL cells sharing a (radius-rank, depth-rank) bin, ACROSS showers.

    THE DECISIVE NULL: it preserves E[z | own position] exactly while destroying every
    within-shower correlation. That is a direct simulation of the proposed cheap fix -- an
    `EnergyHead` conditioned on each cell's own position and nothing else. So

        real xi_res ~ this null   ->  per-cell position conditioning SUFFICES
        real xi_res >  this null  ->  the cells must see each other (set model / attention)

    MEASURED ESTIMATOR FLOOR (synthetic, 2026-08-27). A one-point-only arm does not land exactly on
    this null: the residualisation bins RANKS, while that arm's energy depends on exact radius, so
    cells sharing a rank bin still share some true radius. Calibration from the two synthetic arms,
    first bin of xi_res: coherent blob **+0.365**, one-point-only **+0.064**, null +0.001. So read
    real data against ~0.06, not against 0 -- anything at or below ~0.06 is consistent with "a
    perfect per-cell position conditioning would reproduce it".

    A plain within-shower permutation cannot serve here: it destroys the one-point profile too, so
    it under-states the reference. The synthetic arms show why the distinction matters -- a
    radial-profile-only arm still leaves ~0.12 of residual xi against the within-shower null purely
    from imperfect binned profile removal, and against THIS null it lands at zero.
    """
    b = (np.clip((rk_r * nb).astype(np.int64), 0, nb - 1) * nb
         + np.clip((rk_d * nb).astype(np.int64), 0, nb - 1))
    n = len(v)
    dst = np.lexsort((np.arange(n), b))
    srcp = np.lexsort((rng.random(n), b))
    out = np.empty_like(v)
    out[dst] = v[srcp]
    return out


def permute_within(le, off, n_per, rng):
    """Shuffle each shower's energies among ITS OWN cells: multiset intact, pairing destroyed."""
    src = np.repeat(np.arange(len(n_per)), n_per)
    order = np.lexsort((rng.random(len(le)), src))
    return le[order]


def analyse(tag, qe, qp, dep, le, off, n_per, rng, nbin, pairs_per_shower, min_n):
    S = len(n_per)
    src = np.repeat(np.arange(S), n_per)
    # re-centre on the realised centroid (generated clouds are offsets from a SEPARATELY sampled
    # core, so their raw coordinates are not centred the way a real slice's are)
    ce = np.bincount(src, weights=qe, minlength=S) / n_per
    cp = np.bincount(src, weights=qp, minlength=S) / n_per
    de, dp = qe - ce[src], qp - cp[src]
    width = np.sqrt(np.bincount(src, weights=de ** 2 + dp ** 2, minlength=S) / n_per)
    z = standardise_within(le, src, S, n_per)
    # remove each side's OWN one-point profile, so what remains is structure BEYOND it
    rk_r = within_rank(np.hypot(de, dp), off, n_per, rng)
    rk_d = within_rank(dep, off, n_per, rng)
    zres = residualise(z, rk_r, rk_d)
    zres_np = residualise(shuffle_within_profile(z, rk_r, rk_d, rng), rk_r, rk_d)
    out = {"tag": tag, "showers": int(S), "cells": int(len(le))}

    big = np.where(n_per >= min_n)[0]
    out["showers_used"] = int(len(big))
    if len(big) < 100:
        return out

    # ---------- A. xi(separation) from random within-shower pairs ----------
    K = pairs_per_shower
    sh = np.repeat(big, K)
    nn = n_per[sh].astype(np.int64)
    i = off[sh] + (rng.random(len(sh)) * nn).astype(np.int64)
    j = off[sh] + (rng.random(len(sh)) * nn).astype(np.int64)
    ok = i != j
    i, j, sh = i[ok], j[ok], sh[ok]
    d_perp = np.hypot(de[i] - de[j], dp[i] - dp[j]) / np.maximum(width[sh], 1e-12)
    d_dep = np.abs(dep[i] - dep[j])
    for nm, dd, hi in (("perp_over_width", d_perp, 3.0), ("depth_mm", d_dep, None)):
        edges = (np.linspace(0, hi, nbin + 1) if hi else
                 np.quantile(dd, np.linspace(0, 1, nbin + 1)))
        b = np.clip(np.digitize(dd, edges) - 1, 0, nbin - 1)
        cnt = np.bincount(b, minlength=nbin).astype(np.float64)
        rec = {"edges": np.asarray(edges, float).tolist(),
               "n_pairs": cnt.astype(np.int64).tolist()}
        for key, zv in (("xi", z), ("xi_res", zres), ("xi_res_onepoint_null", zres_np)):
            rec[key] = (np.bincount(b, weights=zv[i] * zv[j], minlength=nbin)
                        / np.maximum(cnt, 1)).tolist()
        out[f"xi_vs_{nm}"] = rec

    # ---------- B. mean z vs rank of distance to the HOTTEST cell ----------
    cellmask = np.repeat(np.isin(np.arange(S), big), n_per)
    offb = np.concatenate([[0], np.cumsum(n_per[big])]).astype(np.int64)
    nb = n_per[big]
    srcb = np.repeat(np.arange(len(big)), nb)
    zb, deb, dpb, depb = z[cellmask], de[cellmask], dp[cellmask], dep[cellmask]
    zrb = zres[cellmask]
    # index of the max-logE cell of each shower: lexsort puts it last in its block
    ordb = np.lexsort((zb, srcb))
    hot = ordb[np.cumsum(nb) - 1]                       # one global index per shower
    dhot = np.hypot(deb - deb[hot][srcb], dpb - dpb[hot][srcb])
    ddep = np.abs(depb - depb[hot][srcb])
    for nm, dd in (("dist_to_hot", dhot), ("depthdist_to_hot", ddep)):
        rk = within_rank(dd, offb, nb, rng)
        b = np.clip((rk * nbin).astype(np.int64), 0, nbin - 1)
        cnt = np.bincount(b, minlength=nbin).astype(np.float64)
        out[f"meanz_vs_{nm}"] = {
            "mean_z": (np.bincount(b, weights=zb, minlength=nbin) / np.maximum(cnt, 1)).tolist(),
            "mean_z_res": (np.bincount(b, weights=zrb, minlength=nbin) / np.maximum(cnt, 1)).tolist(),
            "n": cnt.astype(np.int64).tolist()}

    # ---------- C. centroid pull ----------
    e = np.exp(le)
    tot = np.bincount(src, weights=e, minlength=S)
    w = e / np.maximum(tot[src], 1e-30)
    ew_e = np.bincount(src, weights=w * de, minlength=S)
    ew_p = np.bincount(src, weights=w * dp, minlength=S)
    pull = np.hypot(ew_e, ew_p) / np.maximum(width, 1e-12)      # unweighted centroid is 0 by
    out["centroid_pull"] = {"mean": float(pull[big].mean()),    # construction after re-centring
                            "median": float(np.median(pull[big])),
                            "sem": float(pull[big].std() / np.sqrt(len(big)))}
    ew_d = np.bincount(src, weights=w * dep, minlength=S)
    un_d = np.bincount(src, weights=dep, minlength=S) / n_per
    sd = np.sqrt(np.maximum(np.bincount(src, weights=(dep - un_d[src]) ** 2, minlength=S) / n_per, 1e-12))
    out["depth_pull"] = {"mean": float(((ew_d - un_d) / sd)[big].mean()),
                         "sem": float(((ew_d - un_d) / sd)[big].std() / np.sqrt(len(big)))}
    return out


def generate(ckpt, slice_path, sel, dev, steps, batch, no_partition, no_econs):
    import torch
    from genpu.flow.calo_flow import CaloFlow
    ck = torch.load(ckpt, map_location=dev, weights_only=False)
    sd = ck["model"]; ns = ck.get("norm", sd)

    def _get(b):
        for k in (f"cont_{b}", f"cond_{b}"):
            if k in ns:
                v = ns[k]; return v.cpu().numpy() if hasattr(v, "cpu") else np.asarray(v)
        raise KeyError(b)
    norm = {"cont_mean": _get("mean"), "cont_std": _get("std")}
    for k in ["glob_mean", "glob_std", "pts_mean", "pts_std"]:
        v = ns[k]; norm[k] = v.cpu().numpy() if hasattr(v, "cpu") else np.asarray(v)
    model, _, _ = CaloFlow.from_checkpoint(sd, norm); model = model.to(dev).eval()
    d = np.load(slice_path)
    cont = d["cont"].astype(np.float32)[sel]; pdg = d["pdg"].astype(np.int64)[sel]
    contS = torch.as_tensor((cont - norm["cont_mean"]) / norm["cont_std"], device=dev)
    pdgT = torch.as_tensor(pdg, device=dev)
    E_trueT = torch.as_tensor(d["E_true"].astype(np.float32)[sel], device=dev)
    aT = torch.as_tensor(d["anchor"].astype(np.float32)[sel], device=dev) if bool(model.core_anchored > 0) else None
    mT = torch.as_tensor(d["anchor_mode"].astype(np.int64)[sel], device=dev) if model.anchor_cond else None
    A, B, C, L, N = [], [], [], [], []
    for s in range(0, len(sel), batch):
        e = min(s + batch, len(sel))
        with torch.no_grad():
            sh = model.sample_showers(contS[s:e], pdgT[s:e], steps=steps,
                                      partition=not no_partition,
                                      e_true=None if no_econs else E_trueT[s:e],
                                      core_anchor=None if aT is None else aT[s:e],
                                      anchor_mode=None if mT is None else mT[s:e])
        p = sh["pos"].cpu().numpy()
        if p.shape[1] < 3:
            raise SystemExit("2-D checkpoint: no depth. This test needs a v2 3-D model.")
        A.append(p[:, 0]); B.append(p[:, 1]); C.append(p[:, 2])
        L.append(sh["logE"].cpu().numpy()); N.append(sh["n"].cpu().numpy())
    n = np.concatenate(N).astype(np.int64)
    return (np.concatenate(A).astype(np.float64), np.concatenate(B).astype(np.float64),
            np.concatenate(C).astype(np.float64), np.concatenate(L).astype(np.float64),
            np.concatenate([[0], np.cumsum(n)]).astype(np.int64), n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_slice", required=True)
    ap.add_argument("--pdg_class", type=int, nargs="+", default=None)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--max_showers", type=int, default=300000)
    ap.add_argument("--min_n", type=int, default=6,
                    help="a two-point function needs several cells; showers below this are dropped")
    ap.add_argument("--pairs_per_shower", type=int, default=40)
    ap.add_argument("--nbin", type=int, default=10)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--batch", type=int, default=200000)
    ap.add_argument("--no_partition", action="store_true")
    ap.add_argument("--no_econs", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--outdir", default="/home/lv7805/genpu/plots/calo/metrics")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed); t0 = time.time()

    d = np.load(a.real_slice)
    pts, off = d["points_flat"], d["offsets"].astype(np.int64)
    pdg = d["pdg"].astype(np.int64)
    if pts.shape[1] < 4:
        raise SystemExit(f"{a.real_slice} is a v1 slice; no depth column.")
    n_all = np.diff(off)
    print(f"slice {Path(a.real_slice).name}: {len(pdg):,} showers, {len(pts):,} cells", flush=True)

    groups = []
    if a.pdg_class is not None:
        groups.append((f"pdg{'_'.join(map(str,a.pdg_class))}", np.where(np.isin(pdg, a.pdg_class))[0]))
    else:
        groups.append(("all", np.arange(len(pdg))))
        for c in sorted(set(pdg.tolist())):
            idx = np.where(pdg == c)[0]
            if len(idx) >= 2000:
                groups.append((f"pdg{c}", idx))

    res = {"slice": a.real_slice, "ckpt": a.ckpt, "min_n": a.min_n, "nbin": a.nbin,
           "pairs_per_shower": a.pairs_per_shower, "groups": {}}
    for name, idx in groups:
        if len(idx) > a.max_showers:
            idx = np.sort(rng.choice(idx, a.max_showers, replace=False))
        n_per = n_all[idx]
        ci = (np.repeat(off[idx], n_per)
              + (np.arange(n_per.sum()) - np.repeat(np.cumsum(n_per) - n_per, n_per)))
        p = pts[ci]
        offg = np.concatenate([[0], np.cumsum(n_per)]).astype(np.int64)
        qe, qp, dp3, le = (p[:, 0].astype(np.float64), p[:, 1].astype(np.float64),
                           p[:, 2].astype(np.float64), p[:, 3].astype(np.float64))
        print(f"\n=== {name}: {len(idx):,} showers / {len(p):,} cells ({time.time()-t0:.0f}s)", flush=True)
        g = {"real": analyse("real", qe, qp, dp3, le, offg, n_per, rng, a.nbin,
                             a.pairs_per_shower, a.min_n),
             "null": analyse("null", qe, qp, dp3, permute_within(le, offg, n_per, rng),
                             offg, n_per, rng, a.nbin, a.pairs_per_shower, a.min_n)}
        if a.ckpt:
            import torch
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            ge, gp, gd, gl, goff, gn = generate(a.ckpt, a.real_slice, idx, dev, a.steps,
                                                a.batch, a.no_partition, a.no_econs)
            g["gen"] = analyse("gen", ge, gp, gd, gl, goff, gn, rng, a.nbin,
                               a.pairs_per_shower, a.min_n)
        res["groups"][name] = g
        for side in ("real", "null", "gen"):
            s = g.get(side)
            if not s or "xi_vs_perp_over_width" not in s:
                continue
            print(f"  [{side}] xi RAW  (perp) : "
                  + " ".join(f"{v:+.3f}" for v in s["xi_vs_perp_over_width"]["xi"][:6]))
            print(f"  [{side}] xi RESID(perp) : "
                  + " ".join(f"{v:+.3f}" for v in s["xi_vs_perp_over_width"]["xi_res"][:6])
                  + "   <-- the decision")
            print(f"  [{side}]    1-pt null   : "
                  + " ".join(f"{v:+.3f}" for v in
                             s["xi_vs_perp_over_width"]["xi_res_onepoint_null"][:6]))
            print(f"  [{side}] xi RESID(depth): "
                  + " ".join(f"{v:+.3f}" for v in s["xi_vs_depth_mm"]["xi_res"][:6])
                  + "   1-pt null " + " ".join(f"{v:+.3f}" for v in
                             s["xi_vs_depth_mm"]["xi_res_onepoint_null"][:3]))
            print(f"  [{side}] <z_res> by dist-to-hottest: "
                  + " ".join(f"{v:+.3f}" for v in s["meanz_vs_dist_to_hot"]["mean_z_res"][:6]))
            print(f"  [{side}] centroid pull {s['centroid_pull']['mean']:.4f}"
                  f" ± {s['centroid_pull']['sem']:.4f}   depth pull "
                  f"{s['depth_pull']['mean']:+.4f} ± {s['depth_pull']['sem']:.4f}")

    outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)
    jp = outdir / f"coherence_{a.tag}.json"
    jp.write_text(json.dumps(res, indent=1))
    print(f"\nwrote {jp}  ({time.time()-t0:.0f}s)")

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        gn = [k for k in res["groups"] if "xi_vs_perp_over_width" in res["groups"][k]["real"]][:6]
        fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
        for k in gn:
            for side, ls in (("real", "-o"), ("null", ":x"), ("gen", "--s")):
                s = res["groups"][k].get(side)
                if not s or "xi_vs_perp_over_width" not in s:
                    continue
                e = np.asarray(s["xi_vs_perp_over_width"]["edges"])
                ax[0].plot(0.5 * (e[:-1] + e[1:]), s["xi_vs_perp_over_width"]["xi_res"], ls, ms=3,
                           label=f"{k} {side}")
                e2 = np.asarray(s["xi_vs_depth_mm"]["edges"])
                ax[1].plot(0.5 * (e2[:-1] + e2[1:]), s["xi_vs_depth_mm"]["xi_res"], ls, ms=3)
                ax[2].plot(np.linspace(0, 1, len(s["meanz_vs_dist_to_hot"]["mean_z"])),
                           s["meanz_vs_dist_to_hot"]["mean_z_res"], ls, ms=3)
        for A_, t_, x_ in ((ax[0], "xi RESIDUAL vs transverse sep / width", "sep / shower width"),
                           (ax[1], "xi RESIDUAL vs depth separation", "|dz| (mm)"),
                           (ax[2], "<z_res> vs dist-to-hottest rank", "near -> far")):
            A_.set_title(t_, fontsize=9); A_.set_xlabel(x_); A_.grid(alpha=.3); A_.axhline(0, lw=.6, c="k")
        ax[0].legend(fontsize=6)
        fig.suptitle(f"calo shower coherence (two-point) — {a.tag}")
        fig.tight_layout()
        pp = outdir / f"coherence_{a.tag}.png"; fig.savefig(pp, dpi=130)
        print(f"wrote {pp}")
    except Exception as ex:
        print(f"plot skipped: {ex}")


if __name__ == "__main__":
    main()
