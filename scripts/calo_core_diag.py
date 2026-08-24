"""What normalisation does the shower CORE need?

The core (shower centroid offset from the particle direction) carries 83-93% of the "shower width"
variance the gate reacts to, and it is set by magnetic bending. Its MARGINAL is already matched by
construction (the GlobalHead quantile-normalises dims 2,3), so the defect is CONDITIONAL. This
script asks which normalisation would remove that conditional structure, before anything is retrained:

  (A) location-only   u = core_phi - m(q,pT)                  [subtract the expected bend]
  (B) location+scale  u = (core_phi - m(q,pT)) / s(q,pT)      [also divide by the bend spread]
  (C) conditional quantile — needed only if the residual SHAPE still varies with pT after (B)
      (a low-pT hadron curls and deposits ~uniformly in phi; a stiff one is sharply peaked —
       no location-scale map turns a uniform into a peak)

Diagnosis: bin by charge x pT, report real median/IQR, then measure how bin-DEPENDENT the normalised
residual's shape is. If (B) leaves shape flat across bins, (B) is enough; if not, (C) is required.
Optionally samples a checkpoint to show what the current model actually produces in the same bins.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

CONT_LOGPT, CONT_ETA, CONT_LOGE, CONT_CHARGE, CONT_MASS, CONT_VR, CONT_VZ = range(7)


def wrap_pi(d):
    return (d + np.pi) % (2 * np.pi) - np.pi


def robust_scale(x):
    q1, q3 = np.percentile(x, [25, 75])
    return max((q3 - q1) / 1.349, 1e-9)


def shape_stats(u):
    """Shape descriptors that are invariant under location+scale, so any residual spread across
    bins is exactly the part a location-scale normalisation CANNOT remove."""
    q = np.percentile(u, [5, 10, 25, 50, 75, 90, 95])
    iqr = max(q[4] - q[2], 1e-9)
    return {"p90_p10_over_iqr": float((q[5] - q[1]) / iqr),      # tail weight vs core
            "p95_p05_over_iqr": float((q[6] - q[0]) / iqr),
            "skew_q": float(((q[5] - q[3]) - (q[3] - q[1])) / max(q[5] - q[1], 1e-9))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", required=True)
    ap.add_argument("--ckpt", default=None, help="optional: also sample this model in the same bins")
    ap.add_argument("--tag", default="pion")
    ap.add_argument("--n_bins", type=int, default=6)
    ap.add_argument("--max_showers", type=int, default=600000)
    ap.add_argument("--born", choices=["all", "inside", "outside"], default="all",
                    help="AUDIT 2026-08-24: restrict to showers whose depositing particle was born "
                         "OUTSIDE vs INSIDE the calo front face. 64.8%% of calo depositors are born "
                         "inside (endcap fragments at vr ~ 420 mm but |vz| > 3212 mm) and 91%% of "
                         "those fall to the TURNING-POINT anchor fallback, because a helix cannot be "
                         "extrapolated forward to a face the particle is already behind. The logged "
                         "Phase 1 headline -- e± per-bin core spread 0.585 -> 0.99 -- was measured on "
                         "the pooled set, so it may be a statement about that fallback rather than "
                         "about the helix anchor. Splitting says which.")
    ap.add_argument("--face_r", type=float, default=1259.1966533469083)
    ap.add_argument("--face_z", type=float, default=3212.5)
    ap.add_argument("--gen_showers", type=int, default=40000)
    ap.add_argument("--out", default="/home/lv7805/genpu/plots/calo/metrics")
    args = ap.parse_args()

    d = np.load(args.slice)
    cont, glob, off = d["cont"], d["glob"], d["offsets"]
    npt = np.diff(off).astype(int)
    keep = npt > 1                                   # single-cell showers have core == the cell
    if args.born != "all":
        _in = (cont[:, CONT_VR] >= args.face_r) | (np.abs(cont[:, CONT_VZ]) >= args.face_z)
        keep &= _in if args.born == "inside" else ~_in
        print(f"[born={args.born}] {keep.sum():,} of {(npt > 1).sum():,} multi-cell showers")
    S = min(args.max_showers, int(keep.sum()))
    idx = np.where(keep)[0][:S]
    cont, glob = cont[idx], glob[idx]
    # PHYSICAL core frame. On a --core_anchor helix slice glob dims 2,3 hold the core as a residual
    # from the truth-helix prediction; adding the anchor back puts every table below in the same
    # particle-direction frame an un-anchored slice uses, so the gen/real per-bin spread ratio stays
    # directly comparable with the pre-anchor references (0.58x e±, 0.94x pion).
    anchored = "core_anchor" in d and str(d["core_anchor"]) != "none"
    anchor = d["anchor"][idx] if anchored else np.zeros((len(idx), 2), np.float32)
    ce = glob[:, 2] + anchor[:, 0]
    cp = wrap_pi(glob[:, 3] + anchor[:, 1]) if anchored else glob[:, 3]
    q = np.sign(cont[:, CONT_CHARGE]); logpt = cont[:, CONT_LOGPT]
    print(f"[{args.tag}] {S} multi-cell showers"
          + (f"  |  core_anchor={str(d['core_anchor'])}: tables are in the PHYSICAL core frame "
             f"(residual + anchor); residual sigma(phi) {robust_scale(glob[:,3]):.4f} vs physical "
             f"{robust_scale(cp):.4f}" if anchored else ""))

    edges = np.quantile(logpt, np.linspace(0, 1, args.n_bins + 1))
    bi = np.clip(np.digitize(logpt, edges[1:-1]), 0, args.n_bins - 1)

    print(f"\nREAL core_phi by charge x pT   (m = median, s = IQR/1.349)")
    print(f"{'charge':>7} {'pT bin':>7} {'n':>8} {'m(core_phi)':>12} {'s(core_phi)':>12} "
          f"{'m(core_eta)':>12} {'s(core_eta)':>12}")
    loc, scale = {}, {}
    for sgn, lab in [(-1, "q<0"), (1, "q>0")]:
        for b in range(args.n_bins):
            m = (q == sgn) & (bi == b)
            if m.sum() < 200:
                continue
            loc[(sgn, b)] = float(np.median(cp[m])); scale[(sgn, b)] = float(robust_scale(cp[m]))
            print(f"{lab:>7} {b:>7} {m.sum():>8} {loc[(sgn,b)]:>+12.4f} {scale[(sgn,b)]:>12.4f} "
                  f"{np.median(ce[m]):>+12.4f} {robust_scale(ce[m]):>12.4f}")

    # (B) location+scale normalise, then ask whether the SHAPE is still bin-dependent
    print(f"\nAfter (B) location+scale: shape descriptors per bin (invariant to location & scale,\n"
          f"so any spread here is what (B) CANNOT fix and (C) would be needed for)")
    print(f"{'charge':>7} {'pT bin':>7} {'p90-p10/IQR':>12} {'p95-p05/IQR':>12} {'q-skew':>9}")
    shapes = []
    for sgn, lab in [(-1, "q<0"), (1, "q>0")]:
        for b in range(args.n_bins):
            if (sgn, b) not in loc:
                continue
            m = (q == sgn) & (bi == b)
            u = (cp[m] - loc[(sgn, b)]) / scale[(sgn, b)]
            s = shape_stats(u); shapes.append(s)
            print(f"{lab:>7} {b:>7} {s['p90_p10_over_iqr']:>12.3f} {s['p95_p05_over_iqr']:>12.3f} {s['skew_q']:>9.3f}")
    r = np.array([s["p90_p10_over_iqr"] for s in shapes])
    rr = np.array([s["p95_p05_over_iqr"] for s in shapes])
    print(f"\n  p90-p10/IQR across bins: min {r.min():.3f} max {r.max():.3f} "
          f"-> spread {100*(r.max()-r.min())/r.mean():.1f}% of mean")
    print(f"  p95-p05/IQR across bins: min {rr.min():.3f} max {rr.max():.3f} "
          f"-> spread {100*(rr.max()-rr.min())/rr.mean():.1f}% of mean")
    print(f"  (Gaussian reference: p90-p10/IQR = 2.56, p95-p05/IQR = 2.44; "
          f"uniform = 2.13/2.37 — a curling-vs-stiff change moves this a lot)")
    verdict = ("location+scale (B) SUFFICES — shape is bin-stable"
               if (r.max() - r.min()) / r.mean() < 0.15 else
               "shape is bin-DEPENDENT -> (C) conditional quantile normalisation is required")
    print(f"  VERDICT: {verdict}")

    # how much of the raw conditional structure does (B) remove?
    u_all = np.empty(S); ok = np.zeros(S, bool)
    for (sgn, b), mloc in loc.items():
        m = (q == sgn) & (bi == b)
        u_all[m] = (cp[m] - mloc) / scale[(sgn, b)]; ok |= m
    print(f"\n  raw core_phi : per-bin median spans {min(loc.values()):+.3f}..{max(loc.values()):+.3f}, "
          f"scale spans {min(scale.values()):.3f}..{max(scale.values()):.3f} "
          f"({max(scale.values())/min(scale.values()):.1f}x)")
    print(f"  normalised u : per-bin median ~0 and scale ~1 BY CONSTRUCTION; "
          f"global std {u_all[ok].std():.3f}")

    out = {"tag": args.tag, "showers": int(S), "n_bins": args.n_bins, "core_anchored": bool(anchored),
           "loc": {f"{k[0]}_{k[1]}": v for k, v in loc.items()},
           "scale": {f"{k[0]}_{k[1]}": v for k, v in scale.items()},
           "shape_spread_frac": float((r.max() - r.min()) / r.mean()), "verdict": verdict}

    # optional: what does the CURRENT model produce in these bins?
    if args.ckpt:
        import sys, torch
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
        from genpu.flow.calo_flow import CaloFlow
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        sd = ck["model"]; src = ck.get("norm", sd)
        def _g(base):
            for k in (f"cont_{base}", f"cond_{base}"):
                if k in src:
                    v = src[k]; return v.cpu().numpy() if hasattr(v, "cpu") else np.asarray(v)
        norm = {"cont_mean": _g("mean"), "cont_std": _g("std")}
        for k in ["glob_mean", "glob_std", "pts_mean", "pts_std"]:
            v = src[k]; norm[k] = v.cpu().numpy() if hasattr(v, "cpu") else np.asarray(v)
        model, _, _ = CaloFlow.from_checkpoint(sd, norm)
        model.eval()
        G = min(args.gen_showers, S)
        cs = torch.as_tensor((cont[:G] - norm["cont_mean"]) / norm["cont_std"], dtype=torch.float32)
        pdgT = torch.as_tensor(d["pdg"][idx][:G], dtype=torch.long)
        with torch.no_grad():
            af = (model.anchor_feats(anchor[:G], d["anchor_mode"][idx][:G])
                  if model.anchor_cond else None)
            gs = model.glob.sample(model.cond_embed(cs, pdgT), af)
            # context-normalised checkpoints need the same truth context to un-normalise
            gctx = model.context_of(cs) if bool((model.ctx_mask > 0).any()) else None
            gg = model.unstd_glob(gs, gctx).numpy()
        gg_res = gg.copy()          # what the mixture actually emitted (residual, if anchored)
        if anchored:
            # same add-back as the real side, with each shower's OWN anchor
            gg = gg.copy()
            gg[:, 2] = gg[:, 2] + anchor[:G, 0]
            gg[:, 3] = wrap_pi(gg[:, 3] + anchor[:G, 1])
        print(f"\nGENERATED vs REAL core_phi in the same bins ({G} showers, {Path(args.ckpt).parent.name})")
        print(f"{'charge':>7} {'pT bin':>7} {'m_real':>9} {'m_gen':>9} {'s_real':>9} {'s_gen':>9} {'s_gen/s_real':>12}")
        gen_rows = []
        for sgn, lab in [(-1, "q<0"), (1, "q>0")]:
            for b in range(args.n_bins):
                m = (q[:G] == sgn) & (bi[:G] == b)
                if m.sum() < 200 or (sgn, b) not in loc:
                    continue
                mg, sg = float(np.median(gg[m, 3])), float(robust_scale(gg[m, 3]))
                ratio = sg / scale[(sgn, b)]
                gen_rows.append(ratio)
                print(f"{lab:>7} {b:>7} {loc[(sgn,b)]:>+9.4f} {mg:>+9.4f} {scale[(sgn,b)]:>9.4f} {sg:>9.4f} {ratio:>12.2f}")
        if gen_rows:
            gr = np.array(gen_rows)
            print(f"  gen/real scale ratio: min {gr.min():.2f} max {gr.max():.2f} mean {gr.mean():.2f} "
                  f"(1.00 = the model reproduces the per-bin core spread)")
            out["gen_scale_ratio"] = {"min": float(gr.min()), "max": float(gr.max()), "mean": float(gr.mean())}

        if anchored:
            # The physical frame above is comparable with the pre-anchor references, but it is NOT
            # what the mixture models — the anchor part of it is exact by construction. Repeat in
            # the RESIDUAL frame, and split by anchor branch: the residual scale differs ~10x
            # between a face-reaching shower and a curler, and the GlobalHead conditioning cannot
            # see which branch a shower is in, so it blends them.
            res_r, res_g = glob[:G, 3], gg_res[:, 3]
            print(f"\nRESIDUAL frame (the quantity the mixture actually fits)")
            print(f"{'charge':>7} {'pT bin':>7} {'s_real':>9} {'s_gen':>9} {'ratio':>7}")
            rr = []
            for sgn, lab in [(-1, "q<0"), (1, "q>0")]:
                for b in range(args.n_bins):
                    m = (q[:G] == sgn) & (bi[:G] == b)
                    if m.sum() < 200:
                        continue
                    sr, sg = robust_scale(res_r[m]), robust_scale(res_g[m])
                    rr.append(sg / max(sr, 1e-9))
                    print(f"{lab:>7} {b:>7} {sr:>9.4f} {sg:>9.4f} {sg/max(sr,1e-9):>7.2f}")
            rr = np.array(rr)
            print(f"  residual-frame gen/real spread: min {rr.min():.2f} max {rr.max():.2f} mean {rr.mean():.2f}")
            out["gen_scale_ratio_residual"] = {"min": float(rr.min()), "max": float(rr.max()),
                                               "mean": float(rr.mean())}
            amode_g = d["anchor_mode"][idx][:G]
            print(f"  by anchor branch:")
            per_br = {}
            for c, lab in [(0, "barrel"), (1, "endcap"), (2, "turning")]:
                m = amode_g == c
                if m.sum() < 200:
                    continue
                sr, sg = robust_scale(res_r[m]), robust_scale(res_g[m])
                print(f"    {lab:>8} n={int(m.sum()):>7} s_real {sr:.4f} s_gen {sg:.4f} "
                      f"ratio {sg/max(sr,1e-9):.2f}")
                per_br[lab] = {"n": int(m.sum()), "s_real": float(sr), "s_gen": float(sg),
                               "ratio": float(sg / max(sr, 1e-9))}
            out["residual_by_branch"] = per_br

    o = Path(args.out); o.mkdir(parents=True, exist_ok=True)
    (o / f"core_diag_{args.tag}.json").write_text(json.dumps(out, indent=2))
    print("\nwrote", o / f"core_diag_{args.tag}.json")


if __name__ == "__main__":
    main()
