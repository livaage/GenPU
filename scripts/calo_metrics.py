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


FEATURE_NAMES = ["n_cells", "log_totE", "logE_mean", "logE_std", "logE_max", "logE_p90",
                 "frac_near_floor", "cells_per_src"]
# LATERAL SHAPE, added 2026-08-13. The 8 features above are all energy/multiplicity, so the gate
# was blind to shower width — the pion's and proton's worst observable (W/sigma 0.32/0.33 vs the
# photon's 0.11) could not move it. Reported as a SEPARATE 10-feature gate so the 8-feature series
# stays comparable with every earlier number.
WIDTH_FEATURE_NAMES = ["width_mean", "width_std"]


DEPTH_FEATURE_NAMES = ["depth_mean", "depth_std"]


def event_features(eta, phi, E, n_src, widths=None, depths=None):
    logE = np.log(E + 1e-12); tot = E.sum()
    p90 = np.percentile(logE, 90) if len(logE) else -30.0
    f = [len(E), np.log(tot + 1e-12), logE.mean(), logE.std(), logE.max(), p90,
         float((logE < np.log(5e-5) + 0.5).mean()), len(E) / max(n_src, 1)]
    if depths is not None:
        # LONGITUDINAL. Until 2026-08-24 the calo model generated no depth at all, so the gate was
        # blind to the dimension that separates species most cleanly (measured mean depth: gamma
        # 113 mm, e± 301, p 335, pi± 470-483) and that drives 0.40-0.54 of intrinsic width.
        # Energy-weighted so it is the shower's centre of gravity, not a cell count.
        _w = E / (E.sum() + 1e-12)
        _dm = float((_w * depths).sum()) if len(depths) else 0.0
        _ds = float(np.sqrt(max((_w * (depths - _dm) ** 2).sum(), 0.0))) if len(depths) else 0.0
    if widths is not None:
        # per-shower widths of the showers in this event: their mean sets the typical lateral
        # size, their spread carries the shower-to-shower variation an i.i.d. point model loses
        f += [float(np.mean(widths)) if len(widths) else 0.0,
              float(np.std(widths)) if len(widths) else 0.0]
    if depths is not None:
        f += [_dm, _ds]
    return np.array(f, dtype=np.float32)


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
    ap.add_argument("--real_slice", default=None,
                    help="v2 slice to use as the REAL reference instead of raw stage2. Required for "
                         "any re-attributed model: stage2 gives one shower per DIRECT depositor, "
                         "and 64.8%% of those are fragments born inside the calorimeter.")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--snap_cells", action="store_true",
                    help="project generated points onto real ODD cells and merge co-occupants "
                         "(genpu.calo_cells). Validated on real cells: round-trip recovers the same "
                         "cell id 0.9925 of the time, energy conserved to 1e-6, and within one event "
                         "the id map collides on only 0.0004. OFF by default so numbers stay "
                         "comparable to pre-2026-09-15 runs.")
    ap.add_argument("--max_cells", type=int, default=128,
                    help="cap on generated cells per shower. The historical default 128 truncates "
                         "2.10%% of real showers / 6.87%% of all cells on multispecies_v2 and is "
                         "hadron-selective (pbar 21.3%% of cells, mu± 0.00%%). Raise it to measure "
                         "what the cap was costing; the model's per-class n_max_pdg (when the "
                         "checkpoint carries one) still applies on top.")
    ap.add_argument("--batch", type=int, default=200000)
    ap.add_argument("--tag", default="photon")
    ap.add_argument("--logE_max", type=float, default=None,
                    help="override the energy-head sampling clamp [log GeV]. Pre-audit checkpoints "
                         "carry no bound, so pass the training slice's max cell log-E to bound the "
                         "unphysical mixture tail without retraining or touching the marginal.")
    ap.add_argument("--logE_max_slice", default=None,
                    help="npz slice to fit PER-SPECIES cell clamps from (checkpoints trained before "
                         "the per-class bound existed carry no logE_max_pdg buffer)")
    ap.add_argument("--width_renorm", action="store_true",
                    help="width-normalised models only: project each sampled cloud to exactly unit "
                         "RMS before scaling by the sampled width (removes the ~1/sqrt(2n) smearing "
                         "that i.i.d. draws add to the width distribution)")
    ap.add_argument("--no_econs", action="store_true",
                    help="disable the E_reco <= E_true energy-conservation bound (A/B only)")
    ap.add_argument("--no_partition", action="store_true",
                    help="A/B: let the shower total be the SUM of independently drawn cells "
                         "(pre-audit path) instead of renormalising to the sampled global total")
    ap.add_argument("--anchor_kind", choices=["helix", "line", "auto"], default="helix",
                    help="must MATCH the slice the checkpoint was trained on: 'line' forces the "
                         "straight-line anchor for charged particles too (the brem hypothesis)")
    ap.add_argument("--geometry", default=None,
                    help="calo_geometry.json for the helix core anchor (default: repo root). Only "
                         "used when the checkpoint was trained on a --core_anchor helix slice.")
    ap.add_argument("--dump_features", default=None,
                    help="npz to write the PAIRED per-event gate feature matrices to (real, gen, "
                         "event_id, n_src). The composite gate can sit at 0.81 while no single "
                         "feature exceeds 0.58, so the joint structure has to be read separately "
                         "-- scripts/calo_gate_diagnose.py consumes this file.")
    ap.add_argument("--outdir", default="/home/lv7805/genpu/plots/calo/metrics")
    ap.add_argument("--plots", action="store_true",
                    help="write real-vs-generated marginal histograms next to the metrics json. "
                         "Lives HERE, not in eval_calo_flow.py, because this is the path that "
                         "passes the helix anchor to sample_showers -- eval_calo_flow.py never "
                         "learned about it and raises on a core-anchored checkpoint.")
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
    # energy-head conditioning is inferred from the checkpoint's first layer: post-audit
    # checkpoints take (cond_embed + 2 global dims), pre-audit ones cond_embed alone.
    model, missing, unexpected = CaloFlow.from_checkpoint(sd, norm)
    model = model.to(dev)
    eglob = model.energy.glob_dim > 0
    wnorm = bool(model.width_norm > 0)
    unexp = [k for k in unexpected if not k.startswith(("cond_mean", "cond_std"))]
    missk = [k for k in missing if not k.startswith(("cont_mean", "cont_std"))]
    if missk or unexp:
        print(f"WARN load: missing={missk[:6]} unexpected={unexp[:6]}")
    if bool((model.ctx_mask > 0).any()):
        print(f"ctx_norm ON: {model.ctx_loc.shape[0]} contexts, dims {torch.where(model.ctx_mask>0)[0].tolist()}")
    if args.logE_max is not None:
        model.logE_max.fill_(args.logE_max)
    if args.logE_max_slice:
        sd_ = np.load(args.logE_max_slice)
        pts_pdg = np.repeat(sd_["pdg"].astype(np.int64), np.diff(sd_["offsets"]))
        le_ = sd_["points_flat"][:, -1]   # LAST column = energy; index 2 is DEPTH under v2
        for c in np.unique(pts_pdg):
            model.logE_max_pdg[int(c)] = float(le_[pts_pdg == c].max())
        shown = {int(c): round(float(model.logE_max_pdg[int(c)]), 2) for c in np.unique(pts_pdg)}
        print(f"per-class logE_max from {Path(args.logE_max_slice).name}: {shown}")
    print(f"energy head: cond_glob={eglob}  partition={not args.no_partition}  "
          f"logE_max={float(model.logE_max):.3f}")
    model.eval()

    # REAL REFERENCE. --real_slice reads a v2 slice, so shower ATTRIBUTION has exactly one
    # implementation (build_calo_slice_v2.py) shared by training and evaluation -- the same reason
    # calo_geom.py is shared so the training and generation frames cannot drift apart. Without it a
    # v2-trained model would be scored against v1-attributed real showers (one per DIRECT
    # depositor), which is not a subtle mismatch: 64.8% of v1 "showers" are fragments.
    if args.real_slice:
        sd = np.load(args.real_slice)
        cont = sd["cont"].astype(np.float32)
        pdg = sd["pdg"].astype(np.int64)
        p_eta = cont[:, 1].astype(np.float64)
        p_phi = sd["p_phi"].astype(np.float64)
        E_true = sd["E_true"].astype(np.float64)
        eid_sh = sd["event_id"]
        off = sd["offsets"].astype(np.int64)
        pts, glob = sd["points_flat"], sd["glob"]
        anc_sl = sd["anchor"]; amode_sl = sd["anchor_mode"]
        # SPECIES FILTER. A multi-species v2 slice carries every class, so --pdg_class has to select
        # the population being scored -- without this a per-species number would silently be an
        # all-species one. COMPACT the CSR arrays rather than carrying an index set: every stage
        # below (generation batching, the real per-shower loop, the longitudinal profile) assumes
        # `sel` is arange over dense arrays, exactly as the raw-stage2 branch builds it.
        keep = np.where(np.isin(pdg, args.pdg_class))[0]
        if len(keep) < len(pdg):
            cnt = off[keep + 1] - off[keep]
            ci = (np.repeat(off[keep], cnt)
                  + (np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)))
            pts = pts[ci]
            off = np.concatenate([[0], np.cumsum(cnt)]).astype(np.int64)
            cont, pdg, glob = cont[keep], pdg[keep], glob[keep]
            p_eta, p_phi, E_true, eid_sh = p_eta[keep], p_phi[keep], E_true[keep], eid_sh[keep]
            anc_sl, amode_sl = anc_sl[keep], amode_sl[keep]
            print(f"species filter {args.pdg_class}: {len(keep):,} of {len(sd['pdg']):,} showers, "
                  f"{len(pts):,} cells")
        # points are core-relative deltas from (particle + anchor); undo to absolute cell eta/phi
        src = np.repeat(np.arange(len(off) - 1), np.diff(off))
        ch = np.empty((len(pts), 5), np.float32)
        ch[:, CH_ETA] = pts[:, 0] + glob[src, 2] + p_eta[src] + anc_sl[src, 0]
        ch[:, CH_PHI] = pts[:, 1] + glob[src, 3] + p_phi[src] + anc_sl[src, 1]
        ch[:, CH_LOGE] = pts[:, -1]
        r_depth = pts[:, 2].astype(np.float64) if pts.shape[1] >= 4 else None
        sel = np.arange(len(cont))
        print(f"real reference: v2 slice {Path(args.real_slice).name} — {len(sel):,} showers, "
              f"{len(pts):,} cells (reattributed={bool(sd['reattributed'])})")
    else:
        d = np.load(Path(args.preproc_dir) / f"shard_{args.shard:04d}_stage2.npz")
        pf, aux, ch, off, eid_all = (d["particle_features"], d["particle_aux"], d["calo_hits_flat"],
                                     d["calo_offsets"], d["event_ids"])
        ncal = np.diff(off)
        sel = np.where(np.isin(pf[:, PF_PDG], args.pdg_class) & (ncal >= 1))[0]
        vr = np.hypot(aux[sel, AUX_VX], aux[sel, AUX_VY]); logE = np.log(np.clip(aux[sel, AUX_ENERGY], 1e-6, None))
        cont = np.stack([pf[sel, PF_LOGPT], pf[sel, PF_ETA], logE, pf[sel, PF_CHARGE],
                         pf[sel, PF_MASS], vr, aux[sel, AUX_VZ]], 1).astype(np.float32)
        pdg = pf[sel, PF_PDG].astype(np.int64); p_eta, p_phi = pf[sel, PF_ETA], pf[sel, PF_PHI]
        E_true = aux[sel, AUX_ENERGY].astype(np.float64)
        eid_sh = eid_all[sel]
        r_depth = None; anc_sl = amode_sl = None

    contS = torch.as_tensor((cont - norm["cont_mean"]) / norm["cont_std"], device=dev)
    pdgT = torch.as_tensor(pdg, device=dev)
    # CORE ANCHOR (Phase 1). Recomputed here from TRUTH conditioning with the same code the slice
    # builder used, so the generation frame matches the training frame exactly. `phi` and (vx,vy)
    # are not in the `cont` contract (response is phi-invariant, and cont carries only vr), which is
    # precisely why the anchor is computed outside the model and handed in.
    anchorT = modeT = None
    if bool(model.core_anchored > 0):
        from genpu.calo_geom import core_anchor as anchor_of, load_front_face
        R_face, Z_face = load_front_face(args.geometry)
        if anc_sl is not None:
            # A v2 slice already STORES the anchor its showers were built in — computed for the
            # calo-incident ANCESTOR, which is the whole point of re-attribution. Recomputing it
            # here from stage2 particle rows would silently use the wrong particle.
            a_eta, a_phi = anc_sl[:, 0].astype(np.float64), anc_sl[:, 1].astype(np.float64)
            a_mode = amode_sl.astype(np.int8)
        else:
            a_eta, a_phi, a_mode = anchor_of(np.exp(pf[sel, PF_LOGPT]), p_phi, p_eta,
                                             pf[sel, PF_CHARGE], aux[sel, AUX_VX], aux[sel, AUX_VY],
                                             aux[sel, AUX_VZ], R_face, Z_face,
                                             kind=args.anchor_kind)
        modeT = torch.as_tensor(a_mode.astype(np.int64), device=dev)
        anchorT = torch.as_tensor(np.stack([a_eta, a_phi], 1), dtype=torch.float32, device=dev)
        anchor_branches = {lab: round(float((a_mode == c).mean()), 4) for c, lab in
                           [(0, "barrel"), (1, "endcap"), (2, "turning"), (3, "none")]}
        print(f"core anchor ON: {args.anchor_kind} to face (r={R_face:.0f}, |z|={Z_face:.0f}); "
              f"branches {anchor_branches}, |anchor| median {np.median(np.hypot(a_eta, a_phi)):.4f}"
              f"{'; GlobalHead conditioned on the anchor' if model.anchor_cond else ''}")
    # generate (batched) + time it
    gen_eta, gen_phi, gen_E, gen_src, gen_depth = [], [], [], [], []
    n_sat = 0; n_over = 0; n_trunc = 0; n_zs = 0
    E_trueT = torch.as_tensor(E_true, dtype=torch.float32, device=dev)
    torch.cuda.synchronize() if dev == "cuda" else None
    t0 = time.time()
    for s in range(0, len(sel), args.batch):
        e = min(s + args.batch, len(sel))
        with torch.no_grad():
            sh = model.sample_showers(contS[s:e], pdgT[s:e], steps=args.steps,
                                      max_cells=args.max_cells,
                                      partition=not args.no_partition,
                                      e_true=None if args.no_econs else E_trueT[s:e],
                                      width_renorm=args.width_renorm,
                                      core_anchor=None if anchorT is None else anchorT[s:e],
                                      anchor_mode=None if modeT is None else modeT[s:e])
        repn = sh["src"].cpu().numpy()
        pos = sh["pos"].cpu().numpy(); core = sh["core"].cpu().numpy()
        gen_E.append(np.exp(sh["logE"].cpu().numpy()).astype(np.float32))
        # points are DELTAS from the per-shower core, which is itself an offset from the particle
        gen_eta.append(p_eta[s:e][repn] + core[repn, 0] + pos[:, 0])
        gen_phi.append(p_phi[s:e][repn] + core[repn, 1] + pos[:, 1])
        gen_src.append(repn + s)
        if pos.shape[1] >= 3:
            gen_depth.append(pos[:, 2])
        n_sat += int(sh["scale_saturated"].sum()); n_over += int(sh["over_etrue"].sum())
        n_trunc += int(sh["n_trunc"]); n_zs += int(sh.get("n_zero_suppressed", 0))
    torch.cuda.synchronize() if dev == "cuda" else None
    gen_time = time.time() - t0
    print(f"cell-count cap: max_cells={args.max_cells}  truncated {n_trunc:,} / {len(sel):,} showers "
          f"({100 * n_trunc / max(len(sel), 1):.3f}%)  [real >128 rate on multispecies_v2: 2.10%]")
    gen_depth = np.concatenate(gen_depth) if gen_depth else None
    has_depth = (r_depth is not None) and (gen_depth is not None)
    if (r_depth is not None) != (gen_depth is not None):
        print(f"WARNING: depth available on only one side (real={r_depth is not None}, "
              f"gen={gen_depth is not None}) -- depth observables SKIPPED. A 3D model scored "
              f"against a 2D reference, or vice versa, is not a valid comparison.")
    gen_E = np.concatenate(gen_E); gen_eta = np.concatenate(gen_eta); gen_phi = np.concatenate(gen_phi); gen_src = np.concatenate(gen_src)

    # ---- CELL PROJECTION (PIPELINE gap #1). Deliberately a SEPARATE deterministic step after the
    # continuous sample, which is what pileup_generator_plan.md:257 specifies. Off by default so
    # every number stays comparable to the pre-2026-09-15 runs; turn it on to score a generator
    # whose cells actually exist.
    snap_report = None
    if args.snap_cells:
        if gen_depth is None:
            raise SystemExit("--snap_cells needs a 3-D model (the depth column carries the layer)")
        from genpu.calo_cells import snap_and_merge
        n_before = len(gen_E)
        sm = snap_and_merge(gen_eta, gen_phi, gen_depth, gen_E, src=gen_src)
        gen_eta_s = np.arcsinh(sm["z"] / np.maximum(np.hypot(sm["x"], sm["y"]), 1e-9))
        gen_phi_s = np.arctan2(sm["y"], sm["x"])
        e_before = gen_E.sum()
        gen_eta, gen_phi, gen_E, gen_src = gen_eta_s, gen_phi_s, sm["energy"], sm["src"]
        # depth is now a layer index, not the continuous coordinate the real reference carries, so
        # the depth observables must be dropped. `has_depth` was already computed above, hence the
        # explicit reset -- leaving it True with gen_depth None crashes the depth gate.
        gen_depth = None
        has_depth = False
        snap_report = {"cells_before": int(n_before), "cells_after": int(len(gen_E)),
                       "merged_frac": round(1 - len(gen_E) / max(n_before, 1), 5),
                       "energy_ratio": round(float(gen_E.sum() / max(e_before, 1e-12)), 6),
                       "multi_contrib_frac": round(float(np.mean(sm["n_merged"] > 1)), 5)}
        print(f"cell projection: {n_before:,} points -> {len(gen_E):,} cells "
              f"(merged {snap_report['merged_frac']:.4f}), energy x{snap_report['energy_ratio']:.6f}, "
              f"cells with >1 contributor {snap_report['multi_contrib_frac']:.4f}")
        print("  NOTE depth observables are SKIPPED under --snap_cells (the snapped depth is a "
              "layer index, not the continuous coordinate the real reference carries).")

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
    # WRAP the phi difference. Without `--snap_cells`, `gen_phi` is the UNWRAPPED sum
    # p_phi + core + pos, so a bare subtraction is already small and wrap_pi is a no-op. With
    # `--snap_cells`, `gen_phi` is recomputed as arctan2(y, x) and therefore lives in (-pi, pi],
    # so every shower near +-pi picked up a ~2pi difference. That one missing wrap moved
    # `width_std` 0.5016 -> 0.9565, `width_mean` 0.5618 -> 0.9032 and the d_phi Wasserstein
    # 0.0343 -> 0.4025 in job 13928711, which looked exactly like a real physics regression.
    # Wrapping unconditionally is correct for both paths: cell d_phi is always << pi.
    from genpu.calo_geom import wrap_pi as _wrap
    g_de = gen_eta - p_eta[gen_src]; g_dp = _wrap(gen_phi - p_phi[gen_src])
    g_width = np.sqrt(np.bincount(gen_src, weights=g_de**2 + g_dp**2, minlength=len(sel)) / np.maximum(g_n_arr, 1))

    # (0) THE FLOOR EDGE, real vs generated (2026-09-07). `frac_near_floor` in the gate is a wide
    # band (`logE < LF+0.5`, 3.79% of real e± cells) and cannot see WHERE in that band the cells
    # sit. These three separate the mechanisms: `at` is the point-mass artefact (real = 0.000000
    # exactly, over 13.05M held-out e± cells), `sub` is the physical shared-cell tail below the
    # threshold (real 0.387%), `near` is the gate's band. A model that drops the atom should move
    # `at` to ~0 and must be checked for SMEARING, which shows up as `sub` running above real.
    _LF = np.log(5e-5)
    _gl = np.log(gen_E + 1e-12)
    floor_edge = {
        "at_floor_real": round(float(np.mean(np.abs(RE - _LF) < 1e-6)), 6),
        "at_floor_gen": round(float(np.mean(np.abs(_gl - _LF) < 1e-6)), 6),
        "sub_floor_real": round(float(np.mean(RE < _LF)), 5),
        "sub_floor_gen": round(float(np.mean(_gl < _LF)), 5),
        "near_floor_real": round(float(np.mean(RE < _LF + 0.5)), 5),
        "near_floor_gen": round(float(np.mean(_gl < _LF + 0.5)), 5),
    }
    print("floor edge (real vs gen): " + "  ".join(f"{k}={v}" for k, v in floor_edge.items()),
          flush=True)

    # (1) per-observable Wasserstein
    obs = {"cells_per_shower": (r_n, g_n_arr), "cell_logE": (RE, np.log(gen_E + 1e-12)),
           "d_eta": (RP, g_de), "d_phi": (RG, g_dp), "shower_width": (r_width, g_width),
           "logEreco": (np.log(r_Ereco + 1e-12), np.log(g_Ereco + 1e-12))}
    if has_depth:
        r_dsh = np.array([np.average(r_depth[off[i]:off[i + 1]],
                                     weights=np.exp(ch[off[i]:off[i + 1], CH_LOGE]) + 1e-12)
                          for i in sel])
        gw = gen_E + 1e-12
        g_dsh = (np.bincount(gen_src, weights=gen_depth * gw, minlength=len(sel))
                 / np.maximum(np.bincount(gen_src, weights=gw, minlength=len(sel)), 1e-12))
        obs["cell_depth"] = (r_depth, gen_depth)
        obs["shower_depth"] = (r_dsh, g_dsh)
    wass = {k: float(wasserstein_distance(a, b)) for k, (a, b) in obs.items()}
    wnorm = {k: wass[k] / (obs[k][0].std() + 1e-9) for k in obs}

    if args.plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        panels = [("cell_logE", "cell log E [GeV]", True), ("d_eta", "cell d_eta", False),
                  ("d_phi", "cell d_phi", False), ("cell_depth", "cell depth [mm]", False),
                  ("cells_per_shower", "cells / shower", True), ("logEreco", "shower log E [GeV]", False),
                  ("shower_width", "shower width", False), ("shower_depth", "shower depth [mm]", False)]
        panels = [p for p in panels if p[0] in obs]
        rr = np.sqrt(RP ** 2 + RG ** 2); gg = np.sqrt(g_de ** 2 + g_dp ** 2)
        extra = ("radial profile", "sqrt(d_eta^2 + d_phi^2)", rr, gg)
        n = len(panels) + 1
        ncol = 4 if n > 4 else n
        nrow = int(np.ceil(n / ncol))
        fig, axs = plt.subplots(nrow, ncol, figsize=(4.3 * ncol, 3.5 * nrow))
        axs = np.atleast_1d(axs).ravel()
        for ax, item in zip(axs, panels + [extra]):
            if len(item) == 3:
                key, xlabel, logy = item
                a, b = obs[key]
                title = f"{key}   W/sigma {wnorm[key]:.3f}"
            else:
                title, xlabel, a, b = item
                logy = False
            lo, hi = np.percentile(np.concatenate([a, b]), [0.2, 99.8])
            if not np.isfinite([lo, hi]).all() or hi <= lo:
                lo, hi = float(np.min(a)), float(np.max(a)) + 1e-6
            bins = np.linspace(lo, hi, 60)
            ax.hist(a, bins=bins, density=True, histtype="step", lw=2.0, color="#1f6feb", label="real")
            ax.hist(b, bins=bins, density=True, histtype="step", lw=2.0, color="#e8871a", label="generated")
            if logy:
                ax.set_yscale("log")
            ax.set_title(title, fontsize=11)
            ax.set_xlabel(xlabel, fontsize=10)
            ax.legend(fontsize=9)
        for ax in axs[n:]:
            ax.axis("off")
        fig.suptitle(f"calo response, real vs generated — {args.tag}   "
                     f"({len(sel):,} showers, {len(RE):,} real / {len(gen_E):,} generated cells)",
                     fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        Path(args.outdir).mkdir(parents=True, exist_ok=True)
        pp = Path(args.outdir) / f"marginals_{args.tag}.png"
        fig.savefig(pp, dpi=130); plt.close(fig)
        print("wrote", pp, flush=True)

    # (3) conditional energy response: <E_reco/E_true> + resolution vs E_true
    resp_r = r_Ereco / E_true; resp_g = g_Ereco / E_true
    bins = np.quantile(np.log10(E_true), np.linspace(0, 1, 7))
    bi = np.clip(np.digitize(np.log10(E_true), bins[1:-1]), 0, 5)
    response = []
    for bb in range(6):
        m = bi == bb
        if m.sum() < 50:
            continue
        # mean response is tail-dominated (a single unphysical cell moved a 140k-shower bin
        # mean to 4.7), so report the MEDIAN and a robust IQR/median width alongside it —
        # the mean/max pair then reads as the tail diagnostic it is.
        def iqr_med(r):
            q1, q2, q3 = np.percentile(r, [25, 50, 75])
            return float((q3 - q1) / (q2 + 1e-12))
        response.append({"logE_true_med": round(float(np.median(np.log10(E_true[m]))), 2),
                         "n": int(m.sum()),
                         "resp_real": round(float(np.mean(resp_r[m])), 3), "resp_gen": round(float(np.mean(resp_g[m])), 3),
                         "respmed_real": round(float(np.median(resp_r[m])), 5), "respmed_gen": round(float(np.median(resp_g[m])), 5),
                         "reso_real": round(float(np.std(resp_r[m]) / (np.mean(resp_r[m]) + 1e-9)), 3),
                         "reso_gen": round(float(np.std(resp_g[m]) / (np.mean(resp_g[m]) + 1e-9)), 3),
                         "iqrmed_real": round(iqr_med(resp_r[m]), 3), "iqrmed_gen": round(iqr_med(resp_g[m]), 3),
                         "respmax_real": round(float(resp_r[m].max()), 3), "respmax_gen": round(float(resp_g[m].max()), 3)})

    # (2) event gate
    # GROUP BY EVENT ONCE. The per-event loop used to call `np.where(ev_of == ev)` and
    # `np.isin(gen_src, pm)` on every iteration -- O(n_showers x n_events) and O(n_cells x n_events).
    # On the e+- slice that is ~1e11 element ops; on a multi-species slice it does not finish.
    # A stable argsort gives contiguous per-event blocks for both sides instead.
    ev_of = eid_sh
    sh_ord = np.argsort(ev_of, kind="stable")
    uev, ev_first, ev_cnt = np.unique(ev_of[sh_ord], return_index=True, return_counts=True)
    g_ev = ev_of[gen_src]
    g_ord = np.argsort(g_ev, kind="stable")
    g_lo = np.searchsorted(g_ev[g_ord], uev, "left")
    g_hi = np.searchsorted(g_ev[g_ord], uev, "right")
    Xr, Xg, ev_nsrc = [], [], []
    for k in range(len(uev)):
        pm = sh_ord[ev_first[k]:ev_first[k] + ev_cnt[k]]; n_src = len(pm)
        re, rp, rE = [], [], []
        for i in pm:
            a, b = off[sel[i]], off[sel[i] + 1]; re.append(ch[a:b, CH_ETA]); rp.append(ch[a:b, CH_PHI]); rE.append(np.exp(ch[a:b, CH_LOGE]))
        # per-shower widths of THIS event's showers (already computed per shower above), so the
        # width features are built from the same numbers the Wasserstein table reports
        rd = np.concatenate([r_depth[off[sel[i]]:off[sel[i] + 1]] for i in pm]) if has_depth else None
        Xr.append(event_features(np.concatenate(re), np.concatenate(rp), np.concatenate(rE), n_src,
                                 widths=r_width[pm], depths=rd))
        gm = g_ord[g_lo[k]:g_hi[k]]
        Xg.append(event_features(gen_eta[gm], gen_phi[gm], gen_E[gm], n_src, widths=g_width[pm],
                                 depths=gen_depth[gm] if has_depth else None))
        ev_nsrc.append(n_src)
    Xr, Xg = np.array(Xr), np.array(Xg)
    if args.dump_features:
        # PAIRED by construction: row k of Xr and of Xg are the same event, generated from the same
        # truth particle list, so n_src is identical on both sides. The gate throws that pairing
        # away; the diagnostic needs it.
        dfp = Path(args.dump_features); dfp.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dfp, Xr=Xr.astype(np.float32), Xg=Xg.astype(np.float32),
                            event_id=uev.astype(np.int64), n_src=np.asarray(ev_nsrc, np.int32),
                            feature_names=np.array(FEATURE_NAMES + WIDTH_FEATURE_NAMES
                                                   + (DEPTH_FEATURE_NAMES if has_depth else [])),
                            tag=args.tag, pdg_class=np.asarray(args.pdg_class, np.int64))
        print(f"wrote gate features {dfp}  ({Xr.shape[0]} events x {Xr.shape[1]} features)")
    y = np.concatenate([np.zeros(len(Xr)), np.ones(len(Xg))])
    perm = rng.permutation(len(Xr) + len(Xg)); ntr = (len(Xr) + len(Xg)) // 2
    yt = torch.as_tensor(y, dtype=torch.float32, device=dev)

    def gate(ncol):
        """Train the two-sample classifier on the first `ncol` event features -> held-out AUC."""
        X = np.concatenate([Xr[:, :ncol], Xg[:, :ncol]])
        Xs = (X - X.mean(0)) / (X.std(0) + 1e-6)
        Xt = torch.as_tensor(Xs, dtype=torch.float32, device=dev)
        clf = torch.nn.Sequential(torch.nn.Linear(ncol, 64), torch.nn.SiLU(),
                                  torch.nn.Linear(64, 64), torch.nn.SiLU(), torch.nn.Linear(64, 1)).to(dev)
        opt = torch.optim.Adam(clf.parameters(), lr=1e-3); tri = torch.as_tensor(perm[:ntr], device=dev)
        for _ in range(800):
            opt.zero_grad()
            torch.nn.functional.binary_cross_entropy_with_logits(clf(Xt[tri]).squeeze(-1), yt[tri]).backward()
            opt.step()
        with torch.no_grad():
            s = clf(Xt[torch.as_tensor(perm[ntr:], device=dev)]).squeeze(-1).cpu().numpy()
        return float(rank_auc(s, y[perm[ntr:]])), s, y[perm[ntr:]]

    n8 = len(FEATURE_NAMES)
    auc, s8, y8 = gate(n8)                          # the historical 8-feature gate (comparable)
    auc_w, sw, yw = gate(n8 + len(WIDTH_FEATURE_NAMES))   # + lateral shape (strictly more informative)
    # + LONGITUDINAL. The 8-feature gate is blind to lateral shape; both are blind to depth, which
    # the model only started generating on 2026-08-24. Quote this one for anything 3D.
    if has_depth:
        auc_d, sd_, yd_ = gate(n8 + len(WIDTH_FEATURE_NAMES) + len(DEPTH_FEATURE_NAMES))
    else:
        auc_d = sd_ = yd_ = None

    if args.plots:
        # ROC of the two-sample gate itself. The DIAGONAL is the target: a generator the classifier
        # cannot separate from Geant sits on it. Drawn from the same held-out scores the AUC uses.
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        curves = [("8 energy/multiplicity features", s8, y8, auc, "#1f6feb"),
                  ("+ lateral width", sw, yw, auc_w, "#0e9488")]
        if has_depth:
            curves.append(("+ longitudinal depth", sd_, yd_, auc_d, "#c2384a"))
        figr, axr = plt.subplots(figsize=(6.0, 5.6))
        for lab, sc, yy, aa, col in curves:
            o = np.argsort(-sc)
            yo = yy[o]
            tp = np.cumsum(yo); fp = np.cumsum(1.0 - yo)
            tpr = np.concatenate([[0], tp / max(tp[-1], 1)])
            fpr = np.concatenate([[0], fp / max(fp[-1], 1)])
            axr.plot(fpr, tpr, lw=2.2, color=col, label=f"{lab}   AUC {aa:.3f}")
        axr.plot([0, 1], [0, 1], "k--", lw=1.4, label="indistinguishable (AUC 0.5)")
        axr.set_xlabel("false positive rate (real called generated)")
        axr.set_ylabel("true positive rate (generated called generated)")
        axr.set_title(f"two-sample event gate — {args.tag}\ncloser to the diagonal is better")
        axr.legend(fontsize=9, loc="lower right"); axr.grid(alpha=0.3)
        figr.tight_layout()
        Path(args.outdir).mkdir(parents=True, exist_ok=True)
        rp = Path(args.outdir) / f"roc_{args.tag}.png"
        figr.savefig(rp, dpi=130); plt.close(figr)
        print("wrote", rp, flush=True)

    # LONGITUDINAL PROFILE — energy fraction by depth bin, real vs gen. `pileup_generator_plan.md:356`
    # names "layer-wise energy fractions, shower width/depth profiles" as acceptance criteria; they
    # were never computable because preprocessing dropped the depth coordinate.
    prof = None
    if has_depth:
        edges = np.array([-20, 0, 25, 50, 100, 200, 400, 800, 1e9])
        rw = np.exp(ch[:, CH_LOGE]).astype(np.float64)
        rh = np.histogram(r_depth, bins=edges, weights=rw)[0]; rh = rh / max(rh.sum(), 1e-12)
        gh = np.histogram(gen_depth, bins=edges, weights=gen_E.astype(np.float64))[0]
        gh = gh / max(gh.sum(), 1e-12)
        prof = {f"{int(lo)}-{'inf' if hi > 1e8 else int(hi)}": [round(float(r), 4), round(float(gv), 4)]
                for lo, hi, r, gv in zip(edges[:-1], edges[1:], rh, gh)}
        print("\nlongitudinal profile — energy fraction by depth [mm from front face]")
        print(f"  {'bin':>12} {'real':>8} {'gen':>8} {'ratio':>8}")
        for k, (r, gv) in prof.items():
            print(f"  {k:>12} {r:>8.4f} {gv:>8.4f} {gv/max(r,1e-9):>8.3f}")
        print(f"  energy-weighted mean depth: real {np.average(r_depth, weights=rw):.1f} mm  "
              f"gen {np.average(gen_depth, weights=gen_E.astype(np.float64)):.1f} mm")

    # WHICH event feature carries the gate? A single-variable AUC per feature (sign-agnostic)
    # says what to fix next; the composite AUC alone never does.
    per_feat = {}
    for j, nm in enumerate(FEATURE_NAMES + WIDTH_FEATURE_NAMES
                           + (DEPTH_FEATURE_NAMES if has_depth else [])):
        a = rank_auc(np.concatenate([Xr[:, j], Xg[:, j]]), y)
        per_feat[nm] = round(float(max(a, 1 - a)), 4)

    speed = {"particles": int(len(sel)), "gen_seconds": round(gen_time, 2), "ode_steps": args.steps,
             "particles_per_sec": round(len(sel) / gen_time, 1), "us_per_particle": round(1e6 * gen_time / len(sel), 1)}
    out = {"tag": args.tag, "pdg_class": args.pdg_class, "shard_heldout": args.shard, "n_particles": int(len(sel)),
           "energy_cond_glob": bool(eglob), "partition_energies": bool(not args.no_partition),
           "logE_max": float(model.logE_max), "scale_saturated_frac": round(n_sat / max(len(sel), 1), 5),
           "energy_conservation": bool(not args.no_econs),
           "width_norm": bool(wnorm), "width_renorm": bool(args.width_renorm),
           "core_anchored": bool(model.core_anchored > 0),
           "anchor_kind": args.anchor_kind if anchorT is not None else None,
           "anchor_cond": bool(model.anchor_cond),
           "anchor_branch_frac": (anchor_branches if anchorT is not None else None),
           "over_etrue_frac_gen": round(n_over / max(len(sel), 1), 6),
           # CELL-COUNT TRUNCATION. Non-zero means the cap bound real showers, and because
           # `partition` rescales the survivors to the sampled total those showers keep their
           # energy in too few, too-hot cells -- so this contaminates every per-cell energy
           # feature, not just n_cells. Real reference (multispecies_v2_h5): 2.10% of showers.
           "max_cells": int(args.max_cells),
           "cell_projection": snap_report,
           "n_trunc_frac_gen": round(n_trunc / max(len(sel), 1), 6),
           # joint-energy models drop sub-threshold cells (zero-suppression); report the count so a
           # generator that "matches" by discarding half its cells cannot pass unnoticed
           "n_zero_suppressed_gen": int(n_zs),
           "over_etrue_frac_real": round(float((r_Ereco > E_true).mean()), 6),
           "event_gate_auc": round(float(auc), 4),           # 8 energy/multiplicity features (historical)
           "event_gate_auc_width": round(float(auc_w), 4),   # + per-shower width mean/std
           "event_gate_auc_depth": (round(float(auc_d), 4) if auc_d is not None else None),
           "longitudinal_profile": prof,   # {depth bin: [real, gen]} energy fractions
           "gate_auc_per_feature": per_feat,
           "wasserstein_over_std": {k: round(v, 4) for k, v in wnorm.items()},
           "floor_edge": floor_edge,
           "energy_response": response, "speed": speed}
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"metrics_{args.tag}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2)); print("wrote", outdir / f"metrics_{args.tag}.json")


if __name__ == "__main__":
    main()
