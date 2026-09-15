"""Calo slice v2 — showers attributed to the CALO-INCIDENT particle, with the longitudinal
coordinate the v1 pipeline never had.

Three corrections to `build_calo_slice.py`, all measured on 2026-08-24:

1. RE-ATTRIBUTION. v1 groups by the DIRECT depositing particle, but 64.8% of calo depositors were
   born INSIDE the calorimeter and carry 54.2% of all cells — they are shower FRAGMENTS of an
   earlier particle, trained as if each were its own shower with its own "incident" kinematics.
   Species ordering tracks calo difficulty exactly: photon 0.1% split (best gate 0.557), e± 74%
   (worst). Here every cell is booked to the first ancestor born OUTSIDE the front face.

2. DEDUPLICATION. `calo_hits_flat` rows are (cell, particle) CONTRIBUTION pairs, not cells — mean
   1.206 contributors per cell, and 7.2% of cells are shared WITHIN one shower. Merging must sum
   contributions per cell, not concatenate rows (naive summing inflates the merged count 1.154x).

3. DEPTH. v1 collapsed cell (x,y,z) -> (eta,phi), so the model is 2D and the plan's acceptance
   metrics were never computable. Depth drives 0.40-0.54 of per-shower INTRINSIC width, so this is
   not merely completeness. Read from the `calo_rz` sidecar.

DEPTH REPRESENTATION — deliberately kept changeable. Endcaps are EXACTLY layered (dets 9/11: 48
layers at 5.050 mm; 12/14: 36 at 51.000 mm; 100% of cells on the grid, ~83% of energy) so a
categorical index is exact there. Barrel EM is NOT recoverable (staves tile a cylinder, so layers
overlap in r; max gap 0.985 mm over 106 mm) and `calo_hits` has no layer id. We therefore store
BOTH per point: a continuous front-face-relative `depth` (always defined) and an integer
`layer` (-1 where undefined). The model picks; changing the choice costs a slice rebuild, never a
data pass — the same property the tracker gets from LAYER_MEANS/LAYER_STDS.

Output npz — v1 keys plus:
  points_flat   (P, 4)   [d_eta, d_phi, depth, e_coord]     (v1 was (P,3))
  point_layer   (P,)     int16, endcap layer index or -1
  n_src         (S,)     int32, direct depositors merged into this shower (1 = nothing merged)

SECTION FIELDS (2026-08-27, job 13033802). The calorimeter is TWO detectors -- ECAL at 5.050 mm
layer pitch (71.5% of energy) and HCAL at 51.000 mm (28.5%) -- and mean cell log-E STEPS by +1.12
(3.06x) across the physical gap between them, because an HCAL cell integrates ~10x more material.
`depth` cannot express that: it is measured from a SINGLE front face, so barrel and endcap overlap
on it while transitioning at DIFFERENT depths (388 vs 435 mm), and a cell at depth 400 mm is barrel
HCAL or endcap dead-gap. `points_flat` is left byte-identical so every existing slice, checkpoint
and metric stays comparable; the new coordinate is added ALONGSIDE for the model to choose, exactly
as depth/layer were:
  point_depth_local (P,) float32, depth from the cell's OWN detector front face
  point_region      (P,) int8, 0 barrel-ECAL / 1 endcap-ECAL / 2 barrel-HCAL / 3 endcap-HCAL
  point_det         (P,) int8, raw detector id (9,10,11,12,13,14)
`point_layer` is now GLOBALLY unique (per-detector base offsets, 168 endcap layers total); it
previously numbered every endcap detector from 0, so layer 20 was EM endcap or hadronic endcap
with no way to tell. Nothing read the field, so the renumbering breaks no result.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from genpu.calo_geom import (load_front_face, load_det_front_faces, core_anchor,
                             det_region, wrap_pi)  # noqa: E402
from genpu.preprocessing import pdg_to_class  # noqa: E402

PF_LOGPT, PF_ETA, PF_PHI, PF_PDG, PF_CHARGE, PF_MASS = range(6)
AUX_PRIMARY, AUX_PARENT, AUX_VX, AUX_VY, AUX_VZ, AUX_ENERGY = range(6)
CH_ETA, CH_PHI, CH_LOGE, CH_FRAC, CH_DET = range(5)
# measured 2026-08-24, job 12884120
ENDCAP_LAYERS = {9: (3202.4, 5.050, 48), 11: (3202.4, 5.050, 48),
                 12: (3647.5, 51.000, 36), 14: (3647.5, 51.000, 36)}
BARREL = {10, 13}


# GLOBAL layer index: per-detector base offsets so an index identifies its detector uniquely.
# Before 2026-08-27 every endcap detector numbered from 0, so layer 20 was EM endcap (det 9/11,
# 48 layers) OR hadronic endcap (det 12/14, 36 layers) with no way to tell -- two detectors whose
# cells differ by 3x in mean energy. Nothing read the field yet, so the renumbering breaks nothing.
LAYER_BASE, _b = {}, 0
for _d in (9, 11, 12, 14):
    LAYER_BASE[_d] = _b
    _b += ENDCAP_LAYERS[_d][2]
N_ENDCAP_LAYERS = _b          # 48+48+36+36 = 168


def depth_and_layer(det, r, z, R0, Z0, det_faces=None):
    """Depth + layer for each cell.

    Returns (depth_global, layer, depth_local, region):
      depth_global  from the SINGLE calo front face -- the pre-2026-08-27 column, kept byte-identical
                    so every existing slice, checkpoint and metric stays comparable.
      layer         endcap layer index, now GLOBALLY unique (see LAYER_BASE); -1 for barrel, where
                    staves tile a cylinder and layers overlap in r (PIPELINE.md gap #2).
      depth_local   depth from the cell's OWN detector front face. This is the coordinate that can
                    express the ECAL/HCAL boundary; `depth_global` cannot, because barrel and endcap
                    overlap on it while transitioning at different depths (388 vs 435 mm).
      region        4-way token, 0 barrel-ECAL / 1 endcap-ECAL / 2 barrel-HCAL / 3 endcap-HCAL.
    """
    det = det.astype(np.int64)
    isb = np.isin(det, list(BARREL))
    depth = np.where(isb, r - R0, np.abs(z) - Z0).astype(np.float32)
    layer = np.full(len(det), -1, np.int16)
    for dd, (z0, step, n) in ENDCAP_LAYERS.items():
        m = det == dd
        if m.any():
            layer[m] = (LAYER_BASE[dd]
                        + np.clip(np.rint((np.abs(z[m]) - z0) / step), 0, n - 1)).astype(np.int16)
    faces = det_faces if det_faces is not None else load_det_front_faces()
    depth_local = np.full(len(det), np.nan, np.float32)
    for dd, (dd_barrel, face) in faces.items():
        m = det == dd
        if m.any():
            depth_local[m] = ((r[m] if dd_barrel else np.abs(z[m])) - face).astype(np.float32)
    return depth, layer, depth_local, det_region(det)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--pdg_class", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--preproc_dir", default="/scratch/gpfs/IOJALVO/lv7805/genpu_data/preprocessed")
    ap.add_argument("--graph_dir", default="/scratch/gpfs/IOJALVO/lv7805/genpu_data/cascade_graph")
    ap.add_argument("--depth_dir", default="/scratch/gpfs/IOJALVO/lv7805/genpu_data/calo_depth")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min_cells", type=int, default=1)
    ap.add_argument("--max_per_class", type=int, default=0)
    ap.add_argument("--core_anchor", default="helix", choices=["none", "helix", "line"])
    ap.add_argument("--no_reattribute", action="store_true",
                    help="v1 behaviour: group by DIRECT depositor. Kept so the two can be A/B'd, "
                         "since depth and re-attribution otherwise land together.")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    R0, Z0 = load_front_face(None)
    print(f"front face barrel r={R0:.1f} endcap |z|={Z0:.1f}   "
          f"reattribute={not a.no_reattribute}  anchor={a.core_anchor}", flush=True)

    C, G, PTS, LAY, OFF, PDG, ANC, AMODE, NSRC = [], [], [], [], [0], [], [], [], []
    DLOC, REG, DET = [], [], []          # per-cell section fields (2026-08-27)
    EVID, PPHI, ETRUE = [], [], []   # metrics need these; cont excludes phi by contract
    for sh in a.shards:
        s2 = np.load(Path(a.preproc_dir) / f"shard_{sh:04d}_stage2.npz")
        pf, aux = s2["particle_features"], s2["particle_aux"]
        ch, coff, ev2 = s2["calo_hits_flat"], s2["calo_offsets"], s2["event_ids"]
        rz = np.load(Path(a.depth_dir) / f"shard_{sh:04d}_calo_rz.npz")["calo_rz"]
        pid2 = np.load(Path(a.graph_dir) / f"shard_{sh:04d}_pids.npz")["particle_ids"]
        # MATERIALISE. np.load on an .npz returns a LAZY NpzFile and every g["k"] access
        # decompresses the WHOLE array (8.5M elements here). Job 12891198 left three such accesses
        # inside the per-shower loop and did not finish 100k of 660k showers in an hour.
        _g = np.load(Path(a.graph_dir) / f"shard_{sh:04d}_graph.npz")
        g = {k: _g[k] for k in ("event_id", "particle_id", "parent_id", "pdg_id", "charge",
                                "mass", "energy", "vx", "vy", "vz", "px", "py", "pz")}
        _g.close()
        assert len(rz) == len(ch), f"depth sidecar {len(rz)} vs calo_hits_flat {len(ch)}"

        gkey = (g["event_id"].astype(np.int64) << 32) | g["particle_id"].astype(np.int64)
        go = np.argsort(gkey); gks = gkey[go]
        def grow(ev, pid):
            k = (ev.astype(np.int64) << 32) | pid.astype(np.int64)
            i = np.clip(np.searchsorted(gks, k), 0, len(gks) - 1)
            return np.where(gks[i] == k, go[i], -1)

        gvr = np.hypot(g["vx"], g["vy"]); gaz = np.abs(g["vz"])
        g_inside = (gvr >= R0) | (gaz >= Z0)

        ncell = np.diff(coff)
        # SELECT ON THE ANCESTOR'S SPECIES, not the depositor's. An e± fragment usually belongs to a
        # photon or pion shower, so filtering depositors by class then lifting would key groups by
        # mixed-species ancestors while labelling them with the ancestor pdg. We want "showers whose
        # INCIDENT particle is of this class", so take ALL depositors and filter groups afterwards.
        sel = np.where(ncell >= 1)[0]
        print(f"  shard {sh}: {len(sel):,} depositors (all classes)", flush=True)

        # ---- map each depositor to its calo-incident ancestor (graph row) ----
        my = grow(ev2[sel], pid2[sel])
        if my.min() < 0:
            raise SystemExit("a stage2 depositor is missing from the graph — join is broken")
        anc = my.copy()
        if not a.no_reattribute:
            live = g_inside[anc]
            for _ in range(32):
                if not live.any():
                    break
                nxt = grow(g["event_id"][anc[live]], g["parent_id"][anc[live]])
                ok = nxt >= 0
                idx = np.where(live)[0][ok]
                anc[idx] = nxt[ok]
                live[np.where(live)[0][~ok]] = False
                live[idx] = g_inside[anc[idx]]
            print(f"    lifted {(anc != my).mean():.3f} of depositors; "
                  f"still born-inside {g_inside[anc].mean():.4f}", flush=True)

        # ---- keep only ancestors of the requested class ----
        keep = np.isin(pdg_to_class(g["pdg_id"][anc].astype(np.int64)), a.pdg_class)
        sel, anc = sel[keep], anc[keep]
        print(f"    -> {len(sel):,} depositors under ancestors of class {a.pdg_class}", flush=True)

        # ---- VECTORISED expand + dedup. A per-shower np.unique(axis=0) over ~1M showers does not
        # finish; do one global sort instead. Cell identity is packed into an int64 from
        # (eta, phi, detector) -- (eta,phi) is effectively cell-unique (job 12880353). ----
        cnt = (coff[sel + 1] - coff[sel]).astype(np.int64)
        row_of = np.repeat(np.arange(len(sel)), cnt)                 # which depositor each cell came from
        # cheap expand: repeat each depositor's start offset and add a within-run counter, instead
        # of a 4.4M-element Python comprehension of np.arange
        starts = coff[sel].astype(np.int64)
        flat = (np.repeat(starts, cnt)
                + (np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)))
        hits_a, rz_a = ch[flat], rz[flat]
        ck = ((np.rint((hits_a[:, CH_ETA].astype(np.float64) + 10.0) * 1e5).astype(np.int64) << 33)
              | (np.rint((hits_a[:, CH_PHI].astype(np.float64) + 4.0) * 1e5).astype(np.int64) << 4)
              | hits_a[:, CH_DET].astype(np.int64))
        grp = anc[row_of]                                            # ancestor graph row per cell
        o = np.lexsort((ck, grp))
        grp_s, ck_s = grp[o], ck[o]
        newcell = np.empty(len(o), bool); newcell[0] = True
        newcell[1:] = (grp_s[1:] != grp_s[:-1]) | (ck_s[1:] != ck_s[:-1])
        cell_id = np.cumsum(newcell) - 1                             # deduped cell index
        e_all = np.bincount(cell_id, weights=np.exp(hits_a[o, CH_LOGE].astype(np.float64)))
        firsts = o[newcell]                                          # a representative row per cell
        gsh = grp_s[newcell]                                         # ancestor row per deduped cell
        newsh = np.empty(len(gsh), bool); newsh[0] = True
        newsh[1:] = gsh[1:] != gsh[:-1]
        sh_id = np.cumsum(newsh) - 1
        sh_start = np.where(newsh)[0]
        n_per = np.diff(np.concatenate([sh_start, [len(gsh)]]))
        arows = gsh[newsh]
        # DISTINCT DEPOSITORS merged into each shower. This counted deduped CELLS -- the predicate
        # `row_of >= 0` is true for every row -- so every build so far reported "merged
        # sources/shower" exactly equal to cells/shower (19.82 and 19.82 on the e+- production
        # slice). No model or metric reads the field, but it is part of the slice's data contract
        # and the number was being used to describe how much merging re-attribution does.
        # Count unique (ancestor, depositor) pairs with a second grp-major sort, so shower indices
        # line up with the cell grouping above.
        o2 = np.lexsort((row_of, grp))
        g2, r2 = grp[o2], row_of[o2]
        new2 = np.empty(len(o2), bool); new2[0] = True
        new2[1:] = (g2[1:] != g2[:-1]) | (r2[1:] != r2[:-1])
        sh_id2 = np.cumsum(np.concatenate([[True], g2[1:] != g2[:-1]])) - 1
        nsrc = np.bincount(sh_id2, weights=new2.astype(float),
                           minlength=len(arows)).astype(int)
        print(f"    {len(sel):,} depositors -> {len(arows):,} showers "
              f"(ratio {len(arows)/max(len(sel),1):.3f}); cells {len(hits_a):,} -> {len(gsh):,} "
              f"after dedup ({len(hits_a)/max(len(gsh),1):.3f}x)", flush=True)

        # ANCHOR + kinematics for every ancestor AT ONCE. Was one core_anchor() call per shower on
        # size-1 arrays -- ~660k numpy calls, and the reason job 12887279 hit its wall.
        apx, apy, apz = g["px"][arows], g["py"][arows], g["pz"][arows]
        a_pt = np.hypot(apx, apy)
        a_eta_p = np.arcsinh(np.clip(apz / np.clip(a_pt, 1e-9, None), -30, 30))
        a_phi_p = np.arctan2(apy, apx)
        if a.core_anchor != "none":
            AE, AP, AM = core_anchor(a_pt, a_phi_p, a_eta_p, g["charge"][arows], g["vx"][arows],
                                     g["vy"][arows], g["vz"][arows], R0, Z0, kind=a.core_anchor)
        else:
            AE = np.zeros(len(arows)); AP = np.zeros(len(arows)); AM = np.zeros(len(arows), np.int8)
        a_logE = np.log(np.clip(g["energy"][arows], 1e-6, None))
        a_vr = np.hypot(g["vx"][arows], g["vy"][arows])
        a_cls = pdg_to_class(g["pdg_id"][arows].astype(np.int64)).astype(np.float32)
        a_q, a_m, a_vz = g["charge"][arows], g["mass"][arows], g["vz"][arows]
        a_ev, a_E = g["event_id"][arows], g["energy"][arows]

        c_eta_all, c_phi_all = hits_a[firsts, CH_ETA], hits_a[firsts, CH_PHI]
        det_all = hits_a[firsts, CH_DET].astype(np.int8)   # HOIST: gathering this inside the
        # per-shower loop is O(n_cells) PER SHOWER -- 1.67M showers x 48M cells stalled job 13035236
        # for 41 min without completing a single 50k progress block.
        dep_all, lay_all, dloc_all, reg_all = depth_and_layer(det_all, rz_a[firsts, 0],
                                           rz_a[firsts, 1], R0, Z0)
        pick = np.arange(len(arows))
        if a.max_per_class:
            # PER-CLASS quota, matching build_calo_slice.py. A flat subsample of the whole shard
            # keeps the natural mixture, in which e+- are ~57% of all calo cells, so the majority
            # classes set the shared heads and the rare ones never get seen. Quota is per shard so
            # the cap is the total across --shards, again as v1 defines it.
            quota = max(a.max_per_class // max(len(a.shards), 1), 1)
            keeps = []
            for c in np.unique(a_cls):
                cand = np.where(a_cls == c)[0]
                if len(cand) > quota:
                    cand = rng.choice(cand, quota, replace=False)
                keeps.append(cand)
            pick = np.sort(np.concatenate(keeps))
            print(f"    per-class cap {quota}/shard -> {len(pick):,} of {len(arows):,} showers",
                  flush=True)
        import time as _t; _t0 = _t.time()
        for _ki, k in enumerate(pick):
            if _ki % 50000 == 0:
                print(f"      shower {_ki:,}/{len(pick):,}  ({_t.time()-_t0:.0f}s)", flush=True)
            s0, n = sh_start[k], n_per[k]
            arow = int(arows[k])
            e = e_all[s0:s0 + n]
            if n < a.min_cells or e.sum() <= 0:
                continue
            c_eta, c_phi = c_eta_all[s0:s0 + n], c_phi_all[s0:s0 + n]
            dep, lay = dep_all[s0:s0 + n], lay_all[s0:s0 + n]
            dloc, reg = dloc_all[s0:s0 + n], reg_all[s0:s0 + n]
            dets = det_all[s0:s0 + n]

            p_eta, p_phi = float(a_eta_p[k]), float(a_phi_p[k])
            d_eta = (c_eta - (p_eta + AE[k])).astype(np.float32)
            d_phi = wrap_pi(c_phi - (p_phi + AP[k])).astype(np.float32)
            core_eta, core_phi = float(d_eta.mean()), float(d_phi.mean())
            q_eta, q_phi = d_eta - core_eta, d_phi - core_phi
            e_coord = np.log(np.clip(e, 1e-12, None)).astype(np.float32)

            PTS.append(np.stack([q_eta, q_phi, dep, e_coord], 1).astype(np.float32))
            LAY.append(lay); DLOC.append(dloc); REG.append(reg); DET.append(dets)
            C.append([np.log(max(float(a_pt[k]), 1e-9)), p_eta, float(a_logE[k]),
                      float(a_q[k]), float(a_m[k]), float(a_vr[k]), float(a_vz[k])])
            G.append([np.log(e.sum()), np.log(n), core_eta, core_phi])
            PDG.append(float(a_cls[k]))
            ANC.append([float(AE[k]), float(AP[k])]); AMODE.append(int(AM[k]))
            NSRC.append(int(nsrc[k])); OFF.append(OFF[-1] + n)
            EVID.append(int(a_ev[k])); PPHI.append(p_phi); ETRUE.append(float(a_E[k]))

    cont = np.asarray(C, np.float32); glob = np.asarray(G, np.float32)
    pts = np.concatenate(PTS).astype(np.float32); lay = np.concatenate(LAY).astype(np.int16)
    dloc = np.concatenate(DLOC).astype(np.float32)
    reg = np.concatenate(REG).astype(np.int8); dets = np.concatenate(DET).astype(np.int8)
    off = np.asarray(OFF, np.int32)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, cont=cont, pdg=np.asarray(PDG, np.float32), glob=glob,
                        points_flat=pts, point_layer=lay, offsets=off,
                        point_depth_local=dloc, point_region=reg, point_det=dets,
                        n_src=np.asarray(NSRC, np.int32),
                        event_id=np.asarray(EVID, np.int32),
                        p_phi=np.asarray(PPHI, np.float32),
                        E_true=np.asarray(ETRUE, np.float32),
                        anchor=np.asarray(ANC, np.float32),
                        anchor_mode=np.asarray(AMODE, np.int8),
                        core_anchor=a.core_anchor, energy_mode="abs",
                        reattributed=(not a.no_reattribute),
                        cont_mean=cont.mean(0), cont_std=cont.std(0) + 1e-6,
                        glob_mean=glob.mean(0), glob_std=glob.std(0) + 1e-6,
                        pts_mean=pts.mean(0), pts_std=pts.std(0) + 1e-6)
    n = np.diff(off)
    print(f"\nwrote {out}\n  showers {len(n):,}  cells {len(pts):,}  cells/shower "
          f"mean {n.mean():.2f} median {np.median(n):.0f}")
    print(f"  distinct depositors merged/shower: mean {np.mean(NSRC):.2f}  "
          f"max {max(NSRC)}  frac unmerged {np.mean(np.asarray(NSRC)==1):.3f}")
    for _rlab, _rk in (("barrel-ECAL", 0), ("endcap-ECAL", 1), ("barrel-HCAL", 2), ("endcap-HCAL", 3)):
        _m = reg == _rk
        if _m.any():
            print(f"  region {_rlab:11s} {_m.mean():.4f} of cells   depth_local p1/50/99: "
                  f"{np.percentile(dloc[_m],1):7.1f} /{np.percentile(dloc[_m],50):8.1f} /"
                  f"{np.percentile(dloc[_m],99):8.1f} mm   <logE> {pts[_m,3].mean():.3f}")
    print(f"  layer index range {lay[lay>=0].min() if (lay>=0).any() else -1}"
          f"..{lay.max()} (globally unique; -1 = barrel)")
    print(f"  depth p5/50/95: {np.percentile(pts[:,2],5):.1f} / {np.percentile(pts[:,2],50):.1f} / "
          f"{np.percentile(pts[:,2],95):.1f} mm   layer defined {np.mean(lay>=0):.3f}")
    print(f"  anchor branch: " + "  ".join(
        f"{k}:{np.mean(np.asarray(AMODE)==v):.3f}" for k, v in
        [("barrel", 0), ("endcap", 1), ("turning", 2), ("none", 3)]))
    pdg_arr = np.asarray(PDG, np.int64)
    per_class = {int(c): int(v) for c, v in zip(*np.unique(pdg_arr, return_counts=True))}
    cells_per_class = {int(c): int(n[pdg_arr == c].sum()) for c in per_class}
    print("  showers per PDG class: " + ", ".join(
        f"{c}:{v:,}" for c, v in sorted(per_class.items(), key=lambda kv: -kv[1])))
    summ = {"showers": int(len(n)), "cells": int(len(pts)), "shards": a.shards,
            "pdg_class": a.pdg_class, "max_per_class": a.max_per_class,
            "reattributed": (not a.no_reattribute), "core_anchor": a.core_anchor,
            "showers_per_class": per_class, "cells_per_class": cells_per_class,
            "cells_per_shower_mean": float(n.mean()),
            "layer_defined_frac": float(np.mean(lay >= 0))}
    Path(str(out) + ".summary.json").write_text(json.dumps(summ, indent=2))


if __name__ == "__main__":
    main()
